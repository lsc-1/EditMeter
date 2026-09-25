from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
import gc

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import tqdm
import torch
from loguru import logger
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, logging as hf_logging

#data_loader_triplet2  对 confidence进行了修改
from lib.data_loader_triplet import (
    DEGREE_METHODS,
    SingleTextEditDataset,
    TripletEditDataset,

    # Prepare / dataset utilities
    convert_triplets_to_single_samples,
    is_triplet_format,
    load_json_list,
    maybe_sample_triplets,
    prepare_triplets_with_offline_degree,
    save_json_list,

    # Raw Top-k pseudo-triplet utilities
    load_label_only_samples,
    rank_topk_anchor_candidates,
    safe_text,

    # Semantic / likelihood retrieval utilities
    load_metric_model_and_tokenizer,
    compute_lm_pool_features_batch,
    fit_token_ll_feature_scaler,
    transform_token_ll_feature_map,
)


from lib.exp_triplet import seed_everything,compute_edited_degree_stats
from lib.models.triplet_projection2 import (
    RobertaTripletProjectionRegressor,
)
from lib.training.triplet_projection_common import (
    evaluate_two_models_and_save_test,
    select_score_from_metrics,
    train_triplet_projection_epoch,
    validate_triplet_projection,
)

hf_logging.set_verbosity_error()


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"yes", "true", "t", "1", "y"}:
        return True
    if value in {"no", "false", "f", "0", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def build_optimizer(model, learning_rate, weight_decay):
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    groups = [
        {
            "params": [parameter for name, parameter in named if not any(key in name for key in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [parameter for name, parameter in named if any(key in name for key in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return AdamW(groups, lr=learning_rate)


def setup_runtime(args):
    seed_everything(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning(f"CUDA unavailable; switching {args.device} to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)
    log_parent = os.path.dirname(args.log_file)
    if log_parent:
        os.makedirs(log_parent, exist_ok=True)
    logger.add(args.log_file)
    logger.info("arguments:\n" + json.dumps(vars(args), ensure_ascii=False, indent=2))

    tokenizer = AutoTokenizer.from_pretrained(
        args.backbone_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
    return device, tokenizer


def build_model(args, device):
    return RobertaTripletProjectionRegressor(
        model_path=args.backbone_path,
        proj_dim=args.proj_dim,
        num_labels=3,
        dropout=args.dropout,
        proto_weight=args.proto_weight,
        proto_momentum=args.proto_momentum,
        teacher_kappa_min=args.teacher_kappa_min,
        teacher_kappa_max=args.teacher_kappa_max,
        teacher_y_eps=args.teacher_y_eps,
        use_lora=args.use_lora,
        freeze_backbone=args.freeze_backbone,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    ).to(device)


def build_models(args, device):
    seed_everything(args.seed)
    cls_model = build_model(args, device)
    if not args.use_separate_degree_model:
        return cls_model, None
    seed_everything(args.seed)
    return cls_model, build_model(args, device)


def load_checkpoint(model, path, device, role):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{role} checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=True)
    epoch = checkpoint.get("epoch", checkpoint.get("best_epoch", -1)) if isinstance(checkpoint, dict) else -1
    score = checkpoint.get(
        "best_score",
        checkpoint.get("best_cls_score", checkpoint.get("best_degree_score", -float("inf"))),
    ) if isinstance(checkpoint, dict) else -float("inf")
    logger.info(f"Loaded {role} checkpoint {path}: epoch={epoch}, score={score}")
    return model, int(epoch), float(score)


# def prepare_triplet_file(path, args, device, split_name, sample_limit=0):
#     data = load_json_list(path)
#     if not is_triplet_format(data):
#         return data, False
#     if sample_limit > 0:
#         data = maybe_sample_triplets(data, sample_limit, args.seed)
#     prepared = prepare_triplets_with_offline_degree(data, args, device, split_name)
#     save_json_list(os.path.join(args.output_dir, f"{split_name}_triplets_with_degree.json"), prepared)
#     return prepared, True


def prepare_triplet_file(
    path,
    args,
    device,
    split_name,
    sample_limit=0,
):
    data = load_json_list(path)

    if not is_triplet_format(data):
        logger.warning(
            f"[{split_name}] Input data is not triplet format: {path}"
        )
        return data, False

    if sample_limit > 0:
        data = maybe_sample_triplets(
            data,
            sample_limit,
            args.seed,
        )

    # 为每条三元组计算 edited_degree
    prepared = prepare_triplets_with_offline_degree(
        data,
        args,
        device,
        split_name,
    )

    # 保存带 degree 的原始列表
    data_save_path = os.path.join(
        args.output_dir,
        f"{split_name}_triplets_with_degree.json",
    )
    save_json_list(data_save_path, prepared)

    # 对所有 edited_degree 计算全局统计量
    stats = compute_edited_degree_stats(
        prepared,
        split_name,
    )
    stats["degree_method"] = args.degree_method

    # 单独保存统计结果，避免破坏原列表结构
    stats_save_path = os.path.join(
        args.output_dir,
        f"{split_name}_triplets_with_degree_stats.json",
    )

    with open(stats_save_path, "w", encoding="utf-8") as file:
        json.dump(
            stats,
            file,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(
        f"[{split_name}] Saved prepared data to: "
        f"{os.path.abspath(data_save_path)}"
    )

    logger.info(
        
        f"[{split_name}] edited_degree statistics: "
        f"count={stats['valid_edited_degree_count']}, "
        f"mean={stats['edited_degree_mean']}, "
        f"min={stats['edited_degree_min']}, "
        f"max={stats['edited_degree_max']}, "
        f"variance={stats['edited_degree_variance']}, "
        f"std={stats['edited_degree_std']}"
    )

    logger.info(
        f"[{split_name}] Saved statistics to: "
        f"{os.path.abspath(stats_save_path)}"
    )

    return prepared, True

def build_train_val_loaders(
    args,
    tokenizer,
    device,
):
    """
    Train mode:

    1. Load the prepared training triplets generated by mode=prepare.
    2. Load the original single-text validation samples directly.
    3. Do not rebuild pseudo triplets.
    4. Do not recompute training or validation edit degrees.
    """
    del device  # 当前函数中不再需要device，仅保持接口兼容

    # ---------------------------------------------------------
    # 1. 加载prepare阶段生成的最终训练三元组
    # ---------------------------------------------------------
    train_path = os.path.join(
        args.output_dir,
        "train_triplets_with_degree.json",
    )

    if not os.path.isfile(train_path):
        raise FileNotFoundError(
            "Prepared training file not found: "
            f"{train_path}. Run --mode prepare first."
        )

    train_triplets = load_json_list(
        train_path
    )

    if not is_triplet_format(train_triplets):
        raise ValueError(
            "Prepared training data must use "
            "human/edited/ai triplet format."
        )

    # TripletEditDataset要求每个训练三元组存在edited_degree。
    train_dataset = TripletEditDataset(
        train_triplets,
        tokenizer,
        args.backbone_max_length,
    )

    # ---------------------------------------------------------
    # 2. 直接加载原始单条样本验证集
    # ---------------------------------------------------------
    validation_path = args.validation_file

    if not os.path.isfile(validation_path):
        raise FileNotFoundError(
            "Validation file not found: "
            f"{validation_path}"
        )

    val_data = load_json_list(
        validation_path
    )

    # 你已经确认验证集一直是单条样本，因此这里不需要：
    # - prepare_triplet_file
    # - prepare_triplets_with_offline_degree
    # - convert_triplets_to_single_samples
    if is_triplet_format(val_data):
        raise ValueError(
            "validation_file is expected to contain single-text samples, "
            "but triplet-format data was found."
        )

    val_dataset = SingleTextEditDataset(
        val_data,
        tokenizer,
        args.backbone_max_length,
    )

    # ---------------------------------------------------------
    # 3. 构建DataLoader
    # ---------------------------------------------------------
    generator = torch.Generator().manual_seed(
        args.seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    logger.info(
        f"Loaded prepared training data: "
        f"path={os.path.abspath(train_path)}, "
        f"triplets={len(train_dataset)}"
    )

    logger.info(
        f"Loaded raw single-text validation data: "
        f"path={os.path.abspath(validation_path)}, "
        f"samples={len(val_dataset)}"
    )

    return train_loader, val_loader

def build_test_loader(args, tokenizer, device):
    test_data, is_triplet = prepare_triplet_file(args.test_file, args, device, "test")
    samples = convert_triplets_to_single_samples(test_data) if is_triplet else test_data
    dataset = SingleTextEditDataset(samples, tokenizer, args.backbone_max_length)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )


def build_test_loader2(
    args,
    tokenizer,
    device,
):
    test_path = os.path.join(
        args.output_dir,
        "test_triplets_with_degree.json",
    )

    if not os.path.isfile(test_path):
        raise FileNotFoundError(
            f"Prepared test file not found: "
            f"{test_path}. "
            "Run --mode prepare first."
        )

    test_data = load_json_list(
        test_path
    )

    samples = (
        convert_triplets_to_single_samples(
            test_data
        )
        if is_triplet_format(test_data)
        else test_data
    )

    dataset = SingleTextEditDataset(
        samples,
        tokenizer,
        args.backbone_max_length,
    )

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )


def checkpoint_payload(model, args, epoch, score, role):
    return {
        "epoch": int(epoch),
        "best_epoch": int(epoch),
        "best_score": float(score),
        "model_state_dict": model.state_dict(),
        "backbone_path": args.backbone_path,
        "degree_method": args.degree_method,
        "selection_type": args.select_metric if role == "classification" else args.degree_select_metric,
        "model_role": role,
        "args": vars(args),
    }


def main_train(args):
    device, tokenizer = setup_runtime(args)
    train_loader, val_loader = build_train_val_loaders(args, tokenizer, device)
    cls_model, degree_model = build_models(args, device)
    cls_optimizer = build_optimizer(cls_model, args.lr, args.weight_decay)
    degree_optimizer = (
        build_optimizer(degree_model, args.lr, args.weight_decay)
        if degree_model is not None
        else None
    )

    # best_cls = {"score": -float("inf"), "epoch": -1, "state": None}
    # best_degree = {"score": -float("inf"), "epoch": -1, "state": None}
    best_cls = {"score": -float("inf"), "epoch": -1}
    best_degree = {"score": -float("inf"), "epoch": -1}
    
    for epoch in range(1, args.epochs + 1):
        started = datetime.now()
        cls_loss = train_triplet_projection_epoch(
            cls_model,
            cls_optimizer,
            train_loader,
            device,
            args,
            epoch,
            loss_mode=args.cls_loss_mode,
            tag="CLS",
        )
        cls_metrics, cls_degree_metrics, _ = validate_triplet_projection(cls_model, val_loader, device)
        cls_score = select_score_from_metrics(cls_metrics, cls_degree_metrics, args.select_metric)
        # logger.info(
        #     f"epoch={epoch} cls_loss={cls_loss:.4f} macro_f1={cls_metrics['macro_f1']:.4f} "
        #     f"degree={cls_degree_metrics} score={cls_score:.4f}"
        # )


        
        logger.info(
            f"epoch {epoch} CLS "
            f"train_loss={cls_loss:.4f}, "
        )
        
        logger.info(
            f"val_acc={cls_metrics['accuracy']:.4f}, "
            f"val_bal_acc={cls_metrics['balanced_accuracy']:.4f}, "
            f"val_macro_f1={cls_metrics['macro_f1']:.4f}, "
            f"val_human_f1={cls_metrics.get('human_f1', 0.0):.4f}, "
            f"val_edited_f1={cls_metrics.get('edited_f1', 0.0):.4f}, "
            f"val_ai_f1={cls_metrics.get('ai_f1', 0.0):.4f}, "
        )

        logger.info(
            f"degree={cls_degree_metrics}, "
        )


        logger.info(
            f"Class select_score={cls_score:.4f}"
        )
        
        if cls_score > best_cls["score"]:
            best_cls = {
                "score": cls_score,
                "epoch": epoch,
            }

            cls_best_path = os.path.join(
                args.output_dir,
                args.cls_output_name,
            )

            torch.save(
                checkpoint_payload(
                    cls_model,
                    args,
                    epoch,
                    cls_score,
                    "classification",
                ),
                cls_best_path,
            )

            logger.info(
                f"Saved best classification checkpoint: "
                f"epoch={epoch}, score={cls_score:.4f}, "
                f"path={cls_best_path}"
            )
        # if cls_score > best_cls["score"]:
        #     best_cls = {
        #         "score": cls_score,
        #         "epoch": epoch,
        #         "state": {key: value.detach().cpu().clone() for key, value in cls_model.state_dict().items()},
        #     }
        #     torch.save(
        #         checkpoint_payload(cls_model, args, epoch, cls_score, "classification"),
        #         os.path.join(args.output_dir, args.cls_output_name),
        #     )

        if degree_model is not None:
            degree_loss = train_triplet_projection_epoch(
                degree_model,
                degree_optimizer,
                train_loader,
                device,
                args,
                epoch,
                loss_mode="degree",
                tag="DEGREE",
            )
            degree_cls_metrics, degree_metrics, _ = validate_triplet_projection(
                degree_model, val_loader, device
            )
            degree_score = select_score_from_metrics(
                degree_cls_metrics, degree_metrics, args.degree_select_metric
            )
            logger.info(
                f"epoch={epoch} degree_loss={degree_loss:.4f} metrics={degree_metrics} "
                f"score={degree_score:.4f}"
            )
            if degree_score > best_degree["score"]:
                best_degree = {
                    "score": degree_score,
                    "epoch": epoch,
                }

                degree_best_path = os.path.join(
                    args.output_dir,
                    args.degree_output_name,
                )

                torch.save(
                    checkpoint_payload(
                        degree_model,
                        args,
                        epoch,
                        degree_score,
                        "degree",
                    ),
                    degree_best_path,
                )

                logger.info(
                    f"Saved best degree checkpoint: "
                    f"epoch={epoch}, score={degree_score:.4f}, "
                    f"path={degree_best_path}"
                )

        if args.save_every_epoch:
            torch.save(
                checkpoint_payload(cls_model, args, epoch, cls_score, "classification"),
                os.path.join(args.output_dir, f"epoch{epoch}_{args.cls_output_name}"),
            )
        logger.info(f"epoch {epoch} duration: {datetime.now() - started}")

    cls_best_path = os.path.join(
        args.output_dir,
        args.cls_output_name,
    )

    if not os.path.isfile(cls_best_path):
        raise RuntimeError(
            f"Best classification checkpoint was not saved: "
            f"{cls_best_path}"
        )

    cls_model, _, _ = load_checkpoint(
        cls_model,
        cls_best_path,
        device,
        "classification",
    )

    if degree_model is None:
        degree_model = cls_model
        best_degree = best_cls
    else:
        degree_best_path = os.path.join(
            args.output_dir,
            args.degree_output_name,
        )

        if not os.path.isfile(degree_best_path):
            raise RuntimeError(
                f"Best degree checkpoint was not saved: "
                f"{degree_best_path}"
            )

        degree_model, _, _ = load_checkpoint(
            degree_model,
            degree_best_path,
            device,
            "degree",
        )

    torch.save(
        checkpoint_payload(cls_model, args, best_cls["epoch"], best_cls["score"], "classification"),
        os.path.join(args.output_dir, "triplet_projection_cls_model.pt"),
    )
    torch.save(
        checkpoint_payload(
            degree_model,
            args,
            best_degree["epoch"],
            best_degree["score"],
            "degree" if args.use_separate_degree_model else "classification_as_degree",
        ),
        os.path.join(args.output_dir, "triplet_projection_degree_model.pt"),
    )
    tokenizer.save_pretrained(args.output_dir)


def main_test(args):
    device, tokenizer = setup_runtime(args)
    test_loader = build_test_loader(args, tokenizer, device)
    cls_model, degree_model = build_models(args, device)
    cls_model, cls_epoch, cls_score = load_checkpoint(
        cls_model,
        os.path.join(args.output_dir, args.cls_output_name),
        device,
        "classification",
    )
    if args.use_separate_degree_model:
        degree_model, degree_epoch, degree_score = load_checkpoint(
            degree_model,
            os.path.join(args.output_dir, args.degree_output_name),
            device,
            "degree",
        )
    else:
        degree_model, degree_epoch, degree_score = cls_model, cls_epoch, cls_score

    return evaluate_two_models_and_save_test(
        cls_model=cls_model.eval(),
        degree_model=degree_model.eval(),
        test_loader=test_loader,
        device=device,
        output_dir=args.output_dir,
        backbone_path=args.backbone_path,
        best_cls_epoch=cls_epoch,
        best_cls_score=cls_score,
        best_degree_epoch=degree_epoch,
        best_degree_score=degree_score,
    )





def build_raw_pseudo_triplet_for_edited_sample(
    edited_item,
    human_pool,
    ai_pool,
    args,
    embedding_map=None,
    ll_feature_map=None,
):
    """
    只通过 Top-k 检索构建原始虚拟三元组。

    不计算 edited_degree。
    不进行 AI 锚点修正。
    不按照 degree/confidence/alignment 筛选。
    """
    retrieval_method = str(
        getattr(
            args,
            "pseudo_retrieval_method",
            "levenshtein",
        )
    )

    if retrieval_method == "same":
        retrieval_method = str(args.degree_method)

    topk = max(
        int(getattr(args, "pseudo_topk", 10)),
        1,
    )

    top_humans = rank_topk_anchor_candidates(
        edited_item=edited_item,
        anchor_pool=human_pool,
        retrieval_method=retrieval_method,
        topk=topk,
        embedding_map=embedding_map,
        ll_feature_map=ll_feature_map,
    )

    top_ais = rank_topk_anchor_candidates(
        edited_item=edited_item,
        anchor_pool=ai_pool,
        retrieval_method=retrieval_method,
        topk=topk,
        embedding_map=embedding_map,
        ll_feature_map=ll_feature_map,
    )

    if not top_humans or not top_ais:
        return None

    edited_text = safe_text(
        edited_item["text"]
    )

    pair_candidates = []

    for human_item in top_humans:
        for ai_item in top_ais:
            human_text = safe_text(
                human_item["text"]
            )
            ai_text = safe_text(
                ai_item["text"]
            )

            if (
                not human_text
                or not edited_text
                or not ai_text
            ):
                continue

            if (
                human_text == edited_text
                or ai_text == edited_text
                or human_text == ai_text
            ):
                continue

            human_distance = float(
                human_item[
                    "_pseudo_retrieval_distance"
                ]
            )

            ai_distance = float(
                ai_item[
                    "_pseudo_retrieval_distance"
                ]
            )

            retrieval_distance = (
                0.5
                * (
                    human_distance
                    + ai_distance
                )
            )

            pair_candidates.append({
                "human_item": human_item,
                "ai_item": ai_item,
                "retrieval_distance":
                    retrieval_distance,
            })

    if not pair_candidates:
        return None

    # 这里只按照检索距离选择，不使用 degree。
    best = min(
        pair_candidates,
        key=lambda row: row[
            "retrieval_distance"
        ],
    )

    human_item = best["human_item"]
    ai_item = best["ai_item"]

    return {
        "human": safe_text(
            human_item["text"]
        ),
        "edited": edited_text,
        "ai": safe_text(
            ai_item["text"]
        ),

        "is_pseudo_triplet": True,
        "pseudo_retrieval_method":
            retrieval_method,
        "pseudo_topk": topk,

        "pseudo_selected_retrieval_distance":
            float(best["retrieval_distance"]),

        "pseudo_human_retrieval_distance":
            float(
                human_item[
                    "_pseudo_retrieval_distance"
                ]
            ),

        "pseudo_ai_retrieval_distance":
            float(
                ai_item[
                    "_pseudo_retrieval_distance"
                ]
            ),

        "pseudo_human_retrieval_components":
            human_item.get(
                "_pseudo_retrieval_components",
                {},
            ),

        "pseudo_ai_retrieval_components":
            ai_item.get(
                "_pseudo_retrieval_components",
                {},
            ),

        "edited_source_id":
            edited_item.get("source_id", ""),
        "edited_pair_id":
            edited_item.get("pair_id", ""),

        "pseudo_human_source_id":
            human_item.get("source_id", ""),
        "pseudo_human_pair_id":
            human_item.get("pair_id", ""),

        "pseudo_ai_source_id":
            ai_item.get("source_id", ""),
        "pseudo_ai_pair_id":
            ai_item.get("pair_id", ""),
    }
def build_raw_topk_pseudo_triplets_from_extra_file(
    extra_train_file,
    args,
    device,
):
    """
    从 extra_train_file 构建原始虚拟三元组。

    返回值尚未计算编辑度。
    """
    if not extra_train_file:
        return []

    samples = load_label_only_samples(
        extra_train_file
    )

    human_pool = [
        item
        for item in samples
        if item["label_id"] == 0
    ]

    edited_pool = [
        item
        for item in samples
        if item["label_id"] == 1
    ]

    ai_pool = [
        item
        for item in samples
        if item["label_id"] == 2
    ]

    if (
        not human_pool
        or not edited_pool
        or not ai_pool
    ):
        raise ValueError(
            "extra_train_file must contain "
            "human, edited and ai samples."
        )

    retrieval_method = str(
        getattr(
            args,
            "pseudo_retrieval_method",
            "levenshtein",
        )
    )

    if retrieval_method == "same":
        retrieval_method = str(
            args.degree_method
        )

    embedding_map = None
    ll_feature_map = None

    # 这里只为检索计算必要特征。
    if retrieval_method in {
        "semantic",
        "likelihood",
        "fusion",
    }:
        metric_model, metric_tokenizer = (
            load_metric_model_and_tokenizer(
                args.metric_model_path,
                device,
            )
        )

        unique_texts = []
        seen = set()

        for item in samples:
            text = safe_text(item["text"])
            if text and text not in seen:
                seen.add(text)
                unique_texts.append(text)

        (
            embedding_map,
            _,
            raw_ll_feature_map,
        ) = compute_lm_pool_features_batch(
            texts=unique_texts,
            model=metric_model,
            tokenizer=metric_tokenizer,
            max_length=args.metric_max_length,
            batch_size=args.metric_batch_size,
            likelihood_min_tokens=(
                args.likelihood_min_tokens
            ),
        )

        if retrieval_method in {
            "likelihood",
            "fusion",
        }:
            # 该 scaler 只服务于检索，不是最终 degree scaler。
            retrieval_scaler = (
                fit_token_ll_feature_scaler(
                    raw_ll_feature_map
                )
            )

            ll_feature_map = (
                transform_token_ll_feature_map(
                    raw_ll_feature_map,
                    retrieval_scaler,
                )
            )

        del metric_model, metric_tokenizer
        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    pseudo_triplets = []

    for edited_item in tqdm.tqdm(
        edited_pool,
        desc="Building raw top-k pseudo triplets",
    ):
        pseudo = (
            build_raw_pseudo_triplet_for_edited_sample(
                edited_item=edited_item,
                human_pool=human_pool,
                ai_pool=ai_pool,
                args=args,
                embedding_map=embedding_map,
                ll_feature_map=ll_feature_map,
            )
        )

        if pseudo is not None:
            pseudo_triplets.append(pseudo)

    max_pseudo = int(
        getattr(
            args,
            "max_pseudo_triplets",
            0,
        )
    )

    if max_pseudo > 0:
        pseudo_triplets = (
            pseudo_triplets[:max_pseudo]
        )

    return pseudo_triplets




def prepare_only(args):
    """
    1. 读取可选的 reference_file；
    2. 从 extra_train_file 构建原始 pseudo 三元组；
    3. 合并；
    4. 只要求合并后的 raw_train_triplets 非空；
    5. 对合并结果统一计算编辑度；
    6. 保存最终训练数据。
    """
    device, _ = setup_runtime(args)

    # ---------------------------------------------------------
    # 1. 读取 reference 原始三元组。
    #
    # reference 可以为空；真正需要非空的是最后合并的数据。
    # ---------------------------------------------------------
    reference_triplets = []

    reference_file = str(
        getattr(args, "reference_file", "")
    ).strip()

    if reference_file:
        if not os.path.isfile(reference_file):
            raise FileNotFoundError(
                f"reference_file not found: {reference_file}"
            )

        reference_triplets = load_json_list(
            reference_file
        )

        # 空列表允许；只有非空数据才检查三元组格式。
        if (
            reference_triplets
            and not is_triplet_format(reference_triplets)
        ):
            raise ValueError(
                "Non-empty reference_file must use "
                "human/edited/ai triplet format."
            )

        if (
            args.max_reference_file_triplets > 0
            and reference_triplets
        ):
            reference_triplets = maybe_sample_triplets(
                reference_triplets,
                args.max_reference_file_triplets,
                args.seed,
            )

    logger.info(
        f"Loaded reference triplets: "
        f"{len(reference_triplets)}"
    )

    # ---------------------------------------------------------
    # 2. 构建原始 pseudo 三元组。
    # ---------------------------------------------------------
    pseudo_triplets = []

    extra_train_file = str(
        getattr(args, "extra_train_file", "")
    ).strip()

    if extra_train_file:
        pseudo_triplets = (
            build_raw_topk_pseudo_triplets_from_extra_file(
                extra_train_file=extra_train_file,
                args=args,
                device=device,
            )
        )

        max_pseudo = int(
            getattr(args, "max_pseudo_triplets", 0)
        )

        if max_pseudo > 0:
            pseudo_triplets = pseudo_triplets[:max_pseudo]

        pseudo_max_ratio = float(
            getattr(args, "pseudo_max_ratio", 0.0)
        )

        # 比例限制只有在 reference 非空时才有定义。
        if (
            pseudo_max_ratio > 0.0
            and len(reference_triplets) > 0
        ):
            ratio_limit = int(
                len(reference_triplets)
                * pseudo_max_ratio
            )

            pseudo_triplets = pseudo_triplets[:ratio_limit]

            logger.info(
                f"Applied pseudo ratio cap: "
                f"reference={len(reference_triplets)}, "
                f"ratio={pseudo_max_ratio}, "
                f"limit={ratio_limit}"
            )

        elif (
            pseudo_max_ratio > 0.0
            and len(reference_triplets) == 0
        ):
            logger.warning(
                "reference_triplets is empty; "
                "pseudo_max_ratio is ignored because a "
                "pseudo/reference ratio is undefined."
            )

    # ---------------------------------------------------------
    # 3. 合并。
    # ---------------------------------------------------------
    raw_train_triplets = (
        reference_triplets
        + pseudo_triplets
    )

    logger.info(
        f"Raw training triplets: "
        f"reference={len(reference_triplets)}, "
        f"pseudo={len(pseudo_triplets)}, "
        f"combined={len(raw_train_triplets)}"
    )

    # 真正应该检查的是合并结果。
    if not raw_train_triplets:
        raise RuntimeError(
            "No training triplets are available after combining "
            "reference_triplets and pseudo_triplets. "
            f"reference={len(reference_triplets)}, "
            f"pseudo={len(pseudo_triplets)}."
        )

    # 进一步验证合并后的每条数据确实为三元组。
    if not is_triplet_format(raw_train_triplets):
        raise ValueError(
            "Combined raw_train_triplets must use "
            "human/edited/ai triplet format."
        )

    # ---------------------------------------------------------
    # 4. 保存 degree 计算前的原始组合。
    # ---------------------------------------------------------
    raw_save_path = os.path.join(
        args.output_dir,
        "train_triplets_raw_combined.json",
    )

    save_json_list(
        raw_save_path,
        raw_train_triplets,
    )

    # ---------------------------------------------------------
    # 5. 对整个合并训练集统一计算 degree。
    # ---------------------------------------------------------
    train_triplets = (
        prepare_triplets_with_offline_degree(
            raw_train_triplets,
            args,
            device,
            split_name="train",
        )
    )

    if not train_triplets:
        raise RuntimeError(
            "Degree preparation produced zero valid training triplets."
        )

    # ---------------------------------------------------------
    # 6. 保存最终训练文件。
    # ---------------------------------------------------------
    train_save_path = os.path.join(
        args.output_dir,
        "train_triplets_with_degree.json",
    )

    save_json_list(
        train_save_path,
        train_triplets,
    )

    train_stats = compute_edited_degree_stats(
        train_triplets,
        "train",
    )

    train_stats["degree_method"] = args.degree_method
    train_stats["reference_count"] = len(reference_triplets)
    train_stats["pseudo_count"] = len(pseudo_triplets)
    train_stats["raw_combined_count"] = len(raw_train_triplets)
    train_stats["prepared_count"] = len(train_triplets)

    stats_path = os.path.join(
        args.output_dir,
        "train_triplets_with_degree_stats.json",
    )

    with open(
        stats_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            train_stats,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    logger.info(
        f"Saved final training data: "
        f"{train_save_path}; "
        f"reference={len(reference_triplets)}, "
        f"pseudo={len(pseudo_triplets)}, "
        f"prepared={len(train_triplets)}"
    )

    # ---------------------------------------------------------
    # 7. Validation/test 复用训练阶段 scaler。
    # ---------------------------------------------------------
    prepare_triplet_file(
        args.validation_file,
        args,
        device,
        "validation",
    )

    prepare_triplet_file(
        args.test_file,
        args,
        device,
        "test",
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train a human/edited/AI classifier and Human-relative edit-degree regressor."
    )
    parser.add_argument("--mode", choices=["prepare", "train", "test"], default="train")
    
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--device", type=str, default="cuda:1")
    

    parser.add_argument("--reference_file", type=str, default="")
    parser.add_argument("--validation_file", type=str, default="")#""
    parser.add_argument("--test_file", type=str, default="")
    parser.add_argument("--max_reference_file_triplets", type=int, default=0)
    

    parser.add_argument("--extra_train_file",type=str,default="",)
    parser.add_argument("--max_pseudo_triplets",type=int,default=0 ,help="Maximum pseudo triplets; <=0 means no absolute cap.",)


    parser.add_argument("--degree_method", choices=["levenshtein","jaccard","semantic","likelihood","fusion"], default="fusion")
    
    parser.add_argument("--metric_model_path", type=str, default="pretrain_models/gpt-j-6b")
    parser.add_argument("--metric_max_length", type=int, default=1024)
    parser.add_argument("--metric_batch_size", type=int, default=4)
    parser.add_argument(
        "--ll_anchor_eps",
        type=float,
        default=1e-3,
        help="Minimum valid AI initial degree for likelihood anchor correction.",
    )
    parser.add_argument(
        "--likelihood_delta_tau",
        type=float,
        default=0.0,
        help=(
            "Shared tau for mapping standardized Human--Edited and "
            "Human--AI token-LL distances to [0,1): 1-exp(-distance/tau). "
            "Values <=0 use the adaptive tau fitted from training "
            "Human--Edited differences."
        ),
    )

    parser.add_argument(
        "--likelihood_min_tokens",
        type=int,
        default=16,
        help="Minimum scored tokens required for the three token-LL features.",
    )
    parser.add_argument(
        "--likelihood_scaler_filename",
        type=str,
        default="likelihood_token_feature_scaler.json",
        help="Training-only robust scaler reused by validation, test, and pseudo data.",
    )



    parser.add_argument("--pseudo_topk",type=int,default=10,help="Top-k human and top-k AI retrieval candidates per edited sample.",)
    parser.add_argument(
        "--pseudo_retrieval_method",
        choices=[
            "same",
            "levenshtein",
            "jaccard",
            "semantic",
            "likelihood",
            "fusion",
        ],
        default="same",
        help=(
            "Method used only for top-k retrieval. "
            "'same' uses degree_method."
        ),
    )
    
    parser.add_argument(
        "--pseudo_anchor_gap_weight",
        type=float,
        default=0.5,
        help=(
            "Selection weight for a large, stable AI initial degree. "
            "The historical argument name is retained for compatibility."
        ),
    )
    parser.add_argument(
        "--pseudo_alignment_weight",
        type=float,
        default=0.2,
        help="Selection weight for human->edited and human->AI alignment.",
    )
    parser.add_argument(
        "--pseudo_retrieval_weight",
        type=float,
        default=0.2,
        help="Selection weight for edited-to-anchor retrieval similarity.",
    )
    parser.add_argument(
        "--pseudo_min_alignment",
        type=float,
        default=0.0,
        help="Minimum AI alignment for a pseudo triplet.",
    )
    parser.add_argument(
        "--pseudo_min_ai_initial_degree",
        type=float,
        default=0.05,
        help=(
            "Reject AI anchors whose initial degree is too close to zero, "
            "because their correction scale would be unstable."
        ),
    )
    parser.add_argument(
        "--pseudo_degree_range_margin",
        type=float,
        default=0.05,
        help="Keep corrected pseudo degree inside [0, 1+margin].",
    )

    parser.add_argument("--pseudo_min_confidence",type=float,default=0.05,)
    parser.add_argument("--pseudo_confidence_scale",type=float,default=0.5,help="Down-weight pseudo-triplet degree confidence relative to real triplets.",)

    parser.add_argument("--pseudo_max_ratio",type=float,default=0.0,help=("Maximum pseudo/real triplet count ratio. ""<=0 disables this ratio cap."),)


    parser.add_argument("--backbone_path", type=str, default="pretrain_models/gpt-j-6b")
    parser.add_argument("--backbone_max_length", type=int, default=512)
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--lambda_cls", type=float, default=1.0)
    parser.add_argument("--lambda_degree", type=float, default=1.0)
    parser.add_argument("--lambda_axis", type=float, default=1.0)
    parser.add_argument("--cls_loss_mode", choices=["cls", "joint"], default="joint")

    parser.add_argument(
        "--select_metric",
        choices=["macro_f1", "edited_f1", "balanced_acc", "macro_f1_spearman", "macro_f1_minus_mae"],
        default="macro_f1_minus_mae",
    )
    parser.add_argument(
        "--degree_select_metric",
        choices=["degree_mae", "macro_f1_minus_mae"],
        default="degree_mae",
    )
    parser.add_argument("--use_separate_degree_model", type=str2bool, default=False)
    parser.add_argument("--save_every_epoch", type=str2bool, default=False)

    parser.add_argument("--teacher_kappa_min", type=float, default=2.0)
    parser.add_argument("--teacher_kappa_max", type=float, default=30.0)
    parser.add_argument("--teacher_y_eps", type=float, default=0.01)
    parser.add_argument("--proto_weight", type=float, default=0.3)
    parser.add_argument("--proto_momentum", type=float, default=0.95)

    parser.add_argument("--use_lora", type=str2bool, default=True)
    parser.add_argument("--freeze_backbone", type=str2bool, default=True)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--output_dir", type=str, default="save_model/triplet_edit_degree_fuse")
    parser.add_argument("--cls_output_name", type=str, default="best_triplet_projection_cls_model.pt")
    parser.add_argument("--degree_output_name", type=str, default="best_triplet_projection_degree_model.pt")
    parser.add_argument("--log_file", type=str, default="save_model/triplet_edit_degree_fuse/train.log")
    

    
    return parser


def main():
    args = build_parser().parse_args()
    if args.mode == "prepare":
        prepare_only(args)
    elif args.mode == "train":
        prepare_only(args)
        main_train(args)
        main_test(args)
    else:
        main_test(args)


if __name__ == "__main__":
    main()
