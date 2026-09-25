from __future__ import annotations

import json
import os

import numpy as np
import torch
import tqdm
from loguru import logger
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)

from lib_sep_4methond_distance_logall2.exp_triplet import label_id_to_name


def predict_single_text_batch(model, input_ids, attention_mask):
    z = model.encode(input_ids, attention_mask)
    degree = model.predict_degree_from_z(z)
    logits = model.classify(z)
    probabilities = torch.softmax(logits, dim=-1)
    labels = torch.argmax(probabilities, dim=-1)
    return degree, logits, probabilities, labels


def get_batch_float_metadata(batch, key, index):
    if key not in batch:
        return None
    value = batch[key][index]
    if torch.is_tensor(value):
        value = value.detach().cpu().item()
    value = float(value)
    return None if value < 0 or not np.isfinite(value) else value


def evaluate_single_text_model(model, dataloader, device, desc="Evaluating"):
    model.eval()
    true_labels, pred_labels = [], []
    true_degrees, pred_degrees, pred_probs, details = [], [], [], []

    with torch.inference_mode():
        for batch in tqdm.tqdm(dataloader, desc=desc):
            degree, _, probabilities, pred_label = predict_single_text_batch(
                model,
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            labels = batch["label"]
            targets = batch["edit_degree"]

            true_labels.extend(labels.cpu().tolist())
            pred_labels.extend(pred_label.cpu().tolist())
            true_degrees.extend(targets.cpu().tolist())
            pred_degrees.extend(degree.cpu().tolist())
            pred_probs.extend(probabilities.cpu().tolist())

            for index in range(len(labels)):
                row = {
                    "text": batch["text"][index],
                    "source_id": batch.get("source_id", [""] * len(labels))[index],
                    "pair_id": batch.get("pair_id", [""] * len(labels))[index],
                    "degree_source": batch.get("degree_source", [""] * len(labels))[index],
                    "true_label": (
                        label_id_to_name(int(labels[index]))
                        if int(labels[index]) in {0, 1, 2}
                        else "unknown"
                    ),
                    "pred_label": label_id_to_name(int(pred_label[index].cpu())),
                    "true_edit_degree": float(targets[index].cpu()),
                    "pred_edit_degree": float(degree[index].cpu()),
                    "pred_prob_human": float(probabilities[index, 0].cpu()),
                    "pred_prob_edited": float(probabilities[index, 1].cpu()),
                    "pred_prob_ai": float(probabilities[index, 2].cpu()),
                }
                for key in (
                    "levenshtein_degree_raw",
                    "jaccard_degree_raw",
                    "semantic_degree_raw",
                    "likelihood_degree_raw",
                    "fusion_degree_raw",
                    "edited_initial_degree",
                    "ai_initial_degree",
                    "ai_correction_ratio",
                    "ai_scale",
                    "levenshtein_edited_initial_degree",
                    "levenshtein_ai_initial_degree",
                    "jaccard_edited_initial_degree",
                    "jaccard_ai_initial_degree",
                    "semantic_edited_initial_degree",
                    "semantic_ai_initial_degree",
                    "likelihood_edited_initial_degree",
                    "likelihood_ai_initial_degree",
                    "likelihood_edited_structure_distance",
                    "likelihood_ai_structure_distance",
                    "fusion_edited_initial_degree",
                    "fusion_ai_initial_degree",
                    "degree_confidence",
                ):
                    row[key] = get_batch_float_metadata(batch, key, index)
                details.append(row)

    return true_labels, pred_labels, true_degrees, pred_degrees, pred_probs, details


def safe_corr(x, y, corr_type="pearson"):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y) & (y >= 0)
    x, y = x[mask], y[mask]
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(pearsonr(x, y)[0] if corr_type == "pearson" else spearmanr(x, y)[0])


def compute_degree_metrics(true_degrees, pred_degrees):
    y_true = np.asarray(true_degrees, dtype=np.float64)
    y_pred = np.asarray(pred_degrees, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true >= 0)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {
            "degree_mae": None,
            "degree_rmse": None,
            "degree_pearson": None,
            "degree_spearman": None,
        }
    return {
        "degree_mae": float(mean_absolute_error(y_true, y_pred)),
        "degree_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "degree_pearson": safe_corr(y_pred, y_true, "pearson"),
        "degree_spearman": safe_corr(y_pred, y_true, "spearman"),
    }


def compute_classification_metrics(true_labels, pred_labels):
    valid = [index for index, label in enumerate(true_labels) if label in {0, 1, 2}]
    if not valid:
        return {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "macro_f1": 0.0,
            "human_f1": 0.0,
            "edited_f1": 0.0,
            "ai_f1": 0.0,
            "confusion_matrix": np.zeros((3, 3), dtype=int),
            "classification_report_text": "",
            "classification_report_dict": {},
        }

    y_true = [true_labels[index] for index in valid]
    y_pred = [pred_labels[index] for index in valid]
    class_f1 = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    report_kwargs = dict(
        labels=[0, 1, 2],
        target_names=["human", "edited", "ai"],
        digits=4,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)),
        "human_f1": float(class_f1[0]),
        "edited_f1": float(class_f1[1]),
        "ai_f1": float(class_f1[2]),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]),
        "classification_report_text": classification_report(y_true, y_pred, **report_kwargs),
        "classification_report_dict": classification_report(
            y_true, y_pred, output_dict=True, **report_kwargs
        ),
    }


def train_triplet_projection_epoch(
    model,
    optimizer,
    train_loader,
    device,
    args,
    epoch,
    loss_mode="joint",
    tag="",
):
    model.train()
    total_loss = 0.0
    progress = tqdm.tqdm(
        train_loader,
        desc=f"Training {tag} epoch {epoch}/{args.epochs}",
    )

    for batch in progress:
        optimizer.zero_grad(set_to_none=True)
        output = model(
            h_input_ids=batch["h_input_ids"].to(device),
            h_attention_mask=batch["h_attention_mask"].to(device),
            e_input_ids=batch["e_input_ids"].to(device),
            e_attention_mask=batch["e_attention_mask"].to(device),
            a_input_ids=batch["a_input_ids"].to(device),
            a_attention_mask=batch["a_attention_mask"].to(device),
            edited_degree=batch["edited_degree"].to(device),
            degree_confidence=batch["degree_confidence"].to(device),
            lambda_cls=args.lambda_cls,
            lambda_degree=args.lambda_degree,
            lambda_axis=args.lambda_axis,
            loss_mode=loss_mode,
        )
        loss = output["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        total_loss += float(loss.item())
        progress.set_postfix({
            "mode": loss_mode,
            "loss": f"{loss.item():.4f}",
            "cls": f"{output['classification_loss'].item():.4f}",
            "map": f"{output['mapping_loss'].item():.4f}",
            "beta": f"{output['beta_kl_loss'].item():.4f}",
            "degree": f"{output['degree_loss'].item():.4f}",
            "anchor": f"{output.get('anchor_degree_loss', output['axis_degree_loss']).item():.4f}",
        })

    return total_loss / max(len(train_loader), 1)

def compute_degree_metrics_bundle(
    true_labels,
    true_degrees,
    pred_degrees,
):
    """
    同时计算：
    1. 所有 Human/Edited/AI 样本上的 degree 指标；
    2. 只针对 Edited 样本的 degree 指标。
    """
    overall_metrics = compute_degree_metrics(
        true_degrees,
        pred_degrees,
    )

    edited_indices = [
        index
        for index, label in enumerate(true_labels)
        if int(label) == 1
    ]

    edited_true_degrees = [
        true_degrees[index]
        for index in edited_indices
    ]

    edited_pred_degrees = [
        pred_degrees[index]
        for index in edited_indices
    ]

    edited_metrics = compute_degree_metrics(
        edited_true_degrees,
        edited_pred_degrees,
    )

    return {
        **overall_metrics,

        "edited_degree_mae": (
            edited_metrics["degree_mae"]
        ),
        "edited_degree_rmse": (
            edited_metrics["degree_rmse"]
        ),
        "edited_degree_pearson": (
            edited_metrics["degree_pearson"]
        ),
        "edited_degree_spearman": (
            edited_metrics["degree_spearman"]
        ),
    }

def validate_triplet_projection(
    model,
    val_loader,
    device,
):
    values = evaluate_single_text_model(
        model,
        val_loader,
        device,
        "Validating",
    )

    cls_metrics = compute_classification_metrics(
        values[0],
        values[1],
    )

    degree_metrics = compute_degree_metrics_bundle(
        true_labels=values[0],
        true_degrees=values[2],
        pred_degrees=values[3],
    )

    return (
        cls_metrics,
        degree_metrics,
        values[5],
    )


def select_score_from_metrics(cls_metrics, degree_metrics, mode="macro_f1"):
    if mode == "macro_f1":
        return cls_metrics["macro_f1"]
    if mode == "edited_f1":
        return cls_metrics["edited_f1"]
    if mode == "balanced_acc":
        return cls_metrics["balanced_accuracy"]
    if mode == "macro_f1_spearman":
        spearman = degree_metrics.get(
            "edited_degree_spearman",
            degree_metrics.get("degree_spearman"),
        )

        return (
            cls_metrics["macro_f1"]
            if spearman is None
            else cls_metrics["macro_f1"] + 0.5 * spearman
        )
    if mode == "macro_f1_minus_mae":
        mae = degree_metrics.get("edited_degree_mae",degree_metrics.get("degree_mae"),)

        return (cls_metrics["macro_f1"]
            if mae is None
            else cls_metrics["macro_f1"] - 0.8 * mae
        )
    
    
    if mode == "degree_mae":
        mae = degree_metrics.get(
            "edited_degree_mae",
            degree_metrics.get("degree_mae"),
        )

        return (
            -float("inf")
            if mae is None
            else -mae
        )
        
    raise ValueError(f"Unsupported selection metric: {mode}")


def compute_pred_degree_stats_by_label(details, label_key="true_label"):
    stats = {}
    for label in ("human", "edited", "ai"):
        values = [
            float(row["pred_edit_degree"])
            for row in details
            if row.get(label_key) == label and np.isfinite(float(row["pred_edit_degree"]))
        ]
        stats[label] = (
            {"count": 0, "min": None, "max": None, "mean": None}
            if not values
            else {
                "count": len(values),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
            }
        )
    return stats


def save_results(output_dir, filename, payload):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    return path
def format_metric(value, digits=4):
    """
    安全格式化评估指标。

    当指标为 None 时返回 N/A，避免使用 :.4f 时出错。
    """
    if value is None:
        return "N/A"

    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)

    if not np.isfinite(value):
        return "N/A"

    return f"{value:.{digits}f}"


def evaluate_two_models_and_save_test(
    cls_model,
    degree_model,
    test_loader,
    device,
    output_dir,
    backbone_path,
    best_cls_epoch,
    best_cls_score,
    best_degree_epoch,
    best_degree_score,
    result_prefix="test",
):
    # =========================================================
    # 0. 规范化结果文件名前缀
    # =========================================================
    safe_result_prefix = os.path.splitext(
        os.path.basename(str(result_prefix))
    )[0]

    if not safe_result_prefix:
        safe_result_prefix = "test"

    # =========================================================
    # 1. 使用分类模型进行分类和编辑度预测
    # =========================================================
    cls_values = evaluate_single_text_model(
        cls_model,
        test_loader,
        device,
        "Testing classification model",
    )

    # cls_values 的结构：
    # 0: true_labels
    # 1: pred_labels
    # 2: true_degrees
    # 3: pred_degrees
    # 4: pred_probs
    # 5: details

    # =========================================================
    # 2. 获得最终使用的 degree predictions
    # =========================================================
    if degree_model is cls_model:
        # 分类和编辑度使用同一个模型
        degree_predictions = cls_values[3]
    else:
        # 使用单独的编辑度模型
        degree_values = evaluate_single_text_model(
            degree_model,
            test_loader,
            device,
            "Testing degree model",
        )

        degree_predictions = degree_values[3]

    # =========================================================
    # 3. 计算分类指标和编辑度指标
    # =========================================================
    cls_metrics = compute_classification_metrics(
        cls_values[0],
        cls_values[1],
    )

    degree_metrics = compute_degree_metrics_bundle(
        true_labels=cls_values[0],
        true_degrees=cls_values[2],
        pred_degrees=degree_predictions,
    )

    # =========================================================
    # 3.1 预测编辑度统计
    # =========================================================
    degree_stats = {
        "pred_degree_std": float(
            np.std(degree_predictions)
        ),
    }

    # =========================================================
    # 4. 保存逐条预测详情
    # =========================================================
    details = cls_values[5]

    for row, cls_degree, degree in zip(
        details,
        cls_values[3],
        degree_predictions,
    ):
        row["pred_edit_degree_cls_model"] = float(
            cls_degree
        )

        row["pred_edit_degree"] = float(
            degree
        )

    # =========================================================
    # 5. 构造最终保存结果
    # =========================================================
    payload = {
        # 当前测试集名称
        "test_name": safe_result_prefix,

        # 模型与 checkpoint 信息
        "backbone_path": backbone_path,

        "best_cls_epoch": int(
            best_cls_epoch
        ),
        "best_cls_score": float(
            best_cls_score
        ),

        "best_degree_epoch": int(
            best_degree_epoch
        ),
        "best_degree_score": float(
            best_degree_score
        ),

        # 分类指标
        "accuracy": cls_metrics[
            "accuracy"
        ],
        "balanced_accuracy": cls_metrics[
            "balanced_accuracy"
        ],
        "macro_f1": cls_metrics[
            "macro_f1"
        ],

        "human_f1": cls_metrics.get(
            "human_f1"
        ),
        "edited_f1": cls_metrics.get(
            "edited_f1"
        ),
        "ai_f1": cls_metrics.get(
            "ai_f1"
        ),

        "confusion_matrix": (
            cls_metrics[
                "confusion_matrix"
            ].tolist()
        ),

        "classification_report": (
            cls_metrics[
                "classification_report_dict"
            ]
        ),

        # 编辑度回归指标
        "degree_metrics": degree_metrics,

        # 编辑度预测统计
        "degree_statistics": degree_stats,

        "pred_degree_stats_by_true_label": (
            compute_pred_degree_stats_by_label(
                details
            )
        ),

        "pred_degree_stats_by_pred_label": (
            compute_pred_degree_stats_by_label(
                details,
                "pred_label",
            )
        ),

        # 逐条预测结果
        "results": details,
    }

    # =========================================================
    # 6. 在终端完整打印评估结果
    # =========================================================
    print(
        "\n"
        "============================================================"
    )

    print(
        "=== Two-Model Triplet Projection Test Evaluation ==="
    )

    print(
        "============================================================"
    )

    print("\n[Test Dataset]")
    print(
        f"Test Name         : {safe_result_prefix}"
    )

    print("\n[Checkpoint Information]")
    print(
        f"Backbone Path     : {backbone_path}"
    )
    print(
        f"Best CLS Epoch    : {best_cls_epoch}"
    )
    print(
        "Best CLS Score    : "
        f"{format_metric(best_cls_score)}"
    )
    print(
        f"Best Degree Epoch : {best_degree_epoch}"
    )
    print(
        "Best Degree Score : "
        f"{format_metric(best_degree_score)}"
    )

    print("\n[Classification Metrics]")
    print(
        "Accuracy          : "
        f"{format_metric(cls_metrics['accuracy'])}"
    )
    print(
        "Balanced Accuracy : "
        f"{format_metric(cls_metrics['balanced_accuracy'])}"
    )
    print(
        "Macro-F1          : "
        f"{format_metric(cls_metrics['macro_f1'])}"
    )
    print(
        "Human F1          : "
        f"{format_metric(cls_metrics.get('human_f1'))}"
    )
    print(
        "Edited F1         : "
        f"{format_metric(cls_metrics.get('edited_f1'))}"
    )
    print(
        "AI F1             : "
        f"{format_metric(cls_metrics.get('ai_f1'))}"
    )

    print("\n[Degree Regression Metrics]")
    print(
        "Degree MAE        : "
        f"{format_metric(degree_metrics['degree_mae'])}"
    )
    print(
        "Degree RMSE       : "
        f"{format_metric(degree_metrics['degree_rmse'])}"
    )
    print(
        "Degree Pearson    : "
        f"{format_metric(degree_metrics['degree_pearson'])}"
    )
    print(
        "Degree Spearman   : "
        f"{format_metric(degree_metrics['degree_spearman'])}"
    )

    print(
        "Edited Degree MAE : "
        f"{format_metric(degree_metrics.get('edited_degree_mae'))}"
    )
    print(
        "Edited Degree RMSE: "
        f"{format_metric(degree_metrics.get('edited_degree_rmse'))}"
    )
    print(
        "Edited Pearson    : "
        f"{format_metric(degree_metrics.get('edited_degree_pearson'))}"
    )
    print(
        "Edited Spearman   : "
        f"{format_metric(degree_metrics.get('edited_degree_spearman'))}"
    )

    print("\n[Predicted Degree Statistics]")
    print(
        "Pred Degree Std   : "
        f"{format_metric(degree_stats['pred_degree_std'])}"
    )

    print("\n[Confusion Matrix]")
    print(
        "Rows = true labels; columns = predicted labels"
    )
    print(
        "Label order: [human, edited, ai]"
    )
    print(
        cls_metrics["confusion_matrix"]
    )

    print("\n[Classification Report]")
    print(
        cls_metrics[
            "classification_report_text"
        ]
    )

    print(
        "[Predicted Degree Statistics by True Label]"
    )
    print(
        json.dumps(
            payload[
                "pred_degree_stats_by_true_label"
            ],
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "\n"
        "[Predicted Degree Statistics by Predicted Label]"
    )
    print(
        json.dumps(
            payload[
                "pred_degree_stats_by_pred_label"
            ],
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "============================================================\n"
    )

    # =========================================================
    # 7. 根据测试集文件名保存完整结果
    # =========================================================
    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

        # 例如：
        # result_prefix = "round1_test_max_50_tokens"
        # 最终保存为：
        # round1_test_max_50_tokens.json
        result_filename = (
            f"{safe_result_prefix}.json"
        )

        path = save_results(
            output_dir,
            result_filename,
            payload,
        )

        logger.info(
            f"Saved test results to: "
            f"{os.path.abspath(path)}"
        )

    return payload




def evaluate_two_models_and_save_test2(
    cls_model,
    degree_model,
    test_loader,
    device,
    output_dir=None,
    backbone_path=None,
    best_cls_epoch=-1,
    best_cls_score=-float("inf"),
    best_degree_epoch=-1,
    best_degree_score=-float("inf"),
):
    cls_values = evaluate_single_text_model(
        cls_model, test_loader, device, "Testing classification model"
    )
    if degree_model is cls_model:
        degree_predictions = cls_values[3]
    else:
        degree_values = evaluate_single_text_model(
            degree_model, test_loader, device, "Testing degree model"
        )
        degree_predictions = degree_values[3]

    cls_metrics = compute_classification_metrics(cls_values[0], cls_values[1])
    degree_metrics = compute_degree_metrics(cls_values[2], degree_predictions)
    details = cls_values[5]
    for row, cls_degree, degree in zip(details, cls_values[3], degree_predictions):
        row["pred_edit_degree_cls_model"] = float(cls_degree)
        row["pred_edit_degree"] = float(degree)

    payload = {
        "backbone_path": backbone_path,
        "best_cls_epoch": int(best_cls_epoch),
        "best_cls_score": float(best_cls_score),
        "best_degree_epoch": int(best_degree_epoch),
        "best_degree_score": float(best_degree_score),
        "accuracy": cls_metrics["accuracy"],
        "balanced_accuracy": cls_metrics["balanced_accuracy"],
        "macro_f1": cls_metrics["macro_f1"],
        "confusion_matrix": cls_metrics["confusion_matrix"].tolist(),
        "classification_report": cls_metrics["classification_report_dict"],
        "degree_metrics": degree_metrics,
        "pred_degree_stats_by_true_label": compute_pred_degree_stats_by_label(details),
        "pred_degree_stats_by_pred_label": compute_pred_degree_stats_by_label(details, "pred_label"),
        "results": details,
    }

    
    
    
    print("\n=== Triplet Projection Test Evaluation ===")

    
    
    print(json.dumps({
        "accuracy": payload["accuracy"],
        "balanced_accuracy": payload["balanced_accuracy"],
        "macro_f1": payload["macro_f1"],
        **degree_metrics,
    }, indent=2, ensure_ascii=False))

    if output_dir:
        path = save_results(output_dir, "triplet_projection_results.json", payload)
        logger.info(f"Saved test results to {path}")
    return payload
