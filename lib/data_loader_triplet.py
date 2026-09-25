from __future__ import annotations

import copy
import gc
import json
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from rapidfuzz.distance import Levenshtein
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib_sep_4methond_distance_logall_0.exp_triplet import normalize_label

##你现在离线部分已经把 confidence 改成了 [0,1]，并且 Fusion confidence 加入了方法分歧和覆盖率；
# 但训练代码仍然在 Dataset、forward() 和 make_teacher_beta() 中把 confidence 强制抬到至少 0.05。
# 同时，同一个 confidence 既控制 Teacher Beta 的尖锐程度，又直接加权三项 loss，低 confidence 样本会被重复削弱。


DEGREE_METHODS = (
    "existing",
    "levenshtein",
    "jaccard",
    "semantic",
    "likelihood",
    "fusion",
)

# Select any non-empty subset of the three supported features.
# The order written here is also the output-vector order.
#
# Examples:
# TOKEN_LL_FEATURE_NAMES = ("log_local_roughness",)
# TOKEN_LL_FEATURE_NAMES = ("mean_nll", "log_local_roughness")
# TOKEN_LL_FEATURE_NAMES = (
#     "mean_nll",
#     "log_fluctuation_energy",
#     "log_local_roughness",
# )
TOKEN_LL_FEATURE_NAMES = (
    #"mean_nll",
    "log_fluctuation_energy",
    "log_local_roughness",
)

SUPPORTED_TOKEN_LL_FEATURE_NAMES = (
    "mean_nll",
    "log_fluctuation_energy",
    "log_local_roughness",
)


def validate_token_ll_feature_names(
    feature_names=None,
) -> tuple[str, ...]:
    """
    Validate and return the active token-LL feature configuration.

    Any non-empty subset of SUPPORTED_TOKEN_LL_FEATURE_NAMES is allowed.
    The configured order is preserved and determines vector order.
    """
    names = tuple(
        TOKEN_LL_FEATURE_NAMES
        if feature_names is None
        else feature_names
    )

    if not names:
        raise ValueError(
            "TOKEN_LL_FEATURE_NAMES cannot be empty."
        )

    if len(names) != len(set(names)):
        raise ValueError(
            "TOKEN_LL_FEATURE_NAMES contains duplicate entries: "
            f"{names}"
        )

    unknown = [
        name
        for name in names
        if name not in SUPPORTED_TOKEN_LL_FEATURE_NAMES
    ]
    if unknown:
        raise ValueError(
            "Unsupported TOKEN_LL_FEATURE_NAMES entries: "
            f"{unknown}. Supported names are: "
            f"{list(SUPPORTED_TOKEN_LL_FEATURE_NAMES)}"
        )

    return names


# Fail early at import time when the source configuration is invalid.
validate_token_ll_feature_names()


LM_REQUIRED_METHODS = {"semantic", "likelihood", "fusion"}


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def tokenize_words(text: str) -> list[str]:
    return re.findall(
        r"\b\w+(?:['’\-]\w+)*\b",
        safe_text(text).lower(),
        flags=re.UNICODE,
    )


class TripletEditDataset(Dataset):
    """One human/edited/AI triplet with a scalar target per item."""

    def __init__(self, triplets, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.valid_triplets = []

        for item in triplets:
            if not all(key in item for key in ("human", "edited", "ai")):
                continue
            if not all(safe_text(item[key]) for key in ("human", "edited", "ai")):
                continue
            if "edited_degree" not in item and "edit_degree" not in item:
                raise ValueError(
                    "Triplet is missing edited_degree/edit_degree. "
                    "Run offline degree computation first."
                )
            self.valid_triplets.append(item)

        if not self.valid_triplets:
            raise RuntimeError("No valid non-empty human/edited/AI triplets found.")

    def __len__(self):
        return len(self.valid_triplets)

    def encode_text(self, text: str):
        encoded = self.tokenizer(
            safe_text(text),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        return encoded["input_ids"].squeeze(0), encoded["attention_mask"].squeeze(0)

    def __getitem__(self, idx):
        item = self.valid_triplets[idx]
        h_ids, h_mask = self.encode_text(item["human"])
        e_ids, e_mask = self.encode_text(item["edited"])
        a_ids, a_mask = self.encode_text(item["ai"])

        degree = float(item.get("edited_degree", item.get("edit_degree")))
        confidence = float(item.get("degree_confidence", 1.0))

        return {
            "h_input_ids": h_ids,
            "h_attention_mask": h_mask,
            "e_input_ids": e_ids,
            "e_attention_mask": e_mask,
            "a_input_ids": a_ids,
            "a_attention_mask": a_mask,
            "edited_degree": torch.tensor(np.clip(degree, 0.0, 1.0), dtype=torch.float32),
            "degree_confidence": torch.tensor(
                np.clip(confidence, 0.0, 1.0), dtype=torch.float32
            ),
        }


def safe_float_meta(item, key, default=-1.0):
    value = item.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class SingleTextEditDataset(Dataset):
    """Validation/test data for three-way classification and degree regression."""

    METADATA_KEYS = (
        "polish_ratio",

        # 各方法未裁剪 degree
        "levenshtein_degree_raw",
        "jaccard_degree_raw",
        "semantic_degree_raw",
        "likelihood_degree_raw",
        "fusion_degree_raw",

        # 各方法最终 degree
        "levenshtein_degree",
        "jaccard_degree",
        "semantic_degree",
        "likelihood_degree",
        "fusion_degree",

        # 各方法 confidence
        "levenshtein_confidence",
        "jaccard_confidence",
        "semantic_confidence",
        "likelihood_confidence",
        "fusion_confidence",

        # 最终选中结果
        "edited_initial_degree",
        "ai_initial_degree",
        "ai_correction_ratio",
        "ai_scale",

        # 各方法初始量
        "levenshtein_edited_initial_degree",
        "levenshtein_ai_initial_degree",
        "jaccard_edited_initial_degree",
        "jaccard_ai_initial_degree",
        "semantic_edited_initial_degree",
        "semantic_ai_initial_degree",
        "likelihood_edited_initial_degree",
        "likelihood_ai_initial_degree",
        "fusion_edited_initial_degree",
        "fusion_ai_initial_degree",

        # Likelihood 诊断量
        "likelihood_edited_structure_distance",
        "likelihood_ai_structure_distance",

        # Fusion 权重
        "fusion_weight_levenshtein",
        "fusion_weight_jaccard",
        "fusion_weight_semantic",
        "fusion_weight_likelihood",
        "fusion_component_count",

        "degree_confidence",
    )

    def __init__(self, samples, tokenizer, max_length: int = 512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        text = safe_text(item["text"])
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
            return_token_type_ids=False,
        )

        raw_label = item.get("label", item.get("text_type"))
        label = -1 if raw_label is None else normalize_label(raw_label)

        edit_degree = item.get("edit_degree", item.get("edited_degree"))
        if edit_degree is None:
            edit_degree = 0.0 if label == 0 else 1.0 if label == 2 else -1.0

        result = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(int(label),dtype=torch.long,),
            "edit_degree": torch.tensor(float(edit_degree),dtype=torch.float32,),
            "text": text,
            "source_id": (""if item.get("source_id") is None else str(item["source_id"])),
            "pair_id": (""if item.get("pair_id") is None else str(item["pair_id"])),
            "degree_source": str(item.get("degree_source", "")),
            "fusion_strategy": str(item.get("fusion_strategy", "")),
        }

        for key in self.METADATA_KEYS:
            result[key] = torch.tensor(safe_float_meta(item, key), dtype=torch.float32)

        return result


def is_triplet_format(data) -> bool:
    return (
        isinstance(data, list)
        and bool(data)
        and isinstance(data[0], dict)
        and all(key in data[0] for key in ("human", "edited", "ai"))
    )


def maybe_sample_triplets(triplets, max_triplets: int, seed: int = 41):
    if max_triplets is None or max_triplets <= 0 or len(triplets) <= max_triplets:
        return triplets
    return random.Random(seed).sample(triplets, max_triplets)


def convert_triplets_to_single_samples(triplets):
    samples = []
    propagated_keys = (
        "degree_source",
        "degree_confidence",
        "degree_ai_alignment",

        "edited_initial_degree",
        "ai_initial_degree",
        "ai_correction_ratio",
        "ai_scale",

        "levenshtein_degree_raw",
        "jaccard_degree_raw",
        "semantic_degree_raw",
        "likelihood_degree_raw",
        "fusion_degree_raw",

        "levenshtein_degree",
        "jaccard_degree",
        "semantic_degree",
        "likelihood_degree",
        "fusion_degree",

        "levenshtein_confidence",
        "jaccard_confidence",
        "semantic_confidence",
        "likelihood_confidence",
        "fusion_confidence",

        "levenshtein_edited_initial_degree",
        "levenshtein_ai_initial_degree",
        "jaccard_edited_initial_degree",
        "jaccard_ai_initial_degree",
        "semantic_edited_initial_degree",
        "semantic_ai_initial_degree",
        "likelihood_edited_initial_degree",
        "likelihood_ai_initial_degree",
        "fusion_edited_initial_degree",
        "fusion_ai_initial_degree",

        "likelihood_edited_structure_distance",
        "likelihood_ai_structure_distance",

        "fusion_weight_levenshtein",
        "fusion_weight_jaccard",
        "fusion_weight_semantic",
        "fusion_weight_likelihood",
        "fusion_component_count",

        "fusion_strategy",
    )

    for item in triplets:
        common = {
            "source_id": item.get("source_id"),
            "pair_id": item.get("pair_id"),
        }

        human_item = {"text": item["human"], "label": "human", "edit_degree": 0.0, **common}
        edited_item = {
            "text": item["edited"],
            "label": "edited",
            "edit_degree": float(item.get("edited_degree", item.get("edit_degree", 0.5))),
            **common,
        }
        ai_item = {"text": item["ai"], "label": "ai", "edit_degree": 1.0, **common}

        for key in propagated_keys:
            if key in item:
                edited_item[key] = item[key]

        samples.extend((human_item, edited_item, ai_item))

    return samples


# -----------------------------------------------------------------------------
# Four endpoint-calibrated edit-degree methods
# -----------------------------------------------------------------------------


def normalized_levenshtein_distance(x: str, y: str) -> float:
    x = safe_text(x)
    y = safe_text(y)
    if not x and not y:
        return 0.0
    return float(Levenshtein.distance(x, y) / max(len(x), len(y), 1))


def token_jaccard_distance(x: str, y: str) -> float:
    x_set = set(tokenize_words(x))
    y_set = set(tokenize_words(y))
    if not x_set and not y_set:
        return 0.0
    if not x_set or not y_set:
        return 1.0
    return float(1.0 - len(x_set & y_set) / len(x_set | y_set))


@dataclass
class DegreeResult:
    """
    Human-relative edit-degree result.

    edited_initial_degree:
        Degree computed only from the Human--Edited pair.

    ai_initial_degree:
        Human--AI distance retained only as diagnostic metadata. It never
        divides, rescales, or otherwise changes the edited degree.

    raw:
        Unclipped human-relative degree. For the current methods this is
        already designed to lie in or near [0, 1].

    direction:
        Alignment with the Human->AI direction. This is metadata only.
    """

    raw: float | None
    clipped: float
    confidence: float
    valid: bool
    anchor_gap: float
    direction: float
    residual: float
    edited_initial_degree: float
    ai_initial_degree: float
    ai_correction_ratio: float
    ai_scale: float

    def as_dict(self, prefix: str) -> dict[str, Any]:
        return {
            f"{prefix}_edited_initial_degree": self.edited_initial_degree,
            f"{prefix}_ai_initial_degree": self.ai_initial_degree,
            f"{prefix}_ai_correction_ratio": self.ai_correction_ratio,
            f"{prefix}_ai_scale": self.ai_scale,
            f"{prefix}_degree_raw": self.raw,
            f"{prefix}_degree": self.clipped,
            f"{prefix}_confidence": self.confidence,
            f"{prefix}_anchor_valid": self.valid,
            f"{prefix}_anchor_gap": self.anchor_gap,
            f"{prefix}_ai_alignment": self.direction,
            # Backward-compatible aliases.
            f"{prefix}_direction": self.direction,
            f"{prefix}_residual": self.residual,
        }


def invalid_degree(
    ai_initial_degree: float = 0.0,
    edited_initial_degree: float = 0.0,
) -> DegreeResult:
    return DegreeResult(
        raw=None,
        clipped=0.5,
        confidence=0.0,
        valid=False,
        anchor_gap=float(ai_initial_degree),
        direction=0.0,
        residual=1.0,
        edited_initial_degree=float(edited_initial_degree),
        ai_initial_degree=float(ai_initial_degree),
        ai_correction_ratio=0.0,
        ai_scale=0.0,
    )


def human_relative_degree_result(
    edited_initial_degree: float,
    ai_initial_degree: float = 0.0,
    alignment: float = 0.0,
    residual: float = 0.0,
    confidence: float = 1.0,
) -> DegreeResult:
    """Compatibility helper for an uncorrected Human-relative degree."""
    l_e = float(edited_initial_degree)
    l_a = float(ai_initial_degree)
    values = np.asarray([l_e, l_a, alignment, residual], dtype=np.float64)

    if not np.all(np.isfinite(values)) or l_e < 0.0:
        return invalid_degree(
            ai_initial_degree=l_a if np.isfinite(l_a) else 0.0,
            edited_initial_degree=l_e if np.isfinite(l_e) else 0.0,
        )

    return DegreeResult(
        raw=l_e,
        clipped=float(np.clip(l_e, 0.0, 1.0)),
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        valid=True,
        anchor_gap=l_a,
        direction=float(np.clip(alignment, -1.0, 1.0)),
        residual=max(float(residual), 0.0),
        edited_initial_degree=l_e,
        ai_initial_degree=l_a,
        ai_correction_ratio=0.0,
        ai_scale=1.0,
    )

def ai_correct_initial_degree(
    edited_initial_degree: float,
    ai_initial_degree: float,
    alignment: float = 0.0,
    residual: float = 0.0,
    anchor_eps: float = 1e-6,
    anchor_tau: float = 0.1,
    overshoot_tau: float = 1.0,
    residual_tau: float = 1.0,
) -> DegreeResult:
    """
    使用 AI anchor 对 Human-relative 编辑度进行归一化。

    编辑度：
        raw_degree =
            edited_initial_degree
            / ai_initial_degree

    confidence 由四部分组成：

        1. anchor confidence：
           AI 与 Human 是否具有足够大的距离；

        2. overshoot confidence：
           Edited 是否明显超过 AI anchor；

        3. alignment confidence：
           Edited 的变化方向是否与 Human -> AI 一致；

        4. residual confidence：
           Edited 是否偏离 Human -> AI 主方向。

    最终：

        confidence =
            anchor_confidence
            * overshoot_confidence
            * geometry_confidence
    """

    # =========================================================
    # 1. 转换与有效性检查
    # =========================================================
    l_e = float(
        edited_initial_degree
    )

    l_a = float(
        ai_initial_degree
    )

    alignment = float(
        alignment
    )

    residual = float(
        residual
    )

    values = np.asarray(
        [
            l_e,
            l_a,
            alignment,
            residual,
        ],
        dtype=np.float64,
    )

    if (
        not np.all(np.isfinite(values))
        or l_a < anchor_eps
        or l_e < 0.0
    ):
        return invalid_degree(
            ai_initial_degree=l_a,
            edited_initial_degree=l_e,
        )

    # =========================================================
    # 2. 检查 confidence 超参数
    # =========================================================
    anchor_tau = float(
        anchor_tau
    )

    overshoot_tau = float(
        overshoot_tau
    )

    residual_tau = float(
        residual_tau
    )

    if (
        not np.isfinite(anchor_tau)
        or anchor_tau <= 0.0
    ):
        raise ValueError(
            "anchor_tau must be a finite positive value."
        )

    if (
        not np.isfinite(overshoot_tau)
        or overshoot_tau <= 0.0
    ):
        raise ValueError(
            "overshoot_tau must be a finite positive value."
        )

    if (
        not np.isfinite(residual_tau)
        or residual_tau <= 0.0
    ):
        raise ValueError(
            "residual_tau must be a finite positive value."
        )

    # =========================================================
    # 3. 规范化 alignment 和 residual
    # =========================================================
    alignment = float(
        np.clip(
            alignment,
            -1.0,
            1.0,
        )
    )

    residual = max(
        residual,
        0.0,
    )

    # =========================================================
    # 4. AI-anchor 编辑度修正
    # =========================================================
    correction_ratio = float(
        (1.0 - l_a) / l_a
    )

    scale = float(
        1.0 / l_a
    )

    raw = float(
        l_e / l_a
    )

    clipped = float(
        np.clip(
            raw,
            0.0,
            1.0,
        )
    )

    # =========================================================
    # 5. Anchor confidence
    #
    # AI 与 Human 越远，使用 AI 作为分母越稳定。
    # =========================================================
    anchor_confidence = float(
        1.0
        - math.exp(
            -l_a / anchor_tau
        )
    )

    # =========================================================
    # 6. Overshoot confidence
    #
    # 当 raw <= 1 时没有惩罚；
    # 当 raw > 1 时，Edited 超过 AI anchor，降低 confidence。
    # =========================================================
    overshoot = max(
        0.0,
        raw - 1.0,
    )

    overshoot_confidence = float(
        math.exp(
            -overshoot / overshoot_tau
        )
    )

    # =========================================================
    # 7. Geometry confidence
    #
    # 如果 Edited 与 Human 几乎相同，方向本身没有定义，
    # 此时不应该因为 alignment=0 而惩罚 confidence。
    # =========================================================
    if l_e <= anchor_eps:
        alignment_confidence = 1.0
        residual_confidence = 1.0
        geometry_confidence = 1.0

    else:
        # alignment 范围从 [-1, 1] 映射到 [0, 1]
        #
        # alignment =  1 -> confidence = 1
        # alignment =  0 -> confidence = 0.5
        # alignment = -1 -> confidence = 0
        alignment_confidence = float(
            np.clip(
                (alignment + 1.0) / 2.0,
                0.0,
                1.0,
            )
        )

        # residual 越大，说明 Edited 越偏离 Human -> AI 主轴
        residual_confidence = float(
            math.exp(
                -residual / residual_tau
            )
        )

        # 使用几何平均，避免直接相乘导致惩罚过强
        geometry_confidence = float(
            math.sqrt(
                alignment_confidence
                * residual_confidence
            )
        )

    # =========================================================
    # 8. 最终 confidence
    # =========================================================
    confidence = float(
        anchor_confidence
        * overshoot_confidence
        * geometry_confidence
    )

    confidence = float(
        np.clip(
            confidence,
            0.0,
            1.0,
        )
    )

    # =========================================================
    # 9. 返回结果
    # =========================================================
    return DegreeResult(
        raw=raw,
        clipped=clipped,
        confidence=confidence,
        valid=True,

        anchor_gap=l_a,
        direction=alignment,
        residual=residual,

        edited_initial_degree=l_e,
        ai_initial_degree=l_a,

        ai_correction_ratio=correction_ratio,
        ai_scale=scale,
    )

def distance_ai_corrected_degree(
    d_he: float,
    d_ha: float,
    d_ea: float,
    eps: float = 1e-8,
) -> DegreeResult:
    """
    Original Levenshtein/Jaccard anchor correction:

        l_e = d(h, e)
        l_a = d(h, a)
        degree = l_e / l_a

    d(e, a) is used only to estimate alignment and residual.
    """
    values = np.asarray([d_he, d_ha, d_ea], dtype=np.float64)
    if not np.all(np.isfinite(values)) or d_ha < eps:
        return invalid_degree(
            ai_initial_degree=d_ha,
            edited_initial_degree=d_he,
        )

    if d_he <= eps:
        alignment = 0.0
        residual = 0.0
    else:
        numerator = d_he * d_he + d_ha * d_ha - d_ea * d_ea
        alignment = float(np.clip(
            numerator / (2.0 * d_he * d_ha + eps),
            -1.0,
            1.0,
        ))
        perpendicular = d_he * math.sqrt(
            max(1.0 - alignment * alignment, 0.0)
        )
        residual = float(perpendicular / (d_ha + eps))

    return ai_correct_initial_degree(
        edited_initial_degree=d_he,
        ai_initial_degree=d_ha,
        alignment=alignment,
        residual=residual,
        anchor_eps=eps,
    )


# Compatibility name retained for call sites from the Human-relative patch.
distance_human_relative_degree = distance_ai_corrected_degree


def semantic_ai_corrected_degree(
    z_h: np.ndarray,
    z_e: np.ndarray,
    z_a: np.ndarray,
    eps: float = 1e-10,
) -> DegreeResult:
    """
    Original semantic AI-anchor correction:

        l_e = ||z_e - z_h|| / 2
        l_a = ||z_a - z_h|| / 2
        degree = l_e / l_a
    """
    z_h = np.asarray(z_h, dtype=np.float64).reshape(-1)
    z_e = np.asarray(z_e, dtype=np.float64).reshape(-1)
    z_a = np.asarray(z_a, dtype=np.float64).reshape(-1)

    if not (z_h.shape == z_e.shape == z_a.shape):
        raise ValueError("Semantic embeddings must have identical shapes.")
    if not all(np.all(np.isfinite(z)) for z in (z_h, z_e, z_a)):
        return invalid_degree()

    edited_offset = z_e - z_h
    ai_axis = z_a - z_h
    edited_norm = float(np.linalg.norm(edited_offset))
    ai_norm = float(np.linalg.norm(ai_axis))
    edited_initial = float(np.clip(edited_norm / 2.0, 0.0, 1.0))
    ai_initial = float(np.clip(ai_norm / 2.0, 0.0, 1.0))

    if ai_initial < eps:
        return invalid_degree(
            ai_initial_degree=ai_initial,
            edited_initial_degree=edited_initial,
        )

    if edited_norm <= eps:
        alignment = 0.0
        residual = 0.0
    else:
        alignment = float(np.clip(
            np.dot(edited_offset, ai_axis)
            / (edited_norm * ai_norm + eps),
            -1.0,
            1.0,
        ))
        projection = (
            np.dot(edited_offset, ai_axis)
            / (ai_norm * ai_norm + eps)
        ) * ai_axis
        residual = float(
            np.linalg.norm(edited_offset - projection) / (ai_norm + eps)
        )

    return ai_correct_initial_degree(
        edited_initial_degree=edited_initial,
        ai_initial_degree=ai_initial,
        alignment=alignment,
        residual=residual,
        anchor_eps=eps,
    )


# Compatibility name retained for call sites from the Human-relative patch.
semantic_human_relative_degree = semantic_ai_corrected_degree


def likelihood_structure_ai_corrected_degree(
    feature_h: np.ndarray,
    feature_e: np.ndarray,
    feature_a: np.ndarray,
    anchor_eps: float = 1e-3,
    distance_tau: float = 1.0,
) -> DegreeResult:
    """
    Compute likelihood-based edit degree using a positive tau
    resolved from the training-set scaler.

        d_e = RMS(feature_e - feature_h)
        d_a = RMS(feature_a - feature_h)

        l_e = 1 - exp(-d_e / tau)
        l_a = 1 - exp(-d_a / tau)

        degree = l_e / l_a
    """
    feature_h = np.asarray(feature_h,dtype=np.float64,).reshape(-1)

    feature_e = np.asarray(feature_e,dtype=np.float64,).reshape(-1)

    feature_a = np.asarray(feature_a,dtype=np.float64,).reshape(-1)

    if not (
        feature_h.shape
        == feature_e.shape
        == feature_a.shape
        and feature_h.size > 0
    ):
        raise ValueError(
            "Likelihood feature vectors must have "
            "identical, non-empty shapes."
        )

    if not all(
        np.all(np.isfinite(vector))
        for vector in (
            feature_h,
            feature_e,
            feature_a,
        )
    ):
        return invalid_degree()

    # ---------------------------------------------------------
    # tau必须已经由训练集拟合或显式指定。
    # 不允许在单条三元组内部临时拟合。
    # ---------------------------------------------------------
    distance_tau = float(distance_tau)

    if (
        not np.isfinite(distance_tau)
        or distance_tau <= 0.0
    ):
        raise ValueError(
            "distance_tau must be a finite positive value. "
            "Use fit_token_ll_difference_scaler() on the "
            "training set before computing likelihood degrees."
        )

    edited_offset = feature_e - feature_h
    ai_offset = feature_a - feature_h

    feature_dim = max(
        int(feature_h.size),
        1,
    )

    edited_distance = float(
        np.linalg.norm(edited_offset)
        / math.sqrt(feature_dim)
    )

    ai_distance = float(
        np.linalg.norm(ai_offset)
        / math.sqrt(feature_dim)
    )

    edited_initial = float(
        -math.expm1(
            -edited_distance / distance_tau
        )
    )

    ai_initial = float(
        -math.expm1(
            -ai_distance / distance_tau
        )
    )

    edited_norm = float(
        np.linalg.norm(edited_offset)
    )

    ai_norm = float(
        np.linalg.norm(ai_offset)
    )

    if (
        edited_norm <= 1e-12
        or ai_norm <= 1e-12
    ):
        alignment = 0.0
        residual = 0.0
    else:
        alignment = float(
            np.clip(
                np.dot(
                    edited_offset,
                    ai_offset,
                )
                / (
                    edited_norm
                    * ai_norm
                    + 1e-12
                ),
                -1.0,
                1.0,
            )
        )

        projection = (
            np.dot(
                edited_offset,
                ai_offset,
            )
            / (
                ai_norm * ai_norm
                + 1e-12
            )
        ) * ai_offset

        residual = float(
            np.linalg.norm(
                edited_offset - projection
            )
            / (
                ai_norm + 1e-12
            )
        )

    return ai_correct_initial_degree(
        edited_initial_degree=edited_initial,
        ai_initial_degree=ai_initial,
        alignment=alignment,
        residual=residual,
        anchor_eps=anchor_eps,
    )
    
    

# Compatibility names retained for all historical call sites.
likelihood_structure_human_relative_degree = likelihood_structure_ai_corrected_degree
likelihood_ai_corrected_degree = likelihood_structure_ai_corrected_degree
likelihood_projection_degree = likelihood_structure_ai_corrected_degree
distance_projection_degree = distance_ai_corrected_degree
semantic_projection_degree = semantic_ai_corrected_degree


def fuse_degree_results(
    results: dict[str, DegreeResult],
    eps: float = 1e-12,
    confidence_gamma: float = 1.0,
    agreement_tau: float = 0.15,
) -> tuple[
    DegreeResult,
    dict[str, float],
    dict[str, float],
]:
    """
    Confidence-weighted late fusion with disagreement-aware confidence.

    Fusion degree:
        weight_m =
            confidence_m ** confidence_gamma
            / sum(confidence_j ** confidence_gamma)

        fusion_degree =
            sum(weight_m * degree_m)

    Fusion confidence:
        base_confidence =
            mean(confidence_m)

        weighted_variance =
            sum(
                weight_m
                * (degree_m - fusion_degree) ** 2
            )

        agreement_confidence =
            exp(
                -(weighted_std / agreement_tau) ** 2
            )

        coverage_confidence =
            valid_component_count
            / total_component_count

        fusion_confidence =
            base_confidence
            * agreement_confidence
            * coverage_confidence
    """

    # =========================================================
    # 1. 检查超参数
    # =========================================================
    confidence_gamma = float(
        confidence_gamma
    )

    agreement_tau = float(
        agreement_tau
    )

    if (
        not np.isfinite(confidence_gamma)
        or confidence_gamma <= 0.0
    ):
        raise ValueError(
            "confidence_gamma must be a finite positive value."
        )

    if (
        not np.isfinite(agreement_tau)
        or agreement_tau <= 0.0
    ):
        raise ValueError(
            "agreement_tau must be a finite positive value."
        )

    # Fusion 模式下一般是 4：
    # levenshtein、jaccard、semantic、likelihood
    total_component_count = int(
        len(results)
    )

    # =========================================================
    # 2. 过滤无效结果
    # =========================================================
    valid_items = [
        (method_name, result)
        for method_name, result in results.items()
        if (
            result.valid
            and result.raw is not None
            and np.isfinite(
                float(result.raw)
            )
            and np.isfinite(
                result.clipped
            )
            and np.isfinite(
                result.confidence
            )
            and np.isfinite(
                result.edited_initial_degree
            )
            and np.isfinite(
                result.ai_initial_degree
            )
            and np.isfinite(
                result.direction
            )
            and np.isfinite(
                result.residual
            )
        )
    ]

    valid_component_count = int(
        len(valid_items)
    )

    # =========================================================
    # 3. 所有方法均无效
    # =========================================================
    if not valid_items:
        diagnostics = {
            "fusion_base_confidence": 0.0,
            "fusion_weighted_variance": 0.0,
            "fusion_weighted_std": 0.0,
            "fusion_agreement_confidence": 0.0,
            "fusion_coverage_confidence": 0.0,
            "fusion_valid_component_count": 0.0,
            "fusion_total_component_count": float(
                total_component_count
            ),
            "fusion_confidence_gamma": float(
                confidence_gamma
            ),
            "fusion_agreement_tau": float(
                agreement_tau
            ),
        }

        return (
            invalid_degree(),
            {},
            diagnostics,
        )

    # =========================================================
    # 4. 读取各方法最终编辑度
    #
    # 每个 result.clipped 已经完成：
    #
    # edited_initial_degree / ai_initial_degree
    #
    # 并裁剪到 [0, 1]
    # =========================================================
    degrees = np.asarray(
        [
            result.clipped
            for _, result in valid_items
        ],
        dtype=np.float64,
    )

    # =========================================================
    # 5. 读取各方法 confidence
    #
    # Fusion 阶段允许使用 [0, 1] 的真实 confidence。
    # 训练 DataLoader 后续仍可把下限限制为 0.05。
    # =========================================================
    confidences = np.asarray(
        [
            result.confidence
            for _, result in valid_items
        ],
        dtype=np.float64,
    )

    confidences = np.clip(
        confidences,
        0.0,
        1.0,
    )

    # =========================================================
    # 6. 计算 confidence 权重
    #
    # gamma = 1：原始 confidence 加权
    # gamma > 1：更强调高 confidence 方法
    # gamma < 1：减弱方法之间的权重差异
    # =========================================================
    unnormalized_weights = np.power(
        confidences,
        confidence_gamma,
    )

    weight_sum = float(
        np.sum(
            unnormalized_weights
        )
    )

    if (
        not np.isfinite(weight_sum)
        or weight_sum <= eps
    ):
        # 所有 confidence 均接近 0 时，退化为等权平均
        weights = np.full(
            valid_component_count,
            1.0 / valid_component_count,
            dtype=np.float64,
        )
    else:
        weights = (
            unnormalized_weights
            / weight_sum
        )

    # =========================================================
    # 7. 计算 Fusion degree
    # =========================================================
    fused_degree = float(
        np.sum(
            weights * degrees
        )
    )

    fused_degree = float(
        np.clip(
            fused_degree,
            0.0,
            1.0,
        )
    )

    # =========================================================
    # 8. Base confidence
    #
    # 使用普通平均，不再次按 confidence 加权，
    # 避免高 confidence 被重复强调。
    # =========================================================
    base_confidence = float(
        np.mean(
            confidences
        )
    )

    base_confidence = float(
        np.clip(
            base_confidence,
            0.0,
            1.0,
        )
    )

    # =========================================================
    # 9. 计算方法间分歧
    #
    # degree 越一致，weighted_std 越接近 0；
    # degree 分歧越大，weighted_std 越大。
    # =========================================================
    degree_errors = (
        degrees - fused_degree
    )

    weighted_variance = float(
        np.sum(
            weights
            * degree_errors
            * degree_errors
        )
    )

    weighted_variance = max(
        weighted_variance,
        0.0,
    )

    weighted_std = float(
        math.sqrt(
            weighted_variance
        )
    )

    # =========================================================
    # 10. Agreement confidence
    #
    # weighted_std = 0：
    #     agreement_confidence = 1
    #
    # weighted_std 越大：
    #     agreement_confidence 越接近 0
    # =========================================================
    agreement_confidence = float(
        math.exp(
            -(
                weighted_std
                / agreement_tau
            ) ** 2
        )
    )

    agreement_confidence = float(
        np.clip(
            agreement_confidence,
            0.0,
            1.0,
        )
    )

    # =========================================================
    # 11. Coverage confidence
    #
    # 四种方法全部有效：
    #     coverage = 4 / 4 = 1
    #
    # 只有三种有效：
    #     coverage = 3 / 4 = 0.75
    # =========================================================
    coverage_confidence = float(
        valid_component_count
        / max(
            total_component_count,
            1,
        )
    )

    coverage_confidence = float(
        np.clip(
            coverage_confidence,
            0.0,
            1.0,
        )
    )

    # =========================================================
    # 12. 最终 Fusion confidence
    # =========================================================
    fused_confidence = float(
        base_confidence
        * agreement_confidence
        * coverage_confidence
    )

    fused_confidence = float(
        np.clip(
            fused_confidence,
            0.0,
            1.0,
        )
    )

    # =========================================================
    # 13. 计算诊断字段
    #
    # 以下字段不会重新参与 Fusion degree 计算。
    # =========================================================
    edited_initial_degree = float(
        np.sum([
            weight
            * result.edited_initial_degree
            for weight, (_, result) in zip(
                weights,
                valid_items,
            )
        ])
    )

    ai_initial_degree = float(
        np.sum([
            weight
            * result.ai_initial_degree
            for weight, (_, result) in zip(
                weights,
                valid_items,
            )
        ])
    )

    alignment = float(
        np.sum([
            weight
            * result.direction
            for weight, (_, result) in zip(
                weights,
                valid_items,
            )
        ])
    )

    alignment = float(
        np.clip(
            alignment,
            -1.0,
            1.0,
        )
    )

    residual = float(
        np.sum([
            weight
            * result.residual
            for weight, (_, result) in zip(
                weights,
                valid_items,
            )
        ])
    )

    residual = max(
        residual,
        0.0,
    )

    # =========================================================
    # 14. 保存各方法实际权重
    # =========================================================
    fusion_weights = {
        method_name: float(weight)
        for weight, (method_name, _) in zip(
            weights,
            valid_items,
        )
    }

    # =========================================================
    # 15. 保存 Fusion confidence 诊断信息
    # =========================================================
    diagnostics = {
        "fusion_base_confidence": float(
            base_confidence
        ),
        "fusion_weighted_variance": float(
            weighted_variance
        ),
        "fusion_weighted_std": float(
            weighted_std
        ),
        "fusion_agreement_confidence": float(
            agreement_confidence
        ),
        "fusion_coverage_confidence": float(
            coverage_confidence
        ),
        "fusion_valid_component_count": float(
            valid_component_count
        ),
        "fusion_total_component_count": float(
            total_component_count
        ),
        "fusion_confidence_gamma": float(
            confidence_gamma
        ),
        "fusion_agreement_tau": float(
            agreement_tau
        ),
    }

    # =========================================================
    # 16. 构造最终结果
    #
    # 每个 component 已经完成 AI-anchor correction，
    # Fusion 阶段不再做第二次校正。
    # =========================================================
    fused_result = DegreeResult(
        raw=fused_degree,
        clipped=fused_degree,
        confidence=fused_confidence,
        valid=True,

        anchor_gap=ai_initial_degree,
        direction=alignment,
        residual=residual,

        edited_initial_degree=edited_initial_degree,
        ai_initial_degree=ai_initial_degree,

        ai_correction_ratio=0.0,
        ai_scale=1.0,
    )

    return (
        fused_result,
        fusion_weights,
        diagnostics,
    )

# -----------------------------------------------------------------------------
# Offline semantic embeddings and mean log-likelihoods
# -----------------------------------------------------------------------------


def get_model_input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def load_metric_model_and_tokenizer(model_path: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
        padding_side="right",
    )
    if tokenizer.eos_token_id is None and tokenizer.bos_token_id is None:
        raise ValueError("Metric tokenizer must provide bos_token_id or eos_token_id.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token

    dtype = torch.float32
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False
    return model, tokenizer


def collect_unique_triplet_texts(triplets) -> list[str]:
    seen = set()
    texts = []
    for item in triplets:
        for key in ("human", "edited", "ai"):
            text = safe_text(item.get(key))
            if text and text not in seen:
                seen.add(text)
                texts.append(text)
    return texts




def compute_token_logprob_features(
    token_log_probs: np.ndarray,
    min_tokens: int = 16,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Dynamically extract the features selected by TOKEN_LL_FEATURE_NAMES.

    Supported features
    ------------------
    mean_nll:
        -mean(ll_t)

    log_fluctuation_energy:
        log(mean((ll_t - mean(ll))^2) + eps)

    log_local_roughness:
        log(mean((ll_t - ll_(t-1))^2) + eps)

    Notes
    -----
    - No FFT/DFT/STFT is used.
    - No per-document z-score is applied.
    - Output dimension and order exactly follow TOKEN_LL_FEATURE_NAMES.
    """
    feature_names = validate_token_ll_feature_names()

    values = np.asarray(
        token_log_probs,
        dtype=np.float64,
    ).reshape(-1)
    values = values[np.isfinite(values)]

    per_feature_min_tokens = {
        "mean_nll": 1,
        "log_fluctuation_energy": 2,
        "log_local_roughness": 2,
    }
    required_tokens = max(
        int(min_tokens),
        max(
            per_feature_min_tokens[name]
            for name in feature_names
        ),
    )

    if values.size < required_tokens:
        return np.full(
            len(feature_names),
            np.nan,
            dtype=np.float32,
        )

    computed: dict[str, float] = {}

    mean_ll = None

    if (
        "mean_nll" in feature_names
        or "log_fluctuation_energy" in feature_names
    ):
        mean_ll = float(np.mean(values))

    if "mean_nll" in feature_names:
        computed["mean_nll"] = -float(mean_ll)

    if "log_fluctuation_energy" in feature_names:
        centered = values - float(mean_ll)
        fluctuation_energy = float(
            np.mean(centered * centered)
        )
        computed["log_fluctuation_energy"] = math.log(
            fluctuation_energy + eps
        )

    if "log_local_roughness" in feature_names:
        differences = np.diff(values)
        local_roughness = float(
            np.mean(differences * differences)
        )
        computed["log_local_roughness"] = math.log(
            local_roughness + eps
        )

    feature_vector = np.asarray(
        [
            computed[name]
            for name in feature_names
        ],
        dtype=np.float32,
    )

    if (
        feature_vector.shape
        != (len(feature_names),)
        or not np.all(np.isfinite(feature_vector))
    ):
        return np.full(
            len(feature_names),
            np.nan,
            dtype=np.float32,
        )

    return feature_vector

def fit_token_ll_difference_scaler(
    triplets,
    raw_feature_map: dict[str, np.ndarray],
    eps: float = 1e-6,
) -> dict[str, Any]:
    """
    Fit the scaler on training Human--Edited differences.

    center: median Human feature vector, used only to keep stored transformed
            features numerically well centered.
    scale:  90th percentile of |feature_e - feature_h| per dimension.
    tau:    median standardized H--E RMS distance divided by log(2), so the
            median training edit maps to degree 0.5.
    """
    human_rows = []
    difference_rows = []

    for item in triplets:
        h = safe_text(item.get("human"))
        e = safe_text(item.get("edited"))
        if h not in raw_feature_map or e not in raw_feature_map:
            continue
        feature_h = np.asarray(raw_feature_map[h], dtype=np.float64)
        feature_e = np.asarray(raw_feature_map[e], dtype=np.float64)
        if (
            feature_h.shape != feature_e.shape
            or feature_h.size != len(TOKEN_LL_FEATURE_NAMES)
            or not np.all(np.isfinite(feature_h))
            or not np.all(np.isfinite(feature_e))
        ):
            continue
        human_rows.append(feature_h)
        difference_rows.append(feature_e - feature_h)

    if not difference_rows:
        raise RuntimeError(
            "No valid Human--Edited token-LL feature differences were "
            "available to fit the training scaler."
        )

    human_matrix = np.stack(human_rows, axis=0)
    difference_matrix = np.stack(difference_rows, axis=0)
    center = np.median(human_matrix, axis=0)

    abs_difference = np.abs(difference_matrix)
    scale = np.quantile(abs_difference, 0.90, axis=0)
    fallback = np.quantile(abs_difference, 0.75, axis=0)
    scale = np.where(scale > eps, scale, fallback)
    scale = np.maximum(scale, eps)

    standardized_difference = difference_matrix / scale
    distances = np.linalg.norm(
        standardized_difference,
        axis=1,
    ) / math.sqrt(len(TOKEN_LL_FEATURE_NAMES))
    distances = distances[np.isfinite(distances)]

    median_distance = float(np.median(distances)) if distances.size else 1.0
    distance_tau = max(median_distance / math.log(2.0), eps)

    return {
        "scaler_version": 2,
        "scaler_type": "human_edited_difference_q90",
        "feature_names": list(TOKEN_LL_FEATURE_NAMES),
        "center": center.astype(float).tolist(),
        "scale": scale.astype(float).tolist(),
        "distance_tau": float(distance_tau),
        "num_fit_triplets": int(difference_matrix.shape[0]),
        "median_standardized_distance": median_distance,
    }


def fit_token_ll_feature_scaler(
    raw_feature_map: dict[str, np.ndarray],
    eps: float = 1e-6,
) -> dict[str, Any]:
    """Fallback text-level scaler for legacy/pseudo-only workflows."""
    feature_names = validate_token_ll_feature_names()
    valid_rows = []
    for vector in raw_feature_map.values():
        vector = np.asarray(
            vector,
            dtype=np.float64,
        ).reshape(-1)
        if (
            vector.size == len(feature_names)
            and np.all(np.isfinite(vector))
        ):
            valid_rows.append(vector)
    if not valid_rows:
        raise RuntimeError("No valid token-LL features to fit fallback scaler.")
    matrix = np.stack(valid_rows, axis=0)
    center = np.median(matrix, axis=0)
    q25 = np.quantile(matrix, 0.25, axis=0)
    q75 = np.quantile(matrix, 0.75, axis=0)
    scale = np.maximum(q75 - q25, eps)
    return {
        "scaler_version": 2,
        "scaler_type": "fallback_text_iqr",
        "feature_names": list(TOKEN_LL_FEATURE_NAMES),
        "center": center.astype(float).tolist(),
        "scale": scale.astype(float).tolist(),
        "distance_tau": 1.0,
        "num_fit_texts": int(matrix.shape[0]),
    }


def save_token_ll_feature_scaler(path: str, scaler: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(scaler, file, ensure_ascii=False, indent=2)


def load_token_ll_feature_scaler(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "Likelihood token-feature scaler not found: " + path
        )

    with open(path, "r", encoding="utf-8") as file:
        scaler = json.load(file)

    current_names = list(
        validate_token_ll_feature_names()
    )
    saved_names = scaler.get("feature_names")

    if saved_names != current_names:
        raise ValueError(
            "Likelihood feature configuration changed. "
            f"Saved features={saved_names}; "
            f"current features={current_names}. "
            "Delete the old scaler or use a new output directory, "
            "then recompute train/validation/test degrees."
        )

    center = np.asarray(
        scaler.get("center", scaler.get("median")),
        dtype=np.float64,
    ).reshape(-1)
    scale = np.asarray(
        scaler.get("scale"),
        dtype=np.float64,
    ).reshape(-1)

    expected_dim = len(current_names)
    if (
        center.size != expected_dim
        or scale.size != expected_dim
        or not np.all(np.isfinite(center))
        or not np.all(np.isfinite(scale))
        or np.any(scale <= 0.0)
    ):
        raise ValueError(
            "Invalid likelihood scaler dimensions or values. "
            f"Expected dimension={expected_dim}, "
            f"center dimension={center.size}, "
            f"scale dimension={scale.size}."
        )

    return scaler


def transform_token_ll_feature_map(
    raw_feature_map: dict[str, np.ndarray],
    scaler: dict[str, Any],
) -> dict[str, np.ndarray]:
    center = np.asarray(
        scaler.get("center", scaler.get("median")),
        dtype=np.float64,
    )
    scale = np.asarray(scaler["scale"], dtype=np.float64)

    transformed: dict[str, np.ndarray] = {}
    for text, vector in raw_feature_map.items():
        vector = np.asarray(vector, dtype=np.float64)
        if vector.shape != center.shape or not np.all(np.isfinite(vector)):
            transformed[text] = np.full(
                center.shape,
                np.nan,
                dtype=np.float32,
            )
            continue
        transformed[text] = ((vector - center) / scale).astype(np.float32)
    return transformed


def token_ll_scaler_path(args) -> str:
    filename = str(
        getattr(
            args,
            "likelihood_scaler_filename",
            "likelihood_token_feature_scaler.json",
        )
    )
    return os.path.join(args.output_dir, filename)


def feature_vector_to_dict(vector: np.ndarray) -> dict[str, float | None]:
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    result: dict[str, float | None] = {}
    for index, name in enumerate(TOKEN_LL_FEATURE_NAMES):
        value = vector[index] if index < vector.size else np.nan
        result[name] = float(value) if np.isfinite(value) else None
    return result

def compute_lm_features_batch(
    texts: list[str],
    model,
    tokenizer,
    max_length: int = 1024,
    batch_size: int = 4,
    likelihood_min_tokens: int = 8,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, float],
    dict[str, np.ndarray],
]:
    """
    Compute, in one LM pass:

    1. normalized semantic embedding;
    2. mean conditional token log-likelihood;
    3. raw time-domain token log-probability fluctuation features.

    No FFT or frequency-domain processing is used.
    """
    input_device = get_model_input_device(model)
    prefix_id = tokenizer.bos_token_id
    if prefix_id is None:
        prefix_id = tokenizer.eos_token_id

    token_sequences = []
    for text in tqdm.tqdm(texts, desc="Tokenizing metric texts"):
        ids = tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max(max_length - 1, 1),
        )
        token_sequences.append([prefix_id] + ids)

    ordered = sorted(
        range(len(texts)),
        key=lambda index: len(token_sequences[index]),
    )
    embedding_map: dict[str, np.ndarray] = {}
    ll_map: dict[str, float] = {}
    raw_ll_feature_map: dict[str, np.ndarray] = {}

    for start in tqdm.tqdm(
        range(0, len(ordered), batch_size),
        desc="Metric LM forward",
    ):
        indices = ordered[start:start + batch_size]
        sequences = [token_sequences[index] for index in indices]
        width = max(len(sequence) for sequence in sequences)

        input_ids = torch.full(
            (len(sequences), width),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device=input_device,
        )
        attention_mask = torch.zeros_like(input_ids)

        for row, sequence in enumerate(sequences):
            sequence_tensor = torch.tensor(
                sequence,
                dtype=torch.long,
                device=input_device,
            )
            input_ids[row, :len(sequence)] = sequence_tensor
            attention_mask[row, :len(sequence)] = 1

        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        hidden = outputs.hidden_states[-1].float()
        content_mask = attention_mask.clone()
        content_mask[:, 0] = 0
        expanded = content_mask.unsqueeze(-1).float()
        pooled = (
            (hidden * expanded).sum(dim=1)
            / expanded.sum(dim=1).clamp(min=1.0)
        )
        pooled = F.normalize(pooled, p=2, dim=-1)

        shift_logits = outputs.logits[:, :-1, :].float()
        shift_labels = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:].bool()
        token_ll = F.log_softmax(
            shift_logits,
            dim=-1,
        ).gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)
        counts = shift_mask.sum(dim=1)

        for local_index, text_index in enumerate(indices):
            text = texts[text_index]
            count = int(counts[local_index].item())
            embedding_map[text] = (
                pooled[local_index]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            if count <= 0:
                ll_map[text] = float("nan")
                raw_ll_feature_map[text] = np.full(
                    len(TOKEN_LL_FEATURE_NAMES),
                    np.nan,
                    dtype=np.float32,
                )
                continue

            valid_token_ll = (
                token_ll[local_index, :count]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            ll_map[text] = float(valid_token_ll.mean())
            raw_ll_feature_map[text] = compute_token_logprob_features(
                valid_token_ll,
                min_tokens=likelihood_min_tokens,
            )

        del (
            outputs,
            hidden,
            pooled,
            shift_logits,
            shift_labels,
            shift_mask,
            token_ll,
            counts,
            input_ids,
            attention_mask,
        )

    return embedding_map, ll_map, raw_ll_feature_map

# -----------------------------------------------------------------------------
# Top-k pseudo-triplet construction from label-only single-text data
# -----------------------------------------------------------------------------


PSEUDO_DEGREE_METHODS = (
    "levenshtein",
    "jaccard",
    "semantic",
    "likelihood",
    "fusion",
)


def load_label_only_samples(path: str) -> list[dict[str, Any]]:
    """
    Load single-text samples:

        {
            "text": "...",
            "label": "human" | "edited" | "ai",
            "pair_id": optional,
            "source_id": optional
        }
    """
    data = load_json_list(path)
    samples: list[dict[str, Any]] = []

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            continue

        text = safe_text(item.get("text"))
        raw_label = item.get("label", item.get("text_type"))

        if not text:
            continue
        if raw_label is None:
            raise ValueError(
                f"Extra sample {index} is missing label/text_type."
            )

        label_id = normalize_label(raw_label)
        label_name = {0: "human", 1: "edited", 2: "ai"}[label_id]

        samples.append({
            **copy.deepcopy(item),
            "text": text,
            "label": label_name,
            "label_id": int(label_id),
            "source_id": ""
            if item.get("source_id") is None
            else str(item.get("source_id")),
            "pair_id": ""
            if item.get("pair_id") is None
            else str(item.get("pair_id")),
        })

    return samples


def _resample_likelihood_signature(*args, **kwargs):
    raise RuntimeError(
        "Likelihood signature interpolation has been removed. "
        "Use token log-probability fluctuation features instead."
    )

def compute_lm_pool_features_batch(
    texts: list[str],
    model,
    tokenizer,
    max_length: int = 1024,
    batch_size: int = 4,
    likelihood_min_tokens: int = 8,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, float],
    dict[str, np.ndarray],
]:
    """Alias used by pseudo-triplet construction."""
    return compute_lm_features_batch(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=batch_size,
        likelihood_min_tokens=likelihood_min_tokens,
    )

def _robust_minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        return values

    finite = np.isfinite(values)
    output = np.ones_like(values, dtype=np.float64)

    if not finite.any():
        return output

    valid = values[finite]
    low = float(np.quantile(valid, 0.05))
    high = float(np.quantile(valid, 0.95))

    if high <= low + 1e-12:
        output[finite] = 0.0
        return output

    output[finite] = np.clip(
        (valid - low) / (high - low),
        0.0,
        1.0,
    )
    return output


def likelihood_retrieval_distance(
    query_text: str,
    candidate_text: str,
    ll_feature_map: dict[str, np.ndarray],
) -> float:
    """
    RMS Euclidean distance between robustly standardized time-domain token
    log-probability feature vectors.
    """
    query = np.asarray(
        ll_feature_map[query_text],
        dtype=np.float64,
    )
    candidate = np.asarray(
        ll_feature_map[candidate_text],
        dtype=np.float64,
    )

    if (
        query.shape != candidate.shape
        or query.size == 0
        or not np.all(np.isfinite(query))
        or not np.all(np.isfinite(candidate))
    ):
        return float("inf")

    return float(
        np.linalg.norm(query - candidate)
        / math.sqrt(max(query.size, 1))
    )

def rank_topk_anchor_candidates(
    edited_item: dict[str, Any],
    anchor_pool: list[dict[str, Any]],
    retrieval_method: str,
    topk: int,
    embedding_map: dict[str, np.ndarray] | None,
    ll_feature_map: dict[str, np.ndarray] | None,
) -> list[dict[str, Any]]:
    """Retrieve top-k human or AI candidates for one edited sample."""
    if not anchor_pool:
        return []

    edited_text = safe_text(edited_item["text"])
    component_rows: list[dict[str, float]] = []

    for anchor_item in anchor_pool:
        anchor_text = safe_text(anchor_item["text"])
        row: dict[str, float] = {}

        if retrieval_method in {"levenshtein", "fusion"}:
            row["levenshtein"] = normalized_levenshtein_distance(
                edited_text,
                anchor_text,
            )

        if retrieval_method in {"jaccard", "fusion"}:
            row["jaccard"] = token_jaccard_distance(
                edited_text,
                anchor_text,
            )

        if retrieval_method in {"semantic", "fusion"}:
            if embedding_map is None:
                raise ValueError(
                    "Semantic pseudo retrieval requires embedding_map."
                )
            row["semantic"] = float(
                (
                    1.0
                    - np.clip(
                        np.dot(
                            embedding_map[edited_text],
                            embedding_map[anchor_text],
                        ),
                        -1.0,
                        1.0,
                    )
                )
                / 2.0
            )

        if retrieval_method in {"likelihood", "fusion"}:
            if ll_feature_map is None:
                raise ValueError(
                    "Likelihood pseudo retrieval requires ll_feature_map."
                )
            row["likelihood"] = likelihood_retrieval_distance(
                edited_text,
                anchor_text,
                ll_feature_map=ll_feature_map,
            )

        component_rows.append(row)

    if not component_rows or not component_rows[0]:
        return []

    component_names = list(component_rows[0].keys())
    normalized_components: dict[str, np.ndarray] = {}

    for name in component_names:
        values = np.asarray(
            [row[name] for row in component_rows],
            dtype=np.float64,
        )
        normalized_components[name] = (
            _robust_minmax(values)
            if retrieval_method == "fusion"
            else values
        )

    scored: list[tuple[float, dict[str, Any]]] = []

    for index, anchor_item in enumerate(anchor_pool):
        if retrieval_method == "fusion":
            score = float(np.mean([
                normalized_components[name][index]
                for name in component_names
            ]))
        else:
            score = float(
                normalized_components[component_names[0]][index]
            )

        if not np.isfinite(score):
            continue

        enriched = copy.deepcopy(anchor_item)
        enriched["_pseudo_retrieval_distance"] = score
        
        # enriched["_pseudo_retrieval_components"] = {
        #     name: float(component_rows[index][name])
        #     for name in component_names
        # }
        enriched["_pseudo_retrieval_components"] = {
            name: finite_float_or_none(component_rows[index][name])
            for name in component_names
        }
        scored.append((score, enriched))

    scored.sort(key=lambda pair: pair[0])
    return [
        item
        for _, item in scored[:max(int(topk), 1)]
    ]

def build_pseudo_triplet_for_edited_sample(
    edited_item: dict[str, Any],
    human_pool: list[dict[str, Any]],
    ai_pool: list[dict[str, Any]],
    args,
    embedding_map: dict[str, np.ndarray] | None = None,
    ll_map: dict[str, float] | None = None,
    ll_feature_map: dict[str, np.ndarray] | None = None,
) -> dict[str, Any] | None:
    """
    Build one pseudo triplet using top-k retrieval and AI-only correction.

    Likelihood retrieval and likelihood degree both use time-domain token
    conditional-log-probability features. No FFT is used.
    """
    degree_method = str(args.degree_method)
    retrieval_method = str(
        getattr(args, "pseudo_retrieval_method", "same")
    )
    if retrieval_method == "same":
        retrieval_method = degree_method

    if degree_method not in PSEUDO_DEGREE_METHODS:
        raise ValueError(
            f"Unsupported pseudo degree method: {degree_method!r}"
        )
    if retrieval_method not in PSEUDO_DEGREE_METHODS:
        raise ValueError(
            f"Unsupported pseudo retrieval method: {retrieval_method!r}"
        )

    topk = max(int(getattr(args, "pseudo_topk", 10)), 1)

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

    degree_margin = max(
        float(getattr(args, "pseudo_degree_range_margin", 0.05)),
        0.0,
    )
    min_alignment = float(np.clip(
        getattr(args, "pseudo_min_alignment", 0.0),
        -1.0,
        1.0,
    ))
    min_ai_initial = max(
        float(getattr(args, "pseudo_min_ai_initial_degree", 0.05)),
        0.0,
    )

    pair_candidates: list[dict[str, Any]] = []

    for human_item in top_humans:
        for ai_item in top_ais:
            human_text = safe_text(human_item["text"])
            edited_text = safe_text(edited_item["text"])
            ai_text = safe_text(ai_item["text"])

            if human_text == edited_text or ai_text == edited_text:
                continue
            if human_text == ai_text:
                continue

            computed = compute_triplet_degrees(
                {
                    "human": human_text,
                    "edited": edited_text,
                    "ai": ai_text,
                },
                degree_method=degree_method,
                embedding_map=embedding_map,
                ll_map=ll_map,
                ll_feature_map=ll_feature_map,
                ll_anchor_eps=float(args.ll_anchor_eps),
                likelihood_delta_tau=float(
                    getattr(
                        args,
                        "_resolved_likelihood_tau",
                        getattr(args, "likelihood_delta_tau", 1.0),
                    )
                ),
            )

            prefix = "fusion" if degree_method == "fusion" else degree_method
            raw = computed.get(f"{prefix}_degree_raw")
            valid = bool(computed.get(f"{prefix}_anchor_valid", False))
            ai_initial = float(
                computed.get(f"{prefix}_ai_initial_degree", 0.0)
            )
            edited_initial = float(
                computed.get(f"{prefix}_edited_initial_degree", 0.0)
            )
            alignment = float(
                computed.get(
                    f"{prefix}_ai_alignment",
                    computed.get(f"{prefix}_direction", 0.0),
                )
            )
            confidence = float(
                computed.get(f"{prefix}_confidence", 0.05)
            )

            if raw is None or not np.isfinite(float(raw)) or not valid:
                continue

            raw = float(raw)
            if ai_initial < min_ai_initial:
                continue
            if not (0.0 <= raw <= 1.0 + degree_margin):
                continue
            if alignment < min_alignment:
                continue

            retrieval_distance = 0.5 * (
                float(human_item["_pseudo_retrieval_distance"])
                + float(ai_item["_pseudo_retrieval_distance"])
            )

            pair_candidates.append({
                "computed": computed,
                "human_item": human_item,
                "ai_item": ai_item,
                "raw_degree": raw,
                "edited_initial_degree": edited_initial,
                "ai_initial_degree": ai_initial,
                "alignment": alignment,
                "confidence": confidence,
                "retrieval_distance": retrieval_distance,
            })

    if not pair_candidates:
        return None

    ai_initial_values = np.asarray(
        [item["ai_initial_degree"] for item in pair_candidates],
        dtype=np.float64,
    )
    retrieval_values = np.asarray(
        [item["retrieval_distance"] for item in pair_candidates],
        dtype=np.float64,
    )
    anchor_scores = _robust_minmax(ai_initial_values)
    retrieval_scores = 1.0 - _robust_minmax(retrieval_values)

    anchor_weight = max(
        float(getattr(args, "pseudo_anchor_gap_weight", 0.5)),
        0.0,
    )
    alignment_weight = max(
        float(getattr(args, "pseudo_alignment_weight", 0.2)),
        0.0,
    )
    retrieval_weight = max(
        float(getattr(args, "pseudo_retrieval_weight", 0.2)),
        0.0,
    )
    confidence_weight = max(
        1.0 - anchor_weight - alignment_weight - retrieval_weight,
        0.0,
    )
    weights = np.asarray(
        [anchor_weight, alignment_weight, retrieval_weight, confidence_weight],
        dtype=np.float64,
    )
    if weights.sum() <= 1e-12:
        weights[-1] = 1.0
    weights /= weights.sum()

    best_candidate = None
    best_score = -float("inf")

    for index, candidate in enumerate(pair_candidates):
        alignment_score = (
            float(np.clip(candidate["alignment"], -1.0, 1.0)) + 1.0
        ) / 2.0
        selection_score = (
            weights[0] * float(anchor_scores[index])
            + weights[1] * alignment_score
            + weights[2] * float(retrieval_scores[index])
            + weights[3] * candidate["confidence"]
        )
        if selection_score > best_score:
            best_score = float(selection_score)
            best_candidate = candidate

    if best_candidate is None:
        return None

    computed = copy.deepcopy(best_candidate["computed"])
    human_item = best_candidate["human_item"]
    ai_item = best_candidate["ai_item"]

    confidence_scale = float(np.clip(
        getattr(args, "pseudo_confidence_scale", 0.5),
        0.0,
        1.0,
    ))
    computed["degree_confidence"] = float(np.clip(
        float(computed["degree_confidence"]) * confidence_scale,
        0.0,
        1.0,
    ))

    computed.update({
        "degree_source": f"pseudo_topk_{retrieval_method}_to_{degree_method}",
        "is_pseudo_triplet": True,
        "pseudo_selection_score": float(best_score),
        "pseudo_retrieval_method": retrieval_method,
        "pseudo_degree_method": degree_method,
        "pseudo_topk": topk,
        "pseudo_selected_edited_initial_degree": float(
            best_candidate["edited_initial_degree"]
        ),
        "pseudo_selected_ai_initial_degree": float(
            best_candidate["ai_initial_degree"]
        ),
        "pseudo_selected_alignment": float(best_candidate["alignment"]),
        "pseudo_selected_retrieval_distance": float(
            best_candidate["retrieval_distance"]
        ),
        "pseudo_human_retrieval_distance": float(
            human_item["_pseudo_retrieval_distance"]
        ),
        "pseudo_ai_retrieval_distance": float(
            ai_item["_pseudo_retrieval_distance"]
        ),
        "pseudo_human_retrieval_components": (
            human_item["_pseudo_retrieval_components"]
        ),
        "pseudo_ai_retrieval_components": (
            ai_item["_pseudo_retrieval_components"]
        ),
        "edited_source_id": edited_item.get("source_id", ""),
        "edited_pair_id": edited_item.get("pair_id", ""),
        "pseudo_human_source_id": human_item.get("source_id", ""),
        "pseudo_human_pair_id": human_item.get("pair_id", ""),
        "pseudo_ai_source_id": ai_item.get("source_id", ""),
        "pseudo_ai_pair_id": ai_item.get("pair_id", ""),
    })
    return computed

def build_topk_pseudo_triplets_from_extra_file(
    extra_train_file: str,
    args,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Convert label-only extra training data into temporary triplets."""
    if not extra_train_file:
        return []

    if args.degree_method not in PSEUDO_DEGREE_METHODS:
        raise ValueError(
            "--extra_train_file requires a computed degree method."
        )

    print(f">>> Loading label-only extra data: {extra_train_file}")
    samples = load_label_only_samples(extra_train_file)

    human_pool = [item for item in samples if item["label_id"] == 0]
    edited_pool = [item for item in samples if item["label_id"] == 1]
    ai_pool = [item for item in samples if item["label_id"] == 2]

    print(f"Extra human anchors : {len(human_pool)}")
    print(f"Extra edited samples: {len(edited_pool)}")
    print(f"Extra AI anchors    : {len(ai_pool)}")

    if not human_pool or not edited_pool or not ai_pool:
        print(
            "Cannot construct pseudo triplets: extra data must contain "
            "human, edited, and AI samples."
        )
        return []

    retrieval_method = str(
        getattr(args, "pseudo_retrieval_method", "same")
    )
    if retrieval_method == "same":
        retrieval_method = str(args.degree_method)

    lm_features_required = (
        args.degree_method in LM_REQUIRED_METHODS
        or retrieval_method in LM_REQUIRED_METHODS
    )

    embedding_map = None
    ll_map = None
    ll_feature_map = None

    if lm_features_required:
        print(
            ">>> Loading metric LM for pseudo retrieval/degree "
            f"(retrieval={retrieval_method}, degree={args.degree_method})..."
        )
        metric_model, metric_tokenizer = load_metric_model_and_tokenizer(
            args.metric_model_path,
            device,
        )

        unique_texts = []
        seen = set()
        for item in samples:
            text = safe_text(item["text"])
            if text and text not in seen:
                seen.add(text)
                unique_texts.append(text)

        embedding_map, ll_map, raw_ll_feature_map = compute_lm_pool_features_batch(
            texts=unique_texts,
            model=metric_model,
            tokenizer=metric_tokenizer,
            max_length=args.metric_max_length,
            batch_size=args.metric_batch_size,
            likelihood_min_tokens=int(
                getattr(args, "likelihood_min_tokens", 16)
            ),
        )

        scaler_path = token_ll_scaler_path(args)
        if os.path.isfile(scaler_path):
            scaler = load_token_ll_feature_scaler(scaler_path)
        else:
            # This occurs when likelihood is used only as pseudo retrieval and
            # the real degree method did not need likelihood features.
            scaler = fit_token_ll_feature_scaler(raw_ll_feature_map)
            save_token_ll_feature_scaler(scaler_path, scaler)
            print(
                f">>> Fitted likelihood feature scaler from extra training "
                f"data: {scaler_path}"
            )

        ll_feature_map = transform_token_ll_feature_map(
            raw_ll_feature_map,
            scaler,
        )
        configured_tau = float(getattr(args, "likelihood_delta_tau", 0.0))
        args._resolved_likelihood_tau = (
            configured_tau
            if configured_tau > 0.0
            else float(scaler.get("distance_tau", 1.0))
        )

        del metric_model, metric_tokenizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pseudo_triplets = []
    for edited_item in tqdm.tqdm(
        edited_pool,
        desc=f"Building {args.degree_method} pseudo triplets",
    ):
        pseudo = build_pseudo_triplet_for_edited_sample(
            edited_item=edited_item,
            human_pool=human_pool,
            ai_pool=ai_pool,
            args=args,
            embedding_map=embedding_map,
            ll_map=ll_map,
            ll_feature_map=ll_feature_map,
        )
        if pseudo is None:
            continue

        min_confidence = float(
            getattr(args, "pseudo_min_confidence", 0.05)
        )
        if float(pseudo["degree_confidence"]) < min_confidence:
            continue
        pseudo_triplets.append(pseudo)

    pseudo_triplets.sort(
        key=lambda item: float(item.get("degree_confidence", 0.0)),
        reverse=True,
    )

    max_pseudo = int(getattr(args, "max_pseudo_triplets", 0))
    if max_pseudo > 0:
        pseudo_triplets = pseudo_triplets[:max_pseudo]

    print(f"Built pseudo triplets: {len(pseudo_triplets)}")
    return pseudo_triplets

# -----------------------------------------------------------------------------
# Triplet preparation
# -----------------------------------------------------------------------------
def finite_float_or_none(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    return value if np.isfinite(value) else None

def compute_triplet_degrees(
    item: dict[str, Any],
    degree_method: str,
    embedding_map: dict[str, np.ndarray] | None = None,
    ll_map: dict[str, float] | None = None,
    ll_feature_map: dict[str, np.ndarray] | None = None,
    ll_anchor_eps: float = 1e-3,
    likelihood_delta_tau: float = 1.0,
) -> dict[str, Any]:
    """Compute an initial Human-relative degree, then apply AI-anchor correction."""
    h = safe_text(item["human"])
    e = safe_text(item["edited"])
    a = safe_text(item["ai"])

    results: dict[str, DegreeResult] = {}

    if degree_method in {"levenshtein", "fusion"}:
        d_he = normalized_levenshtein_distance(h, e)
        d_ha = normalized_levenshtein_distance(h, a)
        d_ea = normalized_levenshtein_distance(e, a)
        result = distance_ai_corrected_degree(d_he, d_ha, d_ea)
        results["levenshtein"] = result
        item.update({
            "levenshtein_distance_he": d_he,
            "levenshtein_distance_ha": d_ha,
            "levenshtein_distance_ea": d_ea,
            **result.as_dict("levenshtein"),
        })

    if degree_method in {"jaccard", "fusion"}:
        d_he = token_jaccard_distance(h, e)
        d_ha = token_jaccard_distance(h, a)
        d_ea = token_jaccard_distance(e, a)
        result = distance_ai_corrected_degree(d_he, d_ha, d_ea)
        results["jaccard"] = result
        item.update({
            "jaccard_distance_he": d_he,
            "jaccard_distance_ha": d_ha,
            "jaccard_distance_ea": d_ea,
            **result.as_dict("jaccard"),
        })

    if degree_method in {"semantic", "fusion"}:
        if embedding_map is None:
            raise ValueError("Semantic degree requires embedding_map.")
        result = semantic_ai_corrected_degree(
            embedding_map[h],
            embedding_map[e],
            embedding_map[a],
        )
        results["semantic"] = result
        item.update(result.as_dict("semantic"))

    if degree_method in {"likelihood", "fusion"}:
        if ll_feature_map is None:
            raise ValueError(
                "Likelihood degree requires standardized token-LL features."
            )

        feature_h = ll_feature_map[h]
        feature_e = ll_feature_map[e]
        feature_a = ll_feature_map[a]

        result = likelihood_structure_ai_corrected_degree(
            feature_h,
            feature_e,
            feature_a,
            anchor_eps=ll_anchor_eps,
            distance_tau=likelihood_delta_tau,
        )
        results["likelihood"] = result

        feature_dim = max(len(TOKEN_LL_FEATURE_NAMES), 1)
        edited_structure_distance = float(
            np.linalg.norm(
                np.asarray(feature_e) - np.asarray(feature_h)
            ) / math.sqrt(feature_dim)
        ) if all(
            np.all(np.isfinite(np.asarray(vector)))
            for vector in (feature_h, feature_e)
        ) else None
        ai_structure_distance = float(
            np.linalg.norm(
                np.asarray(feature_a) - np.asarray(feature_h)
            ) / math.sqrt(feature_dim)
        ) if all(
            np.all(np.isfinite(np.asarray(vector)))
            for vector in (feature_h, feature_a)
        ) else None

        item.update({
            "likelihood_feature_names": list(TOKEN_LL_FEATURE_NAMES),
            "human_token_ll_features_z": feature_vector_to_dict(feature_h),
            "edited_token_ll_features_z": feature_vector_to_dict(feature_e),
            "ai_token_ll_features_z": feature_vector_to_dict(feature_a),
            "likelihood_edited_structure_distance": edited_structure_distance,
            "likelihood_ai_structure_distance": ai_structure_distance,
            "likelihood_structure_tau": float(likelihood_delta_tau),
            **result.as_dict("likelihood"),
        })

        # if ll_map is not None:
        #     item.update({
        #         "human_mean_log_likelihood": float(ll_map[h]),
        #         "edited_mean_log_likelihood": float(ll_map[e]),
        #         "ai_mean_log_likelihood": float(ll_map[a]),
        #     })
        if ll_map is not None:
            item.update({
                "human_mean_log_likelihood": finite_float_or_none(ll_map[h]),
                "edited_mean_log_likelihood": finite_float_or_none(ll_map[e]),
                "ai_mean_log_likelihood": finite_float_or_none(ll_map[a]),
            })
    # =========================================================
    # Fusion 或单方法结果选择
    # =========================================================

    if degree_method == "fusion":
        (
            selected,
            fusion_weights,
            fusion_diagnostics,
        ) = fuse_degree_results(
            results,
            confidence_gamma=1.0,
            agreement_tau=0.15,
        )

        # 保存 Fusion 自身结果
        item.update(
            selected.as_dict("fusion")
        )

        # 保存 Fusion confidence 的诊断信息
        item.update(
            fusion_diagnostics
        )

        item["fusion_strategy"] = (
            "confidence_weighted_late_fusion_"
            "with_disagreement_penalty"
        )

        # 实际参与 Fusion 的有效方法数量
        item["fusion_component_count"] = int(
            len(fusion_weights)
        )

        # 保存各方法权重
        for method_name in (
            "levenshtein",
            "jaccard",
            "semantic",
            "likelihood",
        ):
            item[
                f"fusion_weight_{method_name}"
            ] = float(
                fusion_weights.get(
                    method_name,
                    0.0,
                )
            )

    else:
        selected = results[
            degree_method
        ]

    # =========================================================
    # 保存最终被选中的编辑度结果
    # =========================================================
    item["edited_initial_degree"] = (
        selected.edited_initial_degree
    )

    item["ai_initial_degree"] = (
        selected.ai_initial_degree
    )

    item["ai_correction_ratio"] = (
        selected.ai_correction_ratio
    )

    item["ai_scale"] = (
        selected.ai_scale
    )

    item["edited_degree_raw"] = (
        selected.raw
    )

    item["edited_degree"] = (
        selected.clipped
    )

    item["degree_confidence"] = (
        selected.confidence
    )

    item["degree_source"] = (
        degree_method
    )

    item["degree_anchor_valid"] = (
        selected.valid
    )

    item["degree_ai_alignment"] = (
        selected.direction
    )

    return item




def prepare_triplets_with_offline_degree(
    triplets,
    args,
    device,
    split_name: str = "train",
):
    degree_method = args.degree_method
    if degree_method not in DEGREE_METHODS:
        raise ValueError(f"Unsupported degree_method: {degree_method}")

    valid_triplets = []
    for item in triplets:
        if not all(key in item for key in ("human", "edited", "ai")):
            continue
        if not all(safe_text(item[key]) for key in ("human", "edited", "ai")):
            continue
        valid_triplets.append(copy.deepcopy(item))

    if degree_method == "existing":
        checked = []
        for item in valid_triplets:
            if "edited_degree" not in item and "edit_degree" not in item:
                raise ValueError(
                    f"{split_name} triplet lacks edited_degree/edit_degree. "
                    "Choose a computed --degree_method first."
                )
            item["edited_degree"] = float(np.clip(
                float(item.get("edited_degree", item["edit_degree"])),
                0.0,
                1.0,
            ))
            item["degree_confidence"] = float(np.clip(
                float(item.get("degree_confidence", 1.0)),
                0.05,
                1.0,
            ))
            item["degree_source"] = item.get("degree_source", "existing")
            checked.append(item)
        return checked

    embedding_map = None
    ll_map = None
    ll_feature_map = None
    resolved_likelihood_tau = float(
        getattr(args, "likelihood_delta_tau", 1.0)
    )

    if degree_method in LM_REQUIRED_METHODS:
        print(f">>> Computing {degree_method} features for {split_name}...")
        model, metric_tokenizer = load_metric_model_and_tokenizer(
            args.metric_model_path,
            device,
        )
        texts = collect_unique_triplet_texts(valid_triplets)
        embedding_map, ll_map, raw_ll_feature_map = compute_lm_features_batch(
            texts=texts,
            model=model,
            tokenizer=metric_tokenizer,
            max_length=args.metric_max_length,
            batch_size=args.metric_batch_size,
            likelihood_min_tokens=int(
                getattr(args, "likelihood_min_tokens", 16)
            ),
        )

        if degree_method in {"likelihood", "fusion"}:
            scaler_path = token_ll_scaler_path(args)
            if split_name == "train":
                scaler = fit_token_ll_difference_scaler(
                    valid_triplets,
                    raw_ll_feature_map,
                )
                save_token_ll_feature_scaler(scaler_path, scaler)
                print(
                    f">>> Saved training likelihood feature scaler: "
                    f"{scaler_path}"
                )
            else:
                scaler = load_token_ll_feature_scaler(scaler_path)

            ll_feature_map = transform_token_ll_feature_map(
                raw_ll_feature_map,
                scaler,
            )
            configured_tau = float(
                getattr(args, "likelihood_delta_tau", 0.0)
            )
            resolved_likelihood_tau = (
                configured_tau
                if configured_tau > 0.0
                else float(scaler.get("distance_tau", 1.0))
            )
        else:
            resolved_likelihood_tau = float(
                getattr(args, "likelihood_delta_tau", 1.0)
            )

        del model, metric_tokenizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    prepared = []
    for item in tqdm.tqdm(
        valid_triplets,
        desc=f"Computing {degree_method} degree ({split_name})",
    ):
        prepared.append(
            compute_triplet_degrees(
                item,
                degree_method=degree_method,
                embedding_map=embedding_map,
                ll_map=ll_map,
                ll_feature_map=ll_feature_map,
                ll_anchor_eps=args.ll_anchor_eps,
                likelihood_delta_tau=resolved_likelihood_tau,
            )
        )

    valid_count = sum(
        bool(item.get("degree_anchor_valid", False))
        for item in prepared
    )
    print(f"{split_name}: valid degrees {valid_count}/{len(prepared)}")
    return prepared

def load_json_list(path: str):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"JSON file must contain a list: {path}")
    return data


def save_json_list(path: str, data) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, allow_nan=False)


def build_triplet_projection_loaders(args, tokenizer, device):
    train_data = maybe_sample_triplets(
        load_json_list(args.reference_file), args.max_reference_file_triplets, args.seed
    )
    val_data = load_json_list(args.validation_file)
    test_data = load_json_list(args.test_file)

    train_triplets = prepare_triplets_with_offline_degree(train_data, args, device, "train")
    if is_triplet_format(val_data):
        val_data = prepare_triplets_with_offline_degree(val_data, args, device, "validation")
        val_samples = convert_triplets_to_single_samples(val_data)
    else:
        val_samples = val_data
    if is_triplet_format(test_data):
        test_data = prepare_triplets_with_offline_degree(test_data, args, device, "test")
        test_samples = convert_triplets_to_single_samples(test_data)
    else:
        test_samples = test_data

    if args.output_dir:
        save_json_list(os.path.join(args.output_dir, "train_triplets_with_degree.json"), train_triplets)
        if is_triplet_format(val_data):
            save_json_list(os.path.join(args.output_dir, "val_triplets_with_degree.json"), val_data)
        if is_triplet_format(test_data):
            save_json_list(os.path.join(args.output_dir, "test_triplets_with_degree.json"), test_data)

    train_dataset = TripletEditDataset(train_triplets, tokenizer, args.backbone_max_length)
    val_dataset = SingleTextEditDataset(val_samples, tokenizer, args.backbone_max_length)
    test_dataset = SingleTextEditDataset(test_samples, tokenizer, args.backbone_max_length)

    generator = torch.Generator().manual_seed(args.seed)
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
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return train_loader, val_loader, test_loader
