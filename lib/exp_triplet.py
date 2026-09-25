import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Set random seeds used by the triplet projection experiments."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def normalize_label(label):
    """Map labels to: 0=human, 1=edited, 2=AI."""
    if isinstance(label, str):
        value = label.strip().lower()
        if value in {
            "human", "human_written", "pure-human", "real", "written-by-human",
        }:
            return 0
        if value in {
            "edited", "ai_edited", "edit", "human-edited", "ai-edited",
            "mixed", "hybrid", "partially-edited",
        }:
            return 1
        if value in {
            "ai", "ai_generated", "llm", "machine", "generated", "pure-ai",
        }:
            return 2
    elif isinstance(label, (int, np.integer)) and int(label) in {0, 1, 2}:
        return int(label)

    raise ValueError(f"Unsupported label format: {label}")


def label_id_to_name(label_id: int) -> str:
    return {0: "human", 1: "edited", 2: "ai"}[int(label_id)]

import math


def compute_edited_degree_stats(prepared, split_name):
    """
    统计当前 split 中所有有效 edited_degree。

    方差和标准差采用总体统计：
        variance = sum((x - mean) ** 2) / N
        std = sqrt(variance)
    """
    edited_degrees = []

    for index, item in enumerate(prepared):
        if not isinstance(item, dict):
            logger.warning(
                f"[{split_name}] Sample {index} is not a dict, skipped."
            )
            continue

        value = item.get("edited_degree")

        if value is None:
            logger.warning(
                f"[{split_name}] Sample {index} has no edited_degree."
            )
            continue

        try:
            value = float(value)
        except (TypeError, ValueError):
            logger.warning(
                f"[{split_name}] Invalid edited_degree at sample "
                f"{index}: {value!r}"
            )
            continue

        if not math.isfinite(value):
            logger.warning(
                f"[{split_name}] Non-finite edited_degree at sample "
                f"{index}: {value}"
            )
            continue

        edited_degrees.append(value)

    count = len(edited_degrees)

    if count == 0:
        return {
            "split": split_name,
            "total_triplets": len(prepared),
            "valid_edited_degree_count": 0,
            "edited_degree_mean": None,
            "edited_degree_min": None,
            "edited_degree_max": None,
            "edited_degree_variance": None,
            "edited_degree_std": None,
        }

    # 使用 math.fsum 减少浮点数累加误差
    mean_value = math.fsum(edited_degrees) / count

    # 总体方差：除以 N
    variance_value = math.fsum(
        (value - mean_value) ** 2
        for value in edited_degrees
    ) / count

    std_value = math.sqrt(variance_value)

    return {
        "split": split_name,
        "total_triplets": len(prepared),
        "valid_edited_degree_count": count,
        "edited_degree_mean": mean_value,
        "edited_degree_min": min(edited_degrees),
        "edited_degree_max": max(edited_degrees),
        "edited_degree_variance": variance_value,
        "edited_degree_std": std_value,
    }



