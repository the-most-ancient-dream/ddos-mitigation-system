"""Train the live-compatible DDoS Random Forest on a balanced binary CSV.

This experiment deliberately loads one CSV only. It keeps the classifier and
joblib payload used by the application, while splitting rows and fitting
missing-value/scaling statistics without looking at the held-out test rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .feature_extractor import FEATURE_NAMES
from .ml_classifier import DDoSClassifier


LOGGER = logging.getLogger(__name__)
SEED = 42
CLASS_NAMES = ("BENIGN", "ATTACK")
FORBIDDEN_FEATURE_NAMES = {"label", "original_label"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_report_path(path: Path) -> str:
    """Return a repository-relative path, avoiding machine-specific report paths."""
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def _find_target_column(frame: pd.DataFrame) -> str:
    """Select the binary Label column, even if a CSV contains a legacy label."""
    candidates = []
    for column in frame.columns:
        normalized = str(column).strip().casefold()
        if normalized == "label" or re.fullmatch(r"label\.\d+", normalized):
            values = set(
                frame[column].dropna().astype(str).str.strip().str.upper().unique()
            )
            if values == {"BENIGN", "ATTACK"}:
                candidates.append(column)

    if not candidates:
        raise ValueError(
            "Could not find a Label column containing exactly BENIGN and ATTACK."
        )
    # Prefer the exact target name stated for this dataset.
    exact = [column for column in candidates if str(column) == "Label"]
    target = exact[-1] if exact else candidates[-1]
    target_values = set(
        frame[target].dropna().astype(str).str.strip().str.upper().unique()
    )
    if target_values != {"BENIGN", "ATTACK"}:
        raise ValueError(f"Unexpected binary labels in {target!r}: {target_values}")
    return target


def _extract_live_features(
    frame: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    """Keep only numeric columns emitted by the project's live feature extractor."""
    normalized_columns: dict[str, str] = {}
    for column in frame.columns:
        name = str(column).strip()
        if name.casefold() in FORBIDDEN_FEATURE_NAMES:
            continue
        normalized_columns.setdefault(name.casefold(), column)

    selected_sources: list[str] = []
    selected_names: list[str] = []
    for feature in FEATURE_NAMES:
        source = normalized_columns.get(feature.casefold())
        if source is not None:
            selected_sources.append(source)
            selected_names.append(feature)

    if not selected_sources:
        raise ValueError("No live-compatible CIC flow features were found in the CSV.")

    raw_features = frame.loc[:, selected_sources].copy()
    raw_features.columns = selected_names
    # Coercion handles numeric values stored as text; unsupported columns are
    # excluded by the fixed live feature schema above.
    for name in selected_names:
        raw_features[name] = pd.to_numeric(raw_features[name], errors="coerce")
    raw_features = raw_features.replace([np.inf, -np.inf], np.nan)

    labels = frame[target_column].astype(str).str.strip().str.upper().to_numpy()
    if not set(np.unique(labels)).issubset({"BENIGN", "ATTACK"}):
        raise ValueError("Only BENIGN and ATTACK labels are supported.")
    if pd.isna(labels).any():
        raise ValueError("The target Label column contains missing values.")
    y = (labels == "ATTACK").astype(np.uint8)

    original_label_column = next(
        (
            column for column in frame.columns
            if str(column).strip().casefold() == "original_label"
        ),
        None,
    )
    if original_label_column is None:
        original_labels = np.where(y == 1, "ATTACK", "BENIGN")
    else:
        original_labels = (
            frame[original_label_column].fillna("UNKNOWN").astype(str).str.strip().to_numpy()
        )
    attack_types, attack_type_counts = np.unique(original_labels[y == 1], return_counts=True)

    all_label_columns = [
        str(column)
        for column in frame.columns
        if str(column).strip().casefold() == "label"
        or str(column).strip().casefold() == "original_label"
        or re.fullmatch(r"label\.\d+", str(column).strip().casefold())
    ]
    dropped_columns = [
        str(column)
        for column in frame.columns
        if column not in selected_sources and column != target_column
    ]
    nonnumeric_columns = [
        str(column)
        for column in frame.columns
        if column != target_column
        and not pd.api.types.is_numeric_dtype(frame[column])
    ]

    if any(
        name.strip().casefold() in FORBIDDEN_FEATURE_NAMES
        for name in selected_names
    ):
        raise AssertionError("A target/attack-label column entered the features.")

    info = {
        "target_column": str(target_column),
        "label_columns_excluded": all_label_columns,
        "nonnumeric_columns_excluded": nonnumeric_columns,
        "other_columns_excluded": [
            name for name in dropped_columns if name not in nonnumeric_columns
        ],
        "feature_names": selected_names,
        "nonfinite_cells_before_imputation": int(raw_features.isna().sum().sum()),
        "nonfinite_by_feature": {
            str(name): int(count)
            for name, count in raw_features.isna().sum().items()
            if count
        },
        "rows": int(len(frame)),
        "class_counts": {
            "BENIGN": int(np.count_nonzero(y == 0)),
            "ATTACK": int(np.count_nonzero(y == 1)),
        },
        "attack_type_counts": {
            str(name): int(count) for name, count in zip(attack_types, attack_type_counts)
        },
        "original_label_column_for_evaluation_only": (
            str(original_label_column) if original_label_column is not None else None
        ),
    }
    return raw_features, y, original_labels, info


def _split_indices(
    x: pd.DataFrame, y: np.ndarray, original_labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Stratify by binary label while keeping identical feature rows together.

    A plain stratified row split put many repeated 64-feature vectors on both
    sides of the test boundary. Greedily assign whole exact-vector groups to
    70/15/15 binary class targets; original attack type only breaks ties.
    """
    row_hashes = pd.util.hash_pandas_object(x, index=False).to_numpy(dtype=np.uint64)
    unique_hashes, group_ids, group_sizes = np.unique(
        row_hashes, return_inverse=True, return_counts=True
    )
    strata = np.where(y == 0, "BENIGN", original_labels.astype(str))
    _, row_strata = np.unique(strata, return_inverse=True)
    stratum_counts = np.bincount(row_strata)
    group_class_counts = np.zeros((len(unique_hashes), len(stratum_counts)), dtype=np.int64)
    np.add.at(group_class_counts, (group_ids, row_strata), 1)
    group_binary_counts = np.zeros((len(unique_hashes), 2), dtype=np.int64)
    np.add.at(group_binary_counts, (group_ids, y), 1)

    fractions = np.asarray([0.70, 0.15, 0.15], dtype=np.float64)
    targets = fractions[:, None] * stratum_counts[None, :]
    current = np.zeros_like(targets)
    binary_targets = fractions[:, None] * np.bincount(y, minlength=2)[None, :]
    binary_current = np.zeros_like(binary_targets)
    group_assignment = np.full(len(unique_hashes), -1, dtype=np.int8)

    rng = np.random.default_rng(SEED)
    tie_break_order = rng.permutation(len(unique_hashes))
    order = tie_break_order[np.argsort(-group_sizes[tie_break_order], kind="stable")]
    denominators = np.maximum(targets, 1.0)
    binary_denominators = np.maximum(binary_targets, 1.0)
    for group_id in order:
        counts = group_class_counts[group_id]
        binary_counts = group_binary_counts[group_id]
        best_split = 0
        best_binary_delta = float("inf")
        best_type_delta = float("inf")
        for split in range(3):
            type_before = np.square((current[split] - targets[split]) / denominators[split]).sum()
            type_after = np.square(
                (current[split] + counts - targets[split]) / denominators[split]
            ).sum()
            binary_before = np.square(
                (binary_current[split] - binary_targets[split]) / binary_denominators[split]
            ).sum()
            binary_after = np.square(
                (binary_current[split] + binary_counts - binary_targets[split])
                / binary_denominators[split]
            ).sum()
            binary_delta = float(binary_after - binary_before)
            type_delta = float(type_after - type_before)
            # Minimize binary class-count error first; only use attack-category
            # balance to break ties between equally good binary allocations.
            if (
                binary_delta < best_binary_delta - 1e-12
                or (
                    abs(binary_delta - best_binary_delta) <= 1e-12
                    and type_delta < best_type_delta
                )
            ):
                best_binary_delta = binary_delta
                best_type_delta = type_delta
                best_split = split
        group_assignment[group_id] = best_split
        current[best_split] += counts
        binary_current[best_split] += binary_counts

    row_splits = group_assignment[group_ids]
    train_indices = np.flatnonzero(row_splits == 0)
    validation_indices = np.flatnonzero(row_splits == 1)
    test_indices = np.flatnonzero(row_splits == 2)
    split_hashes = [set(row_hashes[index].tolist()) for index in (train_indices, validation_indices, test_indices)]
    if any(
        split_hashes[left].intersection(split_hashes[right])
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise AssertionError("An identical feature vector crossed split boundaries.")

    mixed_group_mask = np.count_nonzero(group_binary_counts, axis=1) > 1
    mixed_groups = int(np.count_nonzero(mixed_group_mask))
    mixed_group_rows = int(group_sizes[mixed_group_mask].sum())
    diagnostics = {
        "unique_feature_vectors": int(len(unique_hashes)),
        "duplicate_rows_beyond_first": int(len(y) - len(unique_hashes)),
        "identical_feature_groups_with_both_labels": mixed_groups,
        "rows_in_mixed_label_feature_groups": mixed_group_rows,
        "identical_feature_vectors_crossing_splits": 0,
    }
    return train_indices, validation_indices, test_indices, diagnostics


def _class_counts(y: np.ndarray) -> dict[str, int]:
    return {
        "BENIGN": int(np.count_nonzero(y == 0)),
        "ATTACK": int(np.count_nonzero(y == 1)),
    }


def _attack_type_split_counts(
    indices: np.ndarray, y: np.ndarray, original_labels: np.ndarray
) -> dict[str, int]:
    mask = y[indices] == 1
    names, counts = np.unique(original_labels[indices][mask], return_counts=True)
    return {str(name): int(count) for name, count in zip(names, counts)}


def _confusion_plot(
    output_dir: Path,
    filename: str,
    matrix: np.ndarray,
    title: str,
    normalize: bool = False,
) -> None:
    values = matrix.astype(float)
    if normalize:
        totals = values.sum(axis=1, keepdims=True)
        values = np.divide(values, totals, out=np.zeros_like(values), where=totals != 0)
    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    image = ax.imshow(values, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        xlabel="Predicted label", ylabel="Actual label", title=title,
    )
    cutoff = values.max() / 2 if values.size else 0
    for row in range(2):
        for column in range(2):
            label = f"{values[row, column]:.1%}" if normalize else f"{int(matrix[row, column]):,}"
            ax.text(
                column, row, label, ha="center", va="center", fontsize=12,
                color="white" if values[row, column] > cutoff else "black",
            )
    fig.colorbar(image, ax=ax, label="Row percentage" if normalize else "Samples")
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_plots(
    output_dir: Path,
    classifier: DDoSClassifier,
    dataset_info: dict[str, Any],
    split_counts: dict[str, dict[str, int]],
    split_sizes: dict[str, int],
    train_accuracy: float,
    validation_accuracy: float,
    test_y: np.ndarray,
    test_pred: np.ndarray,
    test_scores: np.ndarray,
    validation_y: np.ndarray,
    validation_pred: np.ndarray,
    report: dict[str, Any],
    original_test_labels: np.ndarray,
    feature_shift: dict[str, float],
    feature_distributions: dict[str, tuple[np.ndarray, np.ndarray]],
    thresholds: np.ndarray,
    threshold_metrics: dict[str, list[float]],
    attack_type_metrics: dict[str, dict[str, float]],
    calibration: dict[str, Any],
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []

    def save(fig: plt.Figure, filename: str) -> None:
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160, bbox_inches="tight")
        plt.close(fig)
        files.append(filename)

    # Dataset overview charts (01–07).
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    class_names = list(dataset_info["class_counts"])
    values = [dataset_info["class_counts"][name] for name in class_names]
    bars = ax.bar(class_names, values, color=["#4472C4", "#ED7D31"])
    ax.set_ylabel("Rows")
    ax.set_title("Dataset Class Distribution")
    ax.bar_label(bars, labels=[f"{v:,}" for v in values], padding=3)
    save(fig, "01_dataset_class_distribution.png")

    attack_counts = dataset_info["attack_type_counts"]
    fig, ax = plt.subplots(figsize=(9, 6.2))
    names = list(attack_counts)
    values = [attack_counts[name] for name in names]
    bars = ax.barh(names[::-1], values[::-1], color="#ED7D31")
    ax.set_xlabel("ATTACK rows")
    ax.set_title(f"Attack Type Distribution ({len(names)} types; original labels used for reporting only)")
    ax.bar_label(bars, labels=[f"{v:,}" for v in values[::-1]], padding=3)
    save(fig, "02_attack_type_distribution.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    split_names = list(split_sizes)
    sizes = [split_sizes[name] for name in split_names]
    bars = ax.bar(split_names, sizes, color=["#4472C4", "#70AD47", "#ED7D31"])
    ax.set_ylabel("Rows")
    ax.set_title("Train / Validation / Test Row Distribution")
    ax.bar_label(bars, labels=[f"{v:,} ({v / sum(sizes):.1%})" for v in sizes], padding=3)
    save(fig, "03_split_row_distribution.png")

    invalid = dataset_info["nonfinite_by_feature"]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    if invalid:
        names = list(invalid)
        values = [invalid[name] for name in names]
        bars = ax.bar(names, values, color="#C0504D")
        ax.bar_label(bars, labels=[f"{v:,}" for v in values], padding=3)
    else:
        ax.text(0.5, 0.5, "No non-finite values found", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
    ax.set_ylabel("NaN / infinite cells")
    ax.set_title("Non-finite Values by Retained Feature")
    save(fig, "04_nonfinite_values_by_feature.png")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(["Available in live schema", "Retained for model"], [len(FEATURE_NAMES), len(dataset_info["feature_names"])], color=["#A5A5A5", "#70AD47"])
    ax.set_ylim(0, max(len(FEATURE_NAMES), len(dataset_info["feature_names"])) * 1.25)
    ax.set_ylabel("Feature count")
    ax.set_title("Retained Model Feature Count")
    for i, count in enumerate([len(FEATURE_NAMES), len(dataset_info["feature_names"])]):
        ax.text(i, count, str(count), ha="center", va="bottom", fontsize=15)
    save(fig, "05_retained_model_features.png")

    duplicates = dataset_info["duplicate_feature_diagnostics"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    diagnostic_names = ["Unique vectors", "Duplicate rows\nbeyond first", "Mixed-label rows", "Cross-split overlap"]
    diagnostic_values = [
        duplicates["unique_feature_vectors"],
        duplicates["duplicate_rows_beyond_first"],
        duplicates["rows_in_mixed_label_feature_groups"],
        duplicates["identical_feature_vectors_crossing_splits"],
    ]
    bars = ax.bar(diagnostic_names, diagnostic_values, color=["#4472C4", "#ED7D31", "#C0504D", "#70AD47"])
    ax.set_ylabel("Rows / groups")
    ax.set_title("Duplicate-feature Diagnostics")
    ax.bar_label(bars, labels=[f"{v:,}" for v in diagnostic_values], padding=3)
    save(fig, "06_duplicate_feature_diagnostics.png")

    fig, ax = plt.subplots(figsize=(8, 5.2))
    names = list(split_counts)
    positions = np.arange(len(names))
    width = 0.34
    benign = [split_counts[name]["BENIGN"] for name in names]
    attack = [split_counts[name]["ATTACK"] for name in names]
    ax.bar(positions - width / 2, benign, width, label="BENIGN", color="#4472C4")
    ax.bar(positions + width / 2, attack, width, label="ATTACK", color="#ED7D31")
    ax.set_xticks(positions, names)
    ax.set_ylabel("Rows")
    ax.set_title("Class Counts Across All Splits")
    ax.legend()
    save(fig, "07_class_counts_across_splits.png")

    # Validation and final-test confusion matrices (counts and row percentages).
    validation_matrix = confusion_matrix(validation_y, validation_pred, labels=[0, 1])
    test_matrix = confusion_matrix(test_y, test_pred, labels=[0, 1])
    _confusion_plot(output_dir, "08_validation_confusion_counts.png", validation_matrix, "Validation Confusion Matrix — Counts")
    _confusion_plot(output_dir, "09_validation_confusion_percent.png", validation_matrix, "Validation Confusion Matrix — Row Percentages", normalize=True)
    _confusion_plot(output_dir, "10_test_confusion_counts.png", test_matrix, "Held-out Test Confusion Matrix — Counts")
    _confusion_plot(output_dir, "11_test_confusion_percent.png", test_matrix, "Held-out Test Confusion Matrix — Row Percentages", normalize=True)
    files.extend([
        "08_validation_confusion_counts.png", "09_validation_confusion_percent.png",
        "10_test_confusion_counts.png", "11_test_confusion_percent.png",
    ])

    # Fit-vs-holdout accuracy and generalization gap diagnostics.
    accuracy_values = [train_accuracy, validation_accuracy, accuracy_score(test_y, test_pred)]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    bars = ax.bar(["Training", "Validation", "Test"], accuracy_values, color=["#4472C4", "#70AD47", "#ED7D31"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Accuracy")
    ax.set_title("Training vs Validation vs Test Accuracy")
    ax.bar_label(bars, labels=[f"{v:.3%}" for v in accuracy_values], padding=3)
    save(fig, "12_accuracy_by_split.png")

    gaps = {
        "Train → Validation": (train_accuracy - validation_accuracy) * 100,
        "Validation → Test": (validation_accuracy - accuracy_values[2]) * 100,
        "Train → Test": (train_accuracy - accuracy_values[2]) * 100,
    }
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(list(gaps), list(gaps.values()), color=["#4472C4", "#ED7D31", "#A5A5A5"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Accuracy difference (percentage points)")
    ax.set_title("Generalization Gaps")
    ax.bar_label(bars, labels=[f"{value:+.3f} pp" for value in gaps.values()], padding=3)
    save(fig, "13_generalization_gaps.png")

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    positions = np.arange(2)
    width = 0.24
    for offset, metric_name, color in [
        (-width, "precision", "#4472C4"),
        (0, "recall", "#ED7D31"),
        (width, "f1-score", "#70AD47"),
    ]:
        values = [report[label][metric_name] for label in CLASS_NAMES]
        ax.bar(positions + offset, values, width, label=metric_name.title(), color=color)
    ax.set_xticks(positions, CLASS_NAMES)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Test score")
    ax.set_title("Held-out Test Precision, Recall, and F1 by Class")
    ax.legend()
    save(fig, "14_per_class_metrics.png")

    aggregate_metrics = {
        "Accuracy": accuracy_score(test_y, test_pred),
        "Precision (ATTACK)": report["ATTACK"]["precision"],
        "Recall (ATTACK)": report["ATTACK"]["recall"],
        "F1 (ATTACK)": report["ATTACK"]["f1-score"],
    }
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(
        list(aggregate_metrics), list(aggregate_metrics.values()),
        color=["#4472C4", "#70AD47", "#ED7D31", "#A5A5A5"],
    )
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Held-out test score")
    ax.set_title("Accuracy / Precision / Recall / F1 Comparison")
    ax.tick_params(axis="x", rotation=12)
    ax.bar_label(bars, labels=[f"{value:.3%}" for value in aggregate_metrics.values()], padding=3)
    save(fig, "25_test_metrics_comparison.png")

    false_positive_rate, true_positive_rate, _ = roc_curve(test_y, test_scores)
    roc_auc = roc_auc_score(test_y, test_scores)
    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    ax.plot(false_positive_rate, true_positive_rate, lw=2, label=f"ROC-AUC = {roc_auc:.6f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="No-skill reference")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="Held-out Test ROC Curve")
    ax.legend(loc="lower right")
    save(fig, "15_roc_curve.png")

    precision, recall, _ = precision_recall_curve(test_y, test_scores)
    average_precision = average_precision_score(test_y, test_scores)
    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    ax.plot(recall, precision, lw=2, label=f"Average precision = {average_precision:.6f}")
    ax.set(xlabel="Recall (ATTACK)", ylabel="Precision (ATTACK)", title="Held-out Test Precision–Recall Curve")
    ax.legend(loc="best")
    save(fig, "16_precision_recall_curve.png")

    fig, ax = plt.subplots(figsize=(8, 5.2))
    for metric_name, color in [("precision", "#4472C4"), ("recall", "#ED7D31"), ("f1", "#70AD47")]:
        ax.plot(thresholds, threshold_metrics[metric_name], label=metric_name.title(), color=color)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="Default threshold 0.50")
    ax.set(xlabel="ATTACK probability threshold", ylabel="Score", ylim=(0, 1.02), title="Test Threshold vs Precision / Recall / F1")
    ax.legend()
    save(fig, "17_threshold_precision_recall_f1.png")

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(thresholds, threshold_metrics["fpr"], label="False Positive Rate", color="#C0504D")
    ax.plot(thresholds, threshold_metrics["fnr"], label="False Negative Rate", color="#4472C4")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="Default threshold 0.50")
    ax.set(xlabel="ATTACK probability threshold", ylabel="Rate", ylim=(0, 1.02), title="Test Threshold vs False Positive / False Negative Rate")
    ax.legend()
    save(fig, "18_threshold_false_rates.png")

    fig, ax = plt.subplots(figsize=(8, 5.2))
    bins = np.linspace(0, 1, 41)
    ax.hist(test_scores[test_y == 0], bins=bins, density=True, alpha=0.55, label="Actual BENIGN", color="#4472C4")
    ax.hist(test_scores[test_y == 1], bins=bins, density=True, alpha=0.55, label="Actual ATTACK", color="#ED7D31")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="Default threshold 0.50")
    ax.set(xlabel="Predicted ATTACK probability", ylabel="Density", title="Predicted Probability Distribution on Test Rows")
    ax.legend()
    save(fig, "19_probability_distribution.png")

    # Original_Label is used only to group test outcomes for this report.
    attack_names = list(attack_type_metrics)
    fig, ax = plt.subplots(figsize=(10, max(6, len(attack_names) * 0.32)))
    recalls = [attack_type_metrics[name]["recall"] for name in attack_names]
    supports = [int(attack_type_metrics[name]["support"]) for name in attack_names]
    bars = ax.barh(attack_names[::-1], recalls[::-1], color="#70AD47")
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("ATTACK recall")
    ax.set_title("Held-out Test ATTACK Recall by Original Attack Type")
    ax.bar_label(bars, labels=[f"{recall:.1%} (n={n:,})" for recall, n in zip(recalls[::-1], supports[::-1])], padding=3, fontsize=8)
    save(fig, "20_attack_recall_by_type.png")

    missed = [int(attack_type_metrics[name]["missed"]) for name in attack_names]
    fig, ax = plt.subplots(figsize=(10, max(6, len(attack_names) * 0.32)))
    bars = ax.barh(attack_names[::-1], missed[::-1], color="#C0504D")
    ax.set_xlabel("ATTACK rows predicted BENIGN")
    ax.set_title("Missed Held-out Test Attacks by Type")
    ax.bar_label(bars, labels=[f"{value:,}" for value in missed[::-1]], padding=3)
    save(fig, "21_missed_attacks_by_type.png")

    fig, axes = plt.subplots(4, 3, figsize=(15, 14))
    for ax, (feature, samples) in zip(axes.flat, feature_distributions.items()):
        train_values, test_values = samples
        transform = lambda v: np.sign(v) * np.log10(np.abs(v) + 1.0)
        ax.hist(transform(train_values), bins=35, density=True, alpha=0.50, label="Training", color="#4472C4")
        ax.hist(transform(test_values), bins=35, density=True, alpha=0.50, label="Test", color="#ED7D31")
        ax.set_title(f"{feature}\nKS={feature_shift[feature]:.3f}", fontsize=9)
        ax.set_xlabel("signed log10(|value| + 1)", fontsize=8)
    for ax in axes.flat[len(feature_distributions):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Training vs Test Feature Distributions (12 Largest KS Shifts)", y=1.01)
    save(fig, "22_train_test_feature_distributions.png")

    importance = classifier.get_feature_importance(20)
    sorted_importance = list(importance.items())[::-1]
    fig, ax = plt.subplots(figsize=(9, max(5.0, len(sorted_importance) * 0.28)))
    ax.barh(
        [item[0] for item in sorted_importance],
        [item[1] for item in sorted_importance],
        color="#5B9BD5",
    )
    ax.set_xlabel("Random Forest feature importance")
    ax.set_title("Top 20 Model Feature Importances")
    save(fig, "26_top_feature_importance.png")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    names = ["Duplicate rows", "Mixed-label rows", "Vectors crossing splits"]
    values = [
        dataset_info["duplicate_feature_diagnostics"]["duplicate_rows_beyond_first"],
        dataset_info["duplicate_feature_diagnostics"]["rows_in_mixed_label_feature_groups"],
        dataset_info["duplicate_feature_diagnostics"]["identical_feature_vectors_crossing_splits"],
    ]
    bars = ax.bar(names, values, color=["#ED7D31", "#C0504D", "#70AD47"])
    ax.set_ylabel("Count")
    ax.set_title("Duplicate / Split Leakage Diagnostic")
    ax.bar_label(bars, labels=[f"{value:,}" for value in values], padding=3)
    save(fig, "23_duplicate_split_leakage_diagnostic.png")

    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.plot(calibration["mean_predicted_probability"], calibration["observed_attack_fraction"], marker="o", lw=2, label="Random Forest")
    ax.set(xlabel="Mean predicted ATTACK probability", ylabel="Observed ATTACK fraction", title=f"Test Probability Calibration (Brier={calibration['brier_score']:.5f})", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    save(fig, "24_probability_calibration.png")

    return files


def run_training(
    dataset_path: Path,
    model_path: Path,
    results_dir: Path,
) -> dict[str, Any]:
    dataset_path = dataset_path.resolve()
    model_path = model_path.resolve()
    results_dir = results_dir.resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if dataset_path.resolve() == model_path:
        raise ValueError("Model output path must be separate from the source dataset.")

    LOGGER.info("Reading only the requested dataset: %s", dataset_path)
    dataset_hash = _sha256(dataset_path)
    frame = pd.read_csv(dataset_path, low_memory=False, encoding="utf-8")
    target_column = _find_target_column(frame)
    raw_x, y, original_labels, dataset_info = _extract_live_features(frame, target_column)
    del frame

    if dataset_info["class_counts"] != {"BENIGN": 100000, "ATTACK": 100000}:
        LOGGER.warning("Dataset counts differ from the supplied description: %s", dataset_info["class_counts"])

    train_i, val_i, test_i, duplicate_diagnostics = _split_indices(raw_x, y, original_labels)
    split_counts = {
        "Training": _class_counts(y[train_i]),
        "Validation": _class_counts(y[val_i]),
        "Testing": _class_counts(y[test_i]),
    }
    split_sizes = {name: sum(counts.values()) for name, counts in split_counts.items()}
    attack_type_split_counts = {
        "Training": _attack_type_split_counts(train_i, y, original_labels),
        "Validation": _attack_type_split_counts(val_i, y, original_labels),
        "Testing": _attack_type_split_counts(test_i, y, original_labels),
    }

    # Fit imputation statistics and scaling on training rows only.
    x_train_raw = raw_x.iloc[train_i]
    x_test_raw = raw_x.iloc[test_i]
    medians = x_train_raw.median(axis=0, skipna=True).fillna(0.0)
    x_train = x_train_raw.fillna(medians).to_numpy(dtype=np.float64, copy=True)
    x_val = raw_x.iloc[val_i].fillna(medians).to_numpy(dtype=np.float64, copy=True)
    x_test = raw_x.iloc[test_i].fillna(medians).to_numpy(dtype=np.float64, copy=True)

    # Measure train/test feature-distribution shifts on raw finite values. This
    # audit is descriptive only and does not feed feature selection or fitting.
    feature_shift: dict[str, float] = {}
    for name in x_train_raw.columns:
        train_values = x_train_raw[name].to_numpy(dtype=np.float64, copy=False)
        test_values = x_test_raw[name].to_numpy(dtype=np.float64, copy=False)
        train_values = train_values[np.isfinite(train_values)]
        test_values = test_values[np.isfinite(test_values)]
        feature_shift[name] = float(ks_2samp(train_values, test_values).statistic)
    shift_features = sorted(feature_shift, key=feature_shift.get, reverse=True)[:12]
    feature_distributions = {
        name: (
            x_train_raw[name].to_numpy(dtype=np.float64, copy=True),
            x_test_raw[name].to_numpy(dtype=np.float64, copy=True),
        )
        for name in shift_features
    }
    del raw_x, x_train_raw, x_test_raw

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32, copy=False)
    x_val = scaler.transform(x_val).astype(np.float32, copy=False)
    x_test = scaler.transform(x_test).astype(np.float32, copy=False)
    y_train, y_val, y_test = y[train_i], y[val_i], y[test_i]

    LOGGER.info("Splits: %s", json.dumps(split_counts))
    LOGGER.info("Fitting existing Random Forest architecture on %d features", x_train.shape[1])
    classifier = DDoSClassifier(
        model_type="random_forest",
        n_estimators=100,
        max_depth=15,
        n_jobs=-1,
    )
    classifier.scaler = scaler
    classifier.imputation_values = medians.to_numpy(dtype=np.float64)
    training_start = time.perf_counter()
    classifier.train(
        X_train=x_train,
        y_train=y_train,
        X_val=x_val,
        y_val=y_val,
        feature_names=list(dataset_info["feature_names"]),
    )
    training_seconds = time.perf_counter() - training_start

    train_pred = classifier.model.predict(x_train)
    validation_pred = classifier.model.predict(x_val)
    # This is the single final held-out test evaluation; no test values are used
    # for fitting imputation/scaling, training, or choosing model parameters.
    test_pred = classifier.model.predict(x_test)
    classes = list(classifier.model.classes_)
    positive_index = classes.index(1)
    test_scores = classifier.model.predict_proba(x_test)[:, positive_index]
    test_metrics = {
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "precision_attack": float(precision_score(y_test, test_pred, pos_label=1, zero_division=0)),
        "recall_attack": float(recall_score(y_test, test_pred, pos_label=1, zero_division=0)),
        "f1_attack": float(f1_score(y_test, test_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, test_scores)),
        "average_precision_attack": float(average_precision_score(y_test, test_scores)),
        "confusion_matrix": confusion_matrix(y_test, test_pred, labels=[0, 1]).tolist(),
    }
    report_text = classification_report(
        y_test,
        test_pred,
        labels=[0, 1],
        target_names=list(CLASS_NAMES),
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        y_test,
        test_pred,
        labels=[0, 1],
        target_names=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )

    train_accuracy = float(accuracy_score(y_train, train_pred))
    validation_accuracy = float(accuracy_score(y_val, validation_pred))
    validation_matrix = confusion_matrix(y_val, validation_pred, labels=[0, 1])
    accuracy_gaps = {
        "train_minus_validation": train_accuracy - validation_accuracy,
        "validation_minus_test": validation_accuracy - test_metrics["accuracy"],
        "train_minus_test": train_accuracy - test_metrics["accuracy"],
    }
    if max(accuracy_gaps.values()) >= 0.05:
        generalization_note = "At least one accuracy gap is 5 percentage points or larger; inspect for overfitting or distribution differences."
    else:
        generalization_note = "No large accuracy gap (5 percentage points or larger) was observed across train, validation, and test."

    original_test_labels = original_labels[test_i]
    attack_type_metrics: dict[str, dict[str, float]] = {}
    for attack_type in sorted(np.unique(original_test_labels[y_test == 1])):
        mask = (y_test == 1) & (original_test_labels == attack_type)
        support = int(mask.sum())
        detected = int(np.count_nonzero(test_pred[mask] == 1))
        attack_type_metrics[str(attack_type)] = {
            "support": support,
            "detected": detected,
            "missed": support - detected,
            "recall": float(detected / support) if support else 0.0,
        }

    thresholds = np.linspace(0.0, 1.0, 201)
    threshold_metrics = {name: [] for name in ("precision", "recall", "f1", "fpr", "fnr")}
    threshold_table = []
    for threshold in thresholds:
        threshold_pred = (test_scores >= threshold).astype(np.uint8)
        tn, fp, fn, tp = confusion_matrix(y_test, threshold_pred, labels=[0, 1]).ravel()
        precision = float(tp / (tp + fp)) if tp + fp else 0.0
        recall = float(tp / (tp + fn)) if tp + fn else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        fpr = float(fp / (fp + tn)) if fp + tn else 0.0
        fnr = float(fn / (fn + tp)) if fn + tp else 0.0
        for name, value in zip(threshold_metrics, (precision, recall, f1, fpr, fnr)):
            threshold_metrics[name].append(value)
        threshold_table.append({
            "threshold": float(threshold), "precision_attack": precision,
            "recall_attack": recall, "f1_attack": f1,
            "false_positive_rate": fpr, "false_negative_rate": fnr,
        })

    calibration_probability, calibration_observed = calibration_curve(
        y_test, test_scores, n_bins=10, strategy="quantile"
    )
    calibration = {
        "mean_predicted_probability": calibration_probability.tolist(),
        "observed_attack_fraction": calibration_observed.tolist(),
        "brier_score": float(brier_score_loss(y_test, test_scores)),
        "bins": int(len(calibration_probability)),
    }
    operating_index = int(np.argmin(np.abs(thresholds - 0.5)))
    test_metrics.update({
        "precision_benign": float(report_dict["BENIGN"]["precision"]),
        "recall_benign": float(report_dict["BENIGN"]["recall"]),
        "f1_benign": float(report_dict["BENIGN"]["f1-score"]),
        "brier_score": calibration["brier_score"],
        "false_positive_rate_at_0_5": threshold_metrics["fpr"][operating_index],
        "false_negative_rate_at_0_5": threshold_metrics["fnr"][operating_index],
        "validation_confusion_matrix": validation_matrix.tolist(),
    })

    dataset_info["sha256"] = dataset_hash
    dataset_info["split_counts"] = split_counts
    dataset_info["attack_type_split_counts"] = attack_type_split_counts
    dataset_info["duplicate_feature_diagnostics"] = duplicate_diagnostics
    dataset_info["training_rows"] = int(len(train_i))
    dataset_info["validation_rows"] = int(len(val_i))
    dataset_info["testing_rows"] = int(len(test_i))
    dataset_info["training_seconds"] = training_seconds
    dataset_info["seed"] = SEED
    dataset_info["preprocessing"] = (
        "Fixed live CIC flow feature schema; non-finite values imputed with per-feature "
        "training-split medians (all-missing training columns use 0); StandardScaler "
        "fit on training rows only."
    )
    dataset_info["training_medians_by_feature"] = {
        str(name): float(value) for name, value in medians.items()
    }
    dataset_info["model"] = {
        "type": "RandomForestClassifier",
        "n_estimators": 100,
        "max_depth": 15,
        "random_state": SEED,
        "class_weight": "balanced",
    }
    import sklearn
    dataset_info["environment"] = {
        "python": __import__("platform").python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }
    dataset_info["training_accuracy"] = train_accuracy
    dataset_info["validation_accuracy"] = validation_accuracy
    dataset_info["test_metrics"] = test_metrics
    dataset_info["accuracy_gaps"] = accuracy_gaps
    dataset_info["generalization_note"] = generalization_note
    dataset_info["validation_confusion_matrix"] = validation_matrix.tolist()
    dataset_info["test_classification_report"] = report_dict
    dataset_info["threshold_curves"] = threshold_table
    dataset_info["probability_calibration"] = calibration
    dataset_info["attack_type_test_metrics"] = attack_type_metrics
    dataset_info["train_test_feature_ks_statistics"] = feature_shift

    model_path.parent.mkdir(parents=True, exist_ok=True)
    if not classifier.save(str(model_path)):
        raise RuntimeError(f"Could not save the candidate model to {model_path}")

    graph_names = _save_plots(
        output_dir=results_dir,
        classifier=classifier,
        dataset_info=dataset_info,
        split_counts=split_counts,
        split_sizes=split_sizes,
        train_accuracy=train_accuracy,
        validation_accuracy=validation_accuracy,
        test_y=y_test,
        test_pred=test_pred,
        test_scores=test_scores,
        validation_y=y_val,
        validation_pred=validation_pred,
        report=report_dict,
        original_test_labels=original_test_labels,
        feature_shift=feature_shift,
        feature_distributions=feature_distributions,
        thresholds=thresholds,
        threshold_metrics=threshold_metrics,
        attack_type_metrics=attack_type_metrics,
        calibration=calibration,
    )
    dataset_info["graphs"] = graph_names

    matrix = test_metrics["confusion_matrix"]
    matrix_text = (
        "                 Predicted BENIGN  Predicted ATTACK\n"
        f"Actual BENIGN       {matrix[0][0]:>10,}       {matrix[0][1]:>10,}\n"
        f"Actual ATTACK       {matrix[1][0]:>10,}       {matrix[1][1]:>10,}"
    )
    validation_matrix_text = (
        "                 Predicted BENIGN  Predicted ATTACK\n"
        f"Actual BENIGN       {validation_matrix[0][0]:>10,}       {validation_matrix[0][1]:>10,}\n"
        f"Actual ATTACK       {validation_matrix[1][0]:>10,}       {validation_matrix[1][1]:>10,}"
    )
    test_row_percent = np.divide(
        np.asarray(matrix, dtype=float),
        np.asarray(matrix, dtype=float).sum(axis=1, keepdims=True),
        out=np.zeros((2, 2), dtype=float),
        where=np.asarray(matrix, dtype=float).sum(axis=1, keepdims=True) != 0,
    )
    lines = [
        "DDoS binary classifier — balanced_dataset_v2.csv",
        "=" * 58,
        f"Dataset: {dataset_path.name}",
        f"Dataset SHA-256: {dataset_hash}",
        f"Target column: {target_column}",
        f"Rows: {len(y):,}; numeric live-compatible features: {len(dataset_info['feature_names'])}",
        f"Excluded label columns: {', '.join(dataset_info['label_columns_excluded'])}",
        f"Excluded nonnumeric columns: {', '.join(dataset_info['nonnumeric_columns_excluded'])}",
        f"Non-finite feature cells imputed: {dataset_info['nonfinite_cells_before_imputation']:,}",
        "Non-finite cells by feature: " + json.dumps(dataset_info["nonfinite_by_feature"]),
        f"Retained model features: {len(dataset_info['feature_names'])}",
        "Attack-type counts: " + json.dumps(dataset_info["attack_type_counts"]),
        "Attack-type split counts: " + json.dumps(attack_type_split_counts),
        "",
        "Repeated-feature-vector split audit:",
        f"Unique vectors: {duplicate_diagnostics['unique_feature_vectors']:,}",
        f"Duplicate rows beyond first: {duplicate_diagnostics['duplicate_rows_beyond_first']:,}",
        f"Identical vectors with both labels: {duplicate_diagnostics['identical_feature_groups_with_both_labels']:,} groups / {duplicate_diagnostics['rows_in_mixed_label_feature_groups']:,} rows",
        "Identical feature vectors shared across splits: 0 (groups kept intact).",
        "",
        "Split sample counts (stratified by binary Label):",
    ]
    for split, counts in split_counts.items():
        lines.append(f"{split}: BENIGN = {counts['BENIGN']:,}; ATTACK = {counts['ATTACK']:,}")
    lines.extend([
        "",
        "Model: Random Forest, 100 trees, max_depth=15, random_state=42, class_weight=balanced",
        f"Training accuracy (fit rows): {train_accuracy:.6f}",
        f"Validation accuracy (held-out validation rows): {validation_accuracy:.6f}",
        f"Training time: {training_seconds:.2f} seconds",
        "",
        "Final held-out test metrics:",
        f"Accuracy: {test_metrics['accuracy']:.6f}",
        f"Precision (ATTACK): {test_metrics['precision_attack']:.6f}",
        f"Recall (ATTACK): {test_metrics['recall_attack']:.6f}",
        f"F1-score (ATTACK): {test_metrics['f1_attack']:.6f}",
        f"ROC-AUC: {test_metrics['roc_auc']:.6f}",
        f"Average precision (ATTACK): {test_metrics['average_precision_attack']:.6f}",
        f"Brier score: {calibration['brier_score']:.6f}",
        f"False positive rate at default threshold 0.50: {test_metrics['false_positive_rate_at_0_5']:.6f}",
        f"False negative rate at default threshold 0.50: {test_metrics['false_negative_rate_at_0_5']:.6f}",
        "",
        "Classification report (test set):",
        report_text.rstrip(),
        "",
        "Confusion matrix (test set):",
        matrix_text,
        "",
        "Normalized confusion matrix (test; rows actual, columns predicted):",
        json.dumps(test_row_percent.round(6).tolist()),
        "",
        "Validation confusion matrix (counts):",
        validation_matrix_text,
        "",
        "Train/validation/test accuracy gaps:",
        f"Train - validation: {accuracy_gaps['train_minus_validation']:+.6f}",
        f"Validation - test: {accuracy_gaps['validation_minus_test']:+.6f}",
        f"Train - test: {accuracy_gaps['train_minus_test']:+.6f}",
        generalization_note,
        "",
        "Attack recall / misses by original attack type (Original_Label used for evaluation only):",
        *[
            f"{attack_type}: support={int(values['support']):,}, recall={values['recall']:.4%}, missed={int(values['missed']):,}"
            for attack_type, values in attack_type_metrics.items()
        ],
        "",
        "Largest train/test feature-distribution shifts (two-sample KS statistic):",
        *[f"{name}: {feature_shift[name]:.6f}" for name in sorted(feature_shift, key=feature_shift.get, reverse=True)[:12]],
        "",
        "Threshold graphs are descriptive test-set curves. The model's evaluated decision threshold remains the default 0.50; no threshold was tuned on test data.",
        "Threshold metrics table is stored in metrics.json under threshold_curves.",
        "Calibration uses test-set probabilities with quantile bins; Brier score is included.",
        "",
        "Leakage controls:",
        "- Label and Original_Label (including whitespace variants) excluded from features.",
        "- The legacy per-attack label, identifiers, addresses, timestamp, and unsupported text fields were excluded.",
        "- Rows were assigned to exact 70/15/15 BENIGN/ATTACK group-aware splits before fitting imputation or scaling; attack type was only a secondary split tie-breaker.",
        "- Exact feature grouping makes some original attack types uneven in the held-out split; each per-type recall includes its test support for context.",
        "- Exact duplicate live-feature vectors were grouped so no such vector crosses splits.",
        "- Imputation medians and StandardScaler were fit only on training rows.",
        "- The fitted imputation medians and scaler are stored with the candidate model.",
        "- The test split was used only for final test metrics and plots.",
        "",
        "Training/validation accuracy and loss epoch curves were not generated: Random Forest has no epoch history.",
        f"Saved candidate model: {_portable_report_path(model_path)}",
        f"Graphs and this report: {_portable_report_path(results_dir)}",
        "",
        "Graphs:",
        *[f"- {name}" for name in graph_names],
    ])
    metrics_path = results_dir / "metrics.txt"
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (results_dir / "metrics.json").write_text(
        json.dumps(dataset_info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nModel Performance (held-out test set)")
    print("------------------------------------")
    print(f"Accuracy  : {test_metrics['accuracy']:.2%}")
    print(f"Precision : {test_metrics['precision_attack']:.2%} (ATTACK)")
    print(f"Recall    : {test_metrics['recall_attack']:.2%} (ATTACK)")
    print(f"F1 Score  : {test_metrics['f1_attack']:.2%} (ATTACK)")
    print(f"ROC-AUC   : {test_metrics['roc_auc']:.2%}")
    print(f"Avg. Precision: {test_metrics['average_precision_attack']:.2%}")
    print(f"Brier score: {calibration['brier_score']:.6f}")
    print("\nClassification Report:\n" + report_text)
    print("Confusion Matrix [actual rows; predicted columns; BENIGN, ATTACK]:")
    print(np.asarray(matrix))
    print("\nSample counts:")
    for split, counts in split_counts.items():
        print(f"{split}: BENIGN = {counts['BENIGN']:,}; ATTACK = {counts['ATTACK']:,}")
    print(f"\nOverfit check: {generalization_note}")
    print(f"Candidate model: {model_path}")
    print(f"Metrics and graphs: {results_dir}")
    return dataset_info


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the project's live-compatible DDoS classifier on one balanced binary CSV."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the balanced binary CSV (raw datasets are not included in the repository).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("data/models/ddos_classifier_balanced_v2.joblib"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("reports/model_results_balanced_v2"),
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    run_training(args.dataset, args.model_path, args.results_dir)


if __name__ == "__main__":
    main()
