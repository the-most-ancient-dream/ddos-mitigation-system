"""Generate dataset and model-evaluation figures for CIC-DDoS-2019 training."""

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


COLORS = {'BENIGN': '#2b8cbe', 'ATTACK': '#e34a33'}


def _label_column(columns) -> Optional[str]:
    return next((column for column in columns if str(column).strip().casefold() == 'label'), None)


def _normalized_labels(values: pd.Series) -> pd.Series:
    labels = values.astype(str).str.strip()
    return labels.map(lambda label: 'BENIGN' if label.upper() == 'BENIGN' else label)


def _split_name(path: Path) -> str:
    stem = path.stem.casefold()
    if stem.endswith(('-training', '_training')):
        return 'training'
    if stem.endswith(('-testing', '_testing')):
        return 'testing'
    return 'other'


def _read_labels(loader, path: Path) -> pd.Series:
    if path.suffix.casefold() == '.parquet':
        import pyarrow.parquet as parquet

        source_column = _label_column(parquet.read_schema(path).names)
        if source_column is None:
            return pd.Series(dtype=str)
        frame = pd.read_parquet(path, columns=[source_column])
    else:
        frame = loader.load_single_file(path, sample_size=None)
    column = _label_column(frame.columns)
    if column is None:
        return pd.Series(dtype=str)
    return _normalized_labels(frame[column])


def _save_figure(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches='tight')
    plt.close(fig)


def generate_dataset_charts(
    loader,
    output_dir: Path,
    samples_per_file: int = 50000,
    max_files: Optional[int] = None,
) -> Dict:
    """Write dataset overview charts before fitting the model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = loader.find_data_files()
    if max_files:
        files = files[:max_files]
    if not files:
        raise ValueError(f'No supported dataset files found in {loader.data_dir}')

    labels_by_split = defaultdict(list)
    file_rows = []
    sampled_training_frames = []
    for path in files:
        labels = _read_labels(loader, path)
        if labels.empty:
            raise ValueError(f"No 'Label' column found in {path}")
        split = _split_name(path)
        labels_by_split[split].extend(labels.tolist())
        file_rows.append({'file': path.name, 'split': split, 'rows': len(labels)})
        if split == 'training':
            frame = loader.load_single_file(path, sample_size=samples_per_file)
            if not frame.empty:
                sampled_training_frames.append(frame)

    split_order = [split for split in ('training', 'testing', 'other') if labels_by_split[split]]

    # 01: Raw benign/attack counts in the supplied file splits.
    benign_counts = [
        sum(label == 'BENIGN' for label in labels_by_split[split])
        for split in split_order
    ]
    attack_counts = [
        sum(label != 'BENIGN' for label in labels_by_split[split])
        for split in split_order
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(split_order))
    width = 0.36
    ax.bar(x - width / 2, benign_counts, width, label='Benign', color=COLORS['BENIGN'])
    ax.bar(x + width / 2, attack_counts, width, label='Attack', color=COLORS['ATTACK'])
    ax.set_xticks(x, [split.title() for split in split_order])
    ax.set_ylabel('Rows')
    ax.set_title('Raw Class Counts by Dataset Split')
    ax.legend()
    _save_figure(fig, output_dir / '01_class_counts_by_split.png')

    # 02: Attack labels by official split (top 20 keeps the figure readable).
    attack_counts_by_split = defaultdict(Counter)
    for split, labels in labels_by_split.items():
        for label in labels:
            if label != 'BENIGN':
                attack_counts_by_split[split][label] += 1
    attack_labels = sorted(
        {label for counts in attack_counts_by_split.values() for label in counts},
        key=lambda label: sum(counts[label] for counts in attack_counts_by_split.values()),
        reverse=True,
    )[:20]
    y = np.arange(len(attack_labels))
    fig, ax = plt.subplots(figsize=(11, max(6, 0.38 * len(attack_labels))))
    height = 0.36
    for offset, split in zip((-height / 2, height / 2), ('training', 'testing')):
        ax.barh(
            y + offset,
            [attack_counts_by_split[split][label] for label in attack_labels],
            height,
            label=split.title(),
            color='#756bb1' if split == 'training' else '#31a354',
        )
    ax.set_yticks(y, attack_labels)
    ax.invert_yaxis()
    ax.set_xlabel('Rows')
    ax.set_title('Attack Label Counts by Dataset Split')
    ax.legend()
    _save_figure(fig, output_dir / '02_attack_label_counts_by_split.png')

    # 03: Number of original rows in each source file.
    ordered_files = sorted(file_rows, key=lambda item: item['rows'])
    fig, ax = plt.subplots(figsize=(11, max(6, 0.34 * len(ordered_files))))
    ax.barh(
        [item['file'] for item in ordered_files],
        [item['rows'] for item in ordered_files],
        color=[('#756bb1' if item['split'] == 'training' else '#31a354') for item in ordered_files],
    )
    ax.set_xlabel('Rows')
    ax.set_title('Rows per Dataset File')
    _save_figure(fig, output_dir / '03_rows_per_dataset_file.png')

    if not sampled_training_frames:
        raise ValueError('No files identified as training files; expected *-training.csv/parquet names')
    sampled_training = pd.concat(sampled_training_frames, ignore_index=True)
    sampled_label_col = _label_column(sampled_training.columns)
    sampled_labels = _normalized_labels(sampled_training[sampled_label_col])
    balanced_training = loader._balance_classes(sampled_training)
    balanced_label_col = _label_column(balanced_training.columns)
    balanced_labels = _normalized_labels(balanced_training[balanced_label_col])

    def binary_counts(labels):
        return [int((labels == 'BENIGN').sum()), int((labels != 'BENIGN').sum())]

    before_counts = binary_counts(sampled_labels)
    after_counts = binary_counts(balanced_labels)
    # 04: Shows the actual class-balancing step applied to training files.
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(2)
    ax.bar(x - width / 2, before_counts, width, label='Before balancing', color='#9ecae1')
    ax.bar(x + width / 2, after_counts, width, label='Used for training', color='#3182bd')
    ax.set_xticks(x, ['Benign', 'Attack'])
    ax.set_ylabel('Rows')
    ax.set_title('Training Class Balance at the 50,000-Row/File Cap')
    ax.legend()
    _save_figure(fig, output_dir / '04_training_class_balance.png')

    # Convert selected training rows to the same ordered 64-feature schema used by the model.
    X_raw, y_raw_labels, feature_names = loader.preprocess(balanced_training)
    if len(X_raw) == 0:
        raise ValueError('Could not preprocess sampled training data for dataset charts')

    # 05: Correlation among the 20 most variable model features.
    variances = np.nanvar(X_raw, axis=0)
    varying_indices = np.flatnonzero(variances > 1e-12)
    if len(varying_indices) == 0:
        raise ValueError('All sampled features are constant; cannot make correlation chart')
    top_indices = varying_indices[
        np.argsort(variances[varying_indices])[-min(20, len(varying_indices)):]
    ]
    correlation = np.corrcoef(X_raw[:, top_indices], rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    top_names = [feature_names[index] for index in top_indices]
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(correlation, vmin=-1, vmax=1, cmap='coolwarm')
    ax.set_xticks(np.arange(len(top_names)), top_names, rotation=90, fontsize=8)
    ax.set_yticks(np.arange(len(top_names)), top_names, fontsize=8)
    ax.set_title(f'Feature Correlation Heatmap ({len(top_names)} Varying Features)')
    fig.colorbar(image, ax=ax, label='Correlation')
    _save_figure(fig, output_dir / '05_feature_correlation_heatmap.png')

    # 06: Log-scale distributions of six interpretable CIC flow features.
    candidate_features = [
        'Flow Packets/s', 'Flow Bytes/s', 'SYN Flag Count',
        'Total Fwd Packets', 'Packet Length Mean', 'Flow Duration',
        'Fwd Packet Length Mean', 'Bwd Packets/s', 'Min Packet Length',
        'ACK Flag Count', 'URG Flag Count',
    ]
    positions = {name: index for index, name in enumerate(feature_names)}
    class_values = np.asarray(y_raw_labels)
    benign_mask = class_values == 'BENIGN'
    attack_mask = ~benign_mask
    ranked_features = []
    for feature in candidate_features:
        if feature not in positions:
            continue
        values = np.log10(np.maximum(X_raw[:, positions[feature]], 0) + 1)
        benign_values = values[benign_mask]
        attack_values = values[attack_mask]
        combined_std = np.std(values)
        if combined_std <= 1e-9:
            continue
        effect_size = abs(np.mean(benign_values) - np.mean(attack_values)) / (
            np.std(benign_values) + np.std(attack_values) + 1e-9
        )
        ranked_features.append((effect_size, feature))
    ranked_features.sort(reverse=True)
    selected_features = [feature for _, feature in ranked_features[:6]]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    rng = np.random.default_rng(42)
    for ax, feature in zip(axes, selected_features):
        values = X_raw[:, positions[feature]]
        for label, display, color in (("BENIGN", "Benign", COLORS['BENIGN']),
                                      (None, "Attack", COLORS['ATTACK'])):
            mask = class_values == label if label is not None else class_values != 'BENIGN'
            observations = np.log10(np.maximum(values[mask], 0) + 1)
            if len(observations) > 4000:
                observations = rng.choice(observations, size=4000, replace=False)
            ax.hist(observations, bins=36, density=True, alpha=0.55, label=display, color=color)
        ax.set_title(feature)
        ax.set_xlabel('log10(value + 1)')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
    for ax in axes[len(selected_features):]:
        ax.set_visible(False)
    fig.suptitle('Selected Feature Distributions: Benign vs Attack')
    _save_figure(fig, output_dir / '06_feature_distributions_by_class.png')

    return {
        'file_count': len(files),
        'raw_rows': int(sum(item['rows'] for item in file_rows)),
        'file_rows': file_rows,
        'raw_split_class_counts': {
            split: {'benign': benign_counts[index], 'attack': attack_counts[index]}
            for index, split in enumerate(split_order)
        },
        'sampled_training_before_balance': {
            'benign': before_counts[0], 'attack': before_counts[1]
        },
        'sampled_training_after_balance': {
            'benign': after_counts[0], 'attack': after_counts[1]
        },
        'dataset_graphs': [
            '01_class_counts_by_split.png',
            '02_attack_label_counts_by_split.png',
            '03_rows_per_dataset_file.png',
            '04_training_class_balance.png',
            '05_feature_correlation_heatmap.png',
            '06_feature_distributions_by_class.png',
        ],
    }


def generate_model_charts(classifier, data: Dict, output_dir: Path) -> Dict:
    """Write training, validation, and official holdout evaluation charts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = classifier.model
    y_train = np.asarray(data['y_train'])
    y_val = np.asarray(data['y_val'])
    y_test = np.asarray(data['y_test'])
    val_prediction = model.predict(data['X_val'])
    test_prediction = model.predict(data['X_test'])
    classes = list(model.classes_)
    positive_index = classes.index(1)
    test_probabilities = model.predict_proba(data['X_test'])[:, positive_index]

    train_accuracy = getattr(classifier, 'training_accuracy', None)
    if train_accuracy is None:
        train_accuracy = accuracy_score(y_train, model.predict(data['X_train']))
    validation_accuracy = accuracy_score(y_val, val_prediction)
    test_accuracy = accuracy_score(y_test, test_prediction)
    fpr, tpr, _ = roc_curve(y_test, test_probabilities, pos_label=1)
    roc_auc = roc_auc_score(y_test, test_probabilities)
    precision, recall, _ = precision_recall_curve(y_test, test_probabilities, pos_label=1)
    average_precision = average_precision_score(y_test, test_probabilities)
    matrix = confusion_matrix(y_test, test_prediction, labels=[0, 1])

    # 07: Makes the train/validation/official-test gap visible.
    fig, ax = plt.subplots(figsize=(8, 5))
    names = ['Training (fit rows)', 'Validation', 'Official test']
    scores = [train_accuracy, validation_accuracy, test_accuracy]
    bars = ax.bar(names, scores, color=['#756bb1', '#31a354', '#e34a33'])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by Data Split')
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 0.015, f'{score:.2%}', ha='center')
    _save_figure(fig, output_dir / '07_accuracy_by_split.png')

    # 08: Test confusion matrix.
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap='Blues')
    ax.set_xticks([0, 1], ['Predicted benign', 'Predicted attack'], rotation=15)
    ax.set_yticks([0, 1], ['Actual benign', 'Actual attack'])
    ax.set_title('Confusion Matrix: Official Test Files')
    for row in range(2):
        for column in range(2):
            ax.text(column, row, f'{matrix[row, column]:,}', ha='center', va='center',
                    color='white' if matrix[row, column] > matrix.max() / 2 else 'black')
    fig.colorbar(image, ax=ax, label='Rows')
    _save_figure(fig, output_dir / '08_test_confusion_matrix.png')

    # 09: ROC curve on the official testing files.
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='#2b8cbe', lw=2, label=f'ROC AUC = {roc_auc:.4f}')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance')
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel='False positive rate', ylabel='True positive rate')
    ax.set_title('ROC Curve: Official Test Files')
    ax.legend(loc='lower right')
    _save_figure(fig, output_dir / '09_test_roc_curve.png')

    # 10: Precision-recall curve on the official testing files.
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color='#e34a33', lw=2,
            label=f'Average precision = {average_precision:.4f}')
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel='Recall', ylabel='Precision')
    ax.set_title('Precision–Recall Curve: Official Test Files')
    ax.legend(loc='lower left')
    _save_figure(fig, output_dir / '10_test_precision_recall_curve.png')

    # 11: Top 15 Random Forest feature importances.
    importances = np.asarray(model.feature_importances_)
    names = classifier.feature_names
    order = np.argsort(importances)[-15:]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh([names[index] for index in order], importances[order], color='#756bb1')
    ax.set_xlabel('Importance')
    ax.set_title('Top 15 Random Forest Feature Importances')
    _save_figure(fig, output_dir / '11_top_feature_importances.png')

    return {
        'training_accuracy': float(train_accuracy),
        'validation_accuracy': float(validation_accuracy),
        'test_accuracy': float(test_accuracy),
        'roc_auc': float(roc_auc),
        'average_precision': float(average_precision),
        'confusion_matrix': matrix.tolist(),
        'training_samples': int(len(y_train)),
        'validation_samples': int(len(y_val)),
        'test_samples': int(len(y_test)),
        'model_graphs': [
            '07_accuracy_by_split.png',
            '08_test_confusion_matrix.png',
            '09_test_roc_curve.png',
            '10_test_precision_recall_curve.png',
            '11_top_feature_importances.png',
        ],
    }


def write_training_summary(output_dir: Path, dataset_info: Dict, model_info: Dict, model_path: str) -> Path:
    """Write the run metrics and the distinction between each accuracy score."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / 'training_summary.md'
    matrix = model_info['confusion_matrix']
    graph_names = dataset_info.get('dataset_graphs', []) + model_info.get('model_graphs', [])
    dataset_counts = dataset_info.get('raw_split_class_counts', {})
    model_path_text = Path(model_path).as_posix()

    lines = [
        '# DDoS model training summary',
        '',
        f'- Saved model: `{model_path_text}`',
        f"- Dataset files: {dataset_info.get('file_count', 0)}",
        f"- Raw dataset rows: {dataset_info.get('raw_rows', 0):,}",
        f"- Training rows used by the fitted model: {model_info['training_samples']:,}",
        f"- Validation rows: {model_info['validation_samples']:,}",
        f"- Official test rows: {model_info['test_samples']:,}",
        '',
        '## Accuracy and holdout metrics',
        '',
        f"- Training accuracy (rows fitted by the model): **{model_info['training_accuracy']:.2%}**",
        f"- Validation accuracy (held out from the supplied training files): **{model_info['validation_accuracy']:.2%}**",
        f"- Official test accuracy (separate `*-testing` files): **{model_info['test_accuracy']:.2%}**",
        f"- ROC AUC on official test files: **{model_info['roc_auc']:.4f}**",
        f"- Average precision on official test files: **{model_info['average_precision']:.4f}**",
        '',
        '## Raw dataset class counts',
        '',
    ]
    for split, counts in dataset_counts.items():
        lines.append(
            f"- {split.title()}: {counts['benign']:,} benign; {counts['attack']:,} attack"
        )
    sampled_before = dataset_info.get('sampled_training_before_balance', {})
    sampled_after = dataset_info.get('sampled_training_after_balance', {})
    if sampled_before and sampled_after:
        lines.extend([
            f"- Training files after per-file cap, before balancing: {sampled_before['benign']:,} benign; {sampled_before['attack']:,} attack",
            f"- Training files after balancing: {sampled_after['benign']:,} benign; {sampled_after['attack']:,} attack",
        ])
    lines.extend([
        '',
        'Confusion matrix (rows: actual benign/attack; columns: predicted benign/attack):',
        '',
        '| | Predicted benign | Predicted attack |',
        '|---|---:|---:|',
        f'| Actual benign | {matrix[0][0]:,} | {matrix[0][1]:,} |',
        f'| Actual attack | {matrix[1][0]:,} | {matrix[1][1]:,} |',
        '',
        'Training accuracy measures the same rows used to fit the model. The official test score uses the separate dataset testing files and is the better estimate of performance on unseen CIC-DDoS2019 flows.',
        '',
        '## Graph files',
        '',
    ])
    lines.extend(f'- `{name}`' for name in graph_names)
    summary_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary_path
