import torch
import torchmetrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patches
from torchvision.ops import box_iou
import os

# Assuming CLASSES and DEVICE are imported from config
from .config import CLASSES, DEVICE, THRESHOLD

def calculate_map(model, data_loader, device):
    """
    Calculates Mean Average Precision (mAP) using torchmetrics.

    Args:
        model: The PyTorch model to evaluate.
        data_loader: DataLoader for the evaluation dataset.
        device: The device (CPU or GPU) to evaluate on.

    Returns:
        dict: Dictionary containing mAP metrics.
    """
    model.eval()
    # Use standard parameters for compatibility, potentially adjust max_detection_threshold if warnings persist
    metric = torchmetrics.detection.MeanAveragePrecision(
        iou_thresholds=[0.5, 0.75],
        class_metrics=True # Keep class_metrics for detailed analysis later if needed
    )

    with torch.no_grad():
        for batch in data_loader:
            if batch is None:
                continue

            images, targets, names = batch
            if not images or len(images) == 0:
                continue

            images = [img.to(device) for img in images]
            outputs = model(images)

            preds = []
            for output in outputs:
                preds.append({
                    'boxes': output['boxes'].cpu(),
                    'scores': output['scores'].cpu(),
                    'labels': output['labels'].cpu()
                })

            gt = []
            for target in targets:
                gt.append({
                    'boxes': target['boxes'].cpu(),
                    'labels': target['labels'].cpu()
                })

            # Ensure targets have the same keys as preds, even if empty
            # This is a potential source of errors if the metric expects certain keys
            # Let's check if the metric handles empty boxes/labels gracefully or if we need to add checks
            # Based on torchmetrics documentation, it should handle empty lists/tensors.
            # The previous error was likely not directly from this part.

            metric.update(preds, gt)

    # Compute metrics only if updates occurred
    try:
        computed_metrics = metric.compute()
        return computed_metrics
    except RuntimeError as e:
         print(f"Error computing metrics: {e}. This might happen if no detections or ground truths were processed.")
         # Return default values if computation fails (e.g., no data processed)
         return {
            'map': torch.tensor(0.0),
            'map_50': torch.tensor(0.0),
            'map_75': torch.tensor(0.0),
            # Add other expected keys with default values if needed
         }


def calculate_f1_per_class(model, data_loader, device, iou_threshold=0.5, conf_threshold=0.5):
    """
    Calculates F1-Score, Precision, and Recall per class.

    Args:
        model: The PyTorch model to evaluate.
        data_loader: DataLoader for the evaluation dataset.
        device: The device (CPU or GPU) to evaluate on.
        iou_threshold (float): IoU threshold for determining true positives.
        conf_threshold (float): Confidence threshold for filtering predictions.

    Returns:
        tuple: (dict of class metrics, dict of summary metrics)
    """
    model.eval()

    class_stats = {}
    for class_id in CLASSES.keys():
        if class_id == 0:
            continue
        class_stats[class_id] = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0
        }

    global_tp = 0
    global_fp = 0
    global_fn = 0

    with torch.no_grad():
        for images_list, targets_list, img_names in data_loader:
            if images_list is None or len(images_list) == 0:
                continue

            images = [img.to(device) for img in images_list]
            outputs = model(images)

            for i, (output, target) in enumerate(zip(outputs, targets_list)):
                keep = output['scores'] > conf_threshold
                pred_boxes = output['boxes'][keep]
                pred_labels = output['labels'][keep]
                # pred_scores = output['scores'][keep] # Not used in F1 calculation logic

                gt_boxes = target['boxes'].to(device)
                gt_labels = target['labels'].to(device)

                for class_id in class_stats.keys():
                    class_mask = pred_labels == class_id
                    pred_class_boxes = pred_boxes[class_mask]

                    gt_class_mask = gt_labels == class_id
                    gt_class_boxes = gt_boxes[gt_class_mask]

                    if len(pred_class_boxes) > 0:
                        if len(gt_class_boxes) > 0:
                            iou_matrix = box_iou(pred_class_boxes, gt_class_boxes)

                            matched_gt = set()
                            for pred_idx in range(len(pred_class_boxes)):
                                max_iou, gt_idx = iou_matrix[pred_idx].max(0)

                                if max_iou >= iou_threshold and gt_idx not in matched_gt:
                                    class_stats[class_id]['true_positives'] += 1
                                    global_tp += 1
                                    matched_gt.add(gt_idx)
                                else:
                                    class_stats[class_id]['false_positives'] += 1
                                    global_fp += 1

                            unmatched_gt = len(gt_class_boxes) - len(matched_gt)
                            class_stats[class_id]['false_negatives'] += unmatched_gt
                            global_fn += unmatched_gt
                        else:
                            class_stats[class_id]['false_positives'] += len(pred_class_boxes)
                            global_fp += len(pred_class_boxes)
                    else:
                        if len(gt_class_boxes) > 0:
                            class_stats[class_id]['false_negatives'] += len(gt_class_boxes)
                            global_fn += len(gt_class_boxes)

    class_metrics = {}
    macro_f1 = 0
    macro_precision = 0
    macro_recall = 0
    total_support = 0
    total_weighted_f1 = 0
    total_weighted_precision = 0
    total_weighted_recall = 0

    for class_id, stats in class_stats.items():
        tp = stats['true_positives']
        fp = stats['false_positives']
        fn = stats['false_negatives']

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        support = tp + fn

        class_metrics[class_id] = {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'support': support
        }

        macro_f1 += f1
        macro_precision += precision
        macro_recall += recall

        total_support += support
        total_weighted_f1 += f1 * support
        total_weighted_precision += precision * support
        total_weighted_recall += recall * support

    num_classes_with_support = len([s for s in class_stats.values() if s['true_positives'] + s['false_negatives'] > 0])
    macro_f1 /= num_classes_with_support if num_classes_with_support > 0 else 1
    macro_precision /= num_classes_with_support if num_classes_with_support > 0 else 1
    macro_recall /= num_classes_with_support if num_classes_with_support > 0 else 1


    weighted_f1 = total_weighted_f1 / (total_support + 1e-10)
    weighted_precision = total_weighted_precision / (total_support + 1e-10)
    weighted_recall = total_weighted_recall / (total_support + 1e-10)

    micro_precision = global_tp / (global_tp + global_fp + 1e-10)
    micro_recall = global_tp / (global_tp + global_fn + 1e-10)
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-10)

    summary_metrics = {
        'micro': {
            'f1': micro_f1,
            'precision': micro_precision,
            'recall': micro_recall,
            'support': global_tp + global_fn
        },
        'macro': {
            'f1': macro_f1,
            'precision': macro_precision,
            'recall': macro_recall
        },
        'weighted': {
            'f1': weighted_f1,
            'precision': weighted_precision,
            'recall': weighted_recall
        },
        'total_tp': global_tp,
        'total_fp': global_fp,
        'total_fn': global_fn
    }

    return class_metrics, summary_metrics

def visualize_metrics(class_metrics, summary_metrics, classes_dict, save_dir="results"):
    """
    Визуализация метрик F1-Score, Precision и Recall

    Параметры:
    class_metrics (dict): Метрики по классам из calculate_f1_per_class
    summary_metrics (dict): Сводные метрики из calculate_f1_per_class
    classes_dict (dict): Словарь соответствия ID классов и их имен
    save_dir (str): Директория для сохранения графиков
    """
    # Создаем директорию для сохранения, если её нет
    os.makedirs(save_dir, exist_ok=True)

    # Создаем DataFrame для отчета по классам
    class_data = []
    # Handle "unknown" class if it exists and has 0 support
    filtered_class_metrics = {k: v for k, v in class_metrics.items() if v['support'] > 0 or classes_dict[k] != 'unknown'}

    for class_id, metrics in filtered_class_metrics.items():
        class_name = classes_dict[class_id]
        class_data.append({
            'Class': class_name,
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1'],
            'Support': metrics['support']
        })


    df_class_report = pd.DataFrame(class_data)

    # Ensure the DataFrame is not empty before plotting
    if df_class_report.empty:
        print("No classes with support > 0 found for visualization.")
        return {}


    # График F1-Score по классам
    plt.figure(figsize=(12, 6))
    plt.bar(df_class_report['Class'], df_class_report['F1-Score'], color='skyblue')
    plt.axhline(y=summary_metrics['macro']['f1'], color='r', linestyle='--',
                label=f'Macro F1-Score: {summary_metrics["macro"]["f1"]:.4f}')
    plt.title('F1-Score по классам')
    plt.xlabel('Класс')
    plt.ylabel('F1-Score')
    plt.ylim(0, 1)
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    f1_by_class_path = os.path.join(save_dir, 'f1_by_class.png')
    plt.savefig(f1_by_class_path)
    plt.show()
    plt.close()

    # Тепловая карта метрик
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_class_report[['Precision', 'Recall', 'F1-Score']],
                annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=['Precision', 'Recall', 'F1-Score'],
                yticklabels=df_class_report['Class'])
    plt.title('Метрики по классам')
    plt.tight_layout()
    heatmap_path = os.path.join(save_dir, 'metrics_heatmap.png')
    plt.savefig(heatmap_path)
    plt.show()
    plt.close()

    # График Precision-Recall по классам
    plt.figure(figsize=(10, 8))

    # Определяем динамические границы
    min_precision = df_class_report['Precision'].min()
    min_recall = df_class_report['Recall'].min()
    max_precision = df_class_report['Precision'].max()
    max_recall = df_class_report['Recall'].max()

    # Рассчитываем буфер
    buffer = 0.05
    x_min = max(0.0, min_recall - buffer)
    x_max = min(1.0, max_recall + buffer)
    y_min = max(0.0, min_precision - buffer)
    y_max = min(1.0, max_precision + buffer)

    # Ensure at least a small range to avoid errors
    x_min = min(x_min, x_max - 0.01)
    y_min = min(y_min, y_max - 0.01)


    # Создаем scatter plot
    for _, row in df_class_report.iterrows():
        plt.scatter(row['Recall'], row['Precision'], s=120, edgecolors='black')
        plt.text(
            row['Recall'],
            row['Precision'] + 0.005,
            f"{row['Class']}\nF1={row['F1-Score']:.4f}",
            fontsize=9,
            ha='center',
            va='bottom',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2')
        )

    # Настройка осей и сетки
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall по классам')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Добавляем диагональ
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.2) # Draw from (0,0) to (1,1)

    # Добавляем изолинии F1-Score
    for f1 in [0.5, 0.7, 0.9]:
        # Generate points for the isoline within the plot limits
        x_iso = np.linspace(x_min, x_max, 100)
        with np.errstate(divide='ignore', invalid='ignore'):
            y_iso = np.where(
                (2 * x_iso - f1) != 0,
                f1 * x_iso / (2 * x_iso - f1),
                np.nan
            )

        # Filter out points outside the plot limits and invalid values
        valid_points = (~np.isnan(y_iso)) & (y_iso >= y_min) & (y_iso <= y_max)

        # Find segments of valid points
        if np.any(valid_points):
            valid_indices = np.where(valid_points)[0]
            # Find contiguous segments by looking for breaks in consecutive indices
            breaks = np.where(np.diff(valid_indices) != 1)[0]
            segments = np.split(valid_indices, breaks + 1)

            for segment_indices in segments:
                if len(segment_indices) > 1: # Need at least 2 points to draw a line
                    plt.plot(x_iso[segment_indices], y_iso[segment_indices], 'r--', alpha=0.3, linewidth=0.8)

            # Add label to the last segment's end point if available
            if segments and len(segments[-1]) > 0:
                last_idx_in_segment = segments[-1][-1]
                plt.text(x_iso[last_idx_in_segment], y_iso[last_idx_in_segment], f"F1={f1}",
                         color='red', fontsize=8, ha='right', va='bottom')


    plt.tight_layout()
    pr_path = os.path.join(save_dir, 'precision_recall.png')
    plt.savefig(pr_path)
    plt.show()
    plt.close()

    # Дополнительный график: сравнение микро, макро и взвешенных метрик
    metrics_types = ['micro', 'macro', 'weighted']
    metric_names = ['f1', 'precision', 'recall']

    plt.figure(figsize=(12, 6))
    bar_width = 0.25
    index = np.arange(len(metrics_types))

    for i, metric in enumerate(metric_names):
        values = [summary_metrics[m_type][metric] for m_type in metrics_types]
        plt.bar(index + i * bar_width, values, bar_width, label=metric.capitalize())

    plt.xlabel('Тип усреднения')
    plt.ylabel('Значение метрики')
    plt.title('Сравнение микро, макро и взвешенных метрик')
    plt.xticks(index + bar_width/2, [t.capitalize() for t in metrics_types])
    plt.ylim(0, 1.05) # Adjusted y-limit
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    summary_path = os.path.join(save_dir, 'summary_metrics.png')
    plt.savefig(summary_path)
    plt.show()
    plt.close()

    # Сохраняем данные в CSV
    csv_path = os.path.join(save_dir, 'metrics_report.csv')
    df_class_report.to_csv(csv_path, index=False)

    # Добавляем сводные метрики в отдельный CSV
    summary_data = []
    for m_type in metrics_types:
        for metric in metric_names:
            summary_data.append({
                'Type': m_type,
                'Metric': metric,
                'Value': summary_metrics[m_type][metric]
            })

    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = os.path.join(save_dir, 'summary_metrics.csv')
    summary_df.to_csv(summary_csv_path, index=False)

    return {
        'f1_by_class': f1_by_class_path,
        'metrics_heatmap': heatmap_path,
        'precision_recall': pr_path,
        'summary_metrics': summary_path,
        'class_report_csv': csv_path,
        'summary_report_csv': summary_csv_path
    }
