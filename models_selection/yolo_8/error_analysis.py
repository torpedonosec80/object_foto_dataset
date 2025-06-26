# error_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
from pathlib import Path
from config import CLASS_NAME_TO_ID

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def compare_detections_with_gt(detections_df, annotations_df):
    comparison_results = []
    
    # Переименовываем колонки для удобства
    annotations_df = annotations_df.rename(columns={
        'image': 'image_file',
        'class': 'true_class',
        'xmin': 'true_xmin',
        'ymin': 'true_ymin',
        'xmax': 'true_xmax',
        'ymax': 'true_ymax'
    })
    
    detections_df = detections_df.rename(columns={
        'class_name': 'pred_class',
        'x1': 'pred_xmin',
        'y1': 'pred_ymin',
        'x2': 'pred_xmax',
        'y2': 'pred_ymax'
    })
    
    # Группируем по изображениям
    for img_file, img_annots in annotations_df.groupby('image_file'):
        img_detects = detections_df[detections_df['image_file'] == img_file]
        
        for _, annot_row in img_annots.iterrows():
            true_box = [
                annot_row['true_xmin'], annot_row['true_ymin'],
                annot_row['true_xmax'], annot_row['true_ymax']
            ]
            
            best_iou = 0
            best_match = None
            
            for _, detect_row in img_detects.iterrows():
                pred_box = [
                    detect_row['pred_xmin'], detect_row['pred_ymin'],
                    detect_row['pred_xmax'], detect_row['pred_ymax']
                ]
                
                iou = calculate_iou(true_box, pred_box)
                if iou > best_iou:
                    best_iou = iou
                    best_match = detect_row
            
            result = {
                'image_file': img_file,
                'true_class': annot_row['true_class'],
                'true_xmin': annot_row['true_xmin'],
                'true_ymin': annot_row['true_ymin'],
                'true_xmax': annot_row['true_xmax'],
                'true_ymax': annot_row['true_ymax'],
                'iou': best_iou,
                'matched': best_iou > 0.5
            }
            
            if best_match is not None:
                result.update({
                    'pred_class': best_match['pred_class'],
                    'confidence': best_match['confidence'],
                    'pred_xmin': best_match['pred_xmin'],
                    'pred_ymin': best_match['pred_ymin'],
                    'pred_xmax': best_match['pred_xmax'],
                    'pred_ymax': best_match['pred_ymax'],
                    'class_match': best_match['pred_class'] == annot_row['true_class']
                })
            else:
                result.update({
                    'pred_class': None,
                    'confidence': None,
                    'pred_xmin': None,
                    'pred_ymin': None,
                    'pred_xmax': None,
                    'pred_ymax': None,
                    'class_match': False
                })
            
            comparison_results.append(result)
    
    return pd.DataFrame(comparison_results)

def visualize_errors(comparison_df, source_dir):
    os.makedirs("false_positives", exist_ok=True)
    os.makedirs("false_negatives", exist_ok=True)
    
    # False Positives
    fp_images = {}
    for _, row in comparison_df.iterrows():
        if row['matched'] is False and row['pred_class'] is not None:
            img_file = row['image_file']
            if img_file not in fp_images:
                fp_images[img_file] = []
            fp_images[img_file].append({
                'box': [row['pred_xmin'], row['pred_ymin'], row['pred_xmax'], row['pred_ymax']],
                'class': row['pred_class'],
                'conf': row['confidence']
            })
    
    # False Negatives
    fn_images = {}
    for _, row in comparison_df.iterrows():
        if not row['matched']:
            img_file = row['image_file']
            if img_file not in fn_images:
                fn_images[img_file] = []
            fn_images[img_file].append({
                'box': [row['true_xmin'], row['true_ymin'], row['true_xmax'], row['true_ymax']],
                'class': row['true_class']
            })
    
    # Сохранение FP
    for img_file, detections in fp_images.items():
        img_path = os.path.join(source_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        
        for det in detections:
            x1, y1, x2, y2 = det['box']
            plt.gca().add_patch(plt.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor='r', facecolor='none'
            ))
            plt.text(x1, y1-10, f"FP: {det['class']} {det['conf']:.2f}", 
                     color='red', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        
        save_path = f"false_positives/{Path(img_file).stem}.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    
    # Сохранение FN
    for img_file, annotations in fn_images.items():
        img_path = os.path.join(source_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        
        for ann in annotations:
            x1, y1, x2, y2 = ann['box']
            plt.gca().add_patch(plt.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor='b', facecolor='none', linestyle='--'
            ))
            plt.text(x1, y1-10, f"FN: {ann['class']}", 
                     color='blue', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        
        save_path = f"false_negatives/{Path(img_file).stem}.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    
    return len(fp_images), len(fn_images)

def calculate_f1_score(comparison_df):
    """
    Расчет F1-Score для каждого класса и общих метрик
    
    Параметры:
    comparison_df (DataFrame): Результаты сравнения предсказаний с истиной
    
    Возвращает:
    df_class_report (DataFrame): Отчет по классам
    overall_metrics (dict): Общие метрики
    """
    # Уникальные классы
    classes = comparison_df['true_class'].unique().tolist()
    classes = [c for c in classes if c is not None]  # Удаляем None
    
    # Сбор статистики по классам
    class_report = []
    
    for cls in classes:
        # Фильтрация по классу
        cls_data = comparison_df[comparison_df['true_class'] == cls]
        
        # Расчет метрик
        TP = cls_data['matched'].sum()  # True Positives
        FN = len(cls_data) - TP         # False Negatives
        
        # False Positives для класса (предсказания этого класса, но с ошибкой)
        FP = len(comparison_df[
            (comparison_df['pred_class'] == cls) & 
            (comparison_df['true_class'] != cls)
        ])
        
        # Рассчет Precision, Recall, F1
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        class_report.append({
            'Class': cls,
            'TP': TP,
            'FP': FP,
            'FN': FN,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
    
    # Создание DataFrame с результатами по классам
    df_class_report = pd.DataFrame(class_report)
    
    # Общие метрики (микро-усреднение)
    total_TP = df_class_report['TP'].sum()
    total_FP = df_class_report['FP'].sum()
    total_FN = df_class_report['FN'].sum()
    
    micro_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    micro_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
    # Макро-усреднение F1
    macro_f1 = df_class_report['F1-Score'].mean()
    
    # Общие метрики
    overall_metrics = {
        'Micro Precision': micro_precision,
        'Micro Recall': micro_recall,
        'Micro F1-Score': micro_f1,
        'Macro F1-Score': macro_f1
    }
    
    return df_class_report, overall_metrics

def visualize_f1_metrics(df_class_report, overall_metrics, save_dir="."):
    """
    Визуализация метрик F1-Score
    
    Параметры:
    df_class_report (DataFrame): Отчет по классам
    overall_metrics (dict): Общие метрики
    save_dir (str): Директория для сохранения графиков
    """
    # График F1-Score по классам
    plt.figure(figsize=(12, 6))
    plt.bar(df_class_report['Class'], df_class_report['F1-Score'], color='skyblue')
    plt.axhline(y=overall_metrics['Macro F1-Score'], color='r', linestyle='--', label='Macro F1-Score')
    plt.title('F1-Score по классам')
    plt.xlabel('Класс')
    plt.ylabel('F1-Score')
    plt.ylim(0, 1)
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'f1_by_class.png'))
    plt.show()
    
    # Тепловая карта метрик
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_class_report[['Precision', 'Recall', 'F1-Score']], 
                annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=['Precision', 'Recall', 'F1-Score'],
                yticklabels=df_class_report['Class'])
    plt.title('Метрики по классам')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_heatmap.png'))
    plt.show()
    
    # Дополнительный график: Precision-Recall по классам
    plt.figure(figsize=(10, 8))
    for i, row in df_class_report.iterrows():
        plt.scatter(row['Recall'], row['Precision'], s=100, label=row['Class'])
        plt.text(row['Recall'], row['Precision']+0.01, row['Class'], fontsize=9)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall по классам')
    plt.grid(True)
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.savefig(os.path.join(save_dir, 'precision_recall.png'))
    plt.show()
    
    return {
        'f1_by_class': os.path.join(save_dir, 'f1_by_class.png'),
        'metrics_heatmap': os.path.join(save_dir, 'metrics_heatmap.png'),
        'precision_recall': os.path.join(save_dir, 'precision_recall.png')
    }
