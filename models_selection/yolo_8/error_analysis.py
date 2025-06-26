# error_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
