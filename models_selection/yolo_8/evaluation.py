# evaluation.py
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from pathlib import Path
from ultralytics import YOLO
from config import CLASSES
import pandas as pd
from tqdm import tqdm

def evaluate_model(model, data_yaml, split='test'):
    results = model.val(
        data=data_yaml,
        split=split,
        name='yolo_test_evaluation'
    )
    
    print(f'\n=== YOLO Test Results ===')
    print(f'mAP50-95: {results.box.map:.4f}')
    print(f'mAP50: {results.box.map50:.4f}')
    print(f'mAP75: {results.box.map75:.4f}')
    
    return results

def visualize_predictions(model, dataset_dir, num_samples=5, conf=0.5, save_path='yolo_predictions.png'):
    test_images_dir = os.path.join(dataset_dir, 'test', 'images')
    image_files = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))][:num_samples]
    
    if num_samples == 1:
        fig, axs = plt.subplots(1, 1, figsize=(12, 6))
        axs = [axs]
    else:
        fig, axs = plt.subplots(num_samples, 1, figsize=(12, 6*num_samples))
    
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(test_images_dir, img_file)
        results = model.predict(source=img_path, conf=conf, save=False, imgsz=640)
        result = results[0]
        
        img = cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB)
        axs[i].imshow(img)
        axs[i].axis('off')
        axs[i].set_title(f'Image: {img_file}')
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            
            for box, cls_id, conf_val in zip(boxes, clss, confs):
                x1, y1, x2, y2 = box[:4]
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=1.5, edgecolor='r', facecolor='none'
                )
                axs[i].add_patch(rect)
                
                class_name = model.names[int(cls_id)]
                label = f"{class_name} {conf_val:.2f}"
                axs[i].text(
                    x1+2, y1+10, label,
                    color='white', fontsize=10,
                    bbox=dict(facecolor='red', alpha=0.8)
                )
                
            if result.masks is not None:
                masks = result.masks.xy
                for mask in masks:
                    if len(mask) > 0:
                        poly = patches.Polygon(
                            mask, closed=True,
                            edgecolor='lime', facecolor='none', linewidth=1.5
                        )
                        axs[i].add_patch(poly)
        else:
            axs[i].text(0.5, 0.5, 'No detections',
                        fontsize=12, color='red',
                        ha='center', va='center',
                        transform=axs[i].transAxes)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    return save_path

def run_detection_on_dataset(model, source_dir, conf_threshold=0.25):
    """
    Выполняет детекцию на всем датасете и возвращает результаты в DataFrame
    """
    image_extensions = ('.jpg', '.jpeg', '.png', '.JPG')
    image_files = [f for f in os.listdir(source_dir) 
                 if f.lower().endswith(image_extensions)]
    
    results_list = []
    
    for img_file in tqdm(image_files, desc="Running detection"):
        img_path = os.path.join(source_dir, img_file)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
            
        height, width = img.shape[:2]
        
        # Выполняем предсказание
        predictions = model.predict(
            source=img_path,
            conf=conf_threshold,
            save=False,
            imgsz=640
        )
        
        result = predictions[0]
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            
            for box, cls_id, conf_val in zip(boxes, clss, confs):
                class_name = model.names[int(cls_id)]
                x1, y1, x2, y2 = box
                
                results_list.append({
                    'image_file': img_file,
                    'class_id': int(cls_id),
                    'class_name': class_name,
                    'confidence': conf_val,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'width': x2 - x1,
                    'height': y2 - y1,
                    'rel_x1': x1 / width,
                    'rel_y1': y1 / height,
                    'rel_x2': x2 / width,
                    'rel_y2': y2 / height,
                    'rel_width': (x2 - x1) / width,
                    'rel_height': (y2 - y1) / height
                })
        else:
            # Запись для изображений без детекций
            results_list.append({
                'image_file': img_file,
                'class_id': None,
                'class_name': None,
                'confidence': None,
                'x1': None, 'y1': None, 'x2': None, 'y2': None,
                'width': None, 'height': None,
                'rel_x1': None, 'rel_y1': None,
                'rel_x2': None, 'rel_y2': None,
                'rel_width': None, 'rel_height': None
            })
    
    return pd.DataFrame(results_list)
