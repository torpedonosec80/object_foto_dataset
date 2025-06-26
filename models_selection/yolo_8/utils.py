# utils.py
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import shutil
from datetime import datetime
from google.colab import drive

def save_results_to_drive(results_dir, drive_path="/content/drive/MyDrive"):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_folder = f"{drive_path}/object_detection_results/yolo_{current_time}"
    os.makedirs(save_folder, exist_ok=True)
    
    # Копируем важные файлы
    if os.path.exists('/content/runs'):
        shutil.copytree('/content/runs', os.path.join(save_folder, "runs"))
    if os.path.exists('saved_models'):
        shutil.copytree('saved_models', os.path.join(save_folder, "saved_models"))
    
    # Копируем визуализации
    for file in ['yolo_predictions.png', 'training_metrics.png', 
                 'detection_analysis.png', 'class_performance.png']:
        if os.path.exists(file):
            shutil.copy(file, save_folder)
    
    print(f"Results saved to: {save_folder}")
    return save_folder

def view_images_from_folder(folder_path, num_samples=5, max_cols=3, 
                           image_size=(10, 8), title_prefix=""):
    image_files = [f for f in os.listdir(folder_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No images found in {folder_path}")
        return
        
    num_samples = min(num_samples, len(image_files))
    sample_files = image_files[:num_samples]
    
    num_cols = min(max_cols, num_samples)
    num_rows = (num_samples + num_cols - 1) // num_cols
    
    fig_width = num_cols * image_size[0]
    fig_height = num_rows * image_size[1]
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
    
    if num_samples == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, img_file in enumerate(sample_files):
        ax = axes[i]
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Could not read image: {img_path}")
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(f"{title_prefix}{img_file}", fontsize=12)
        ax.axis('off')
    
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout(pad=3.0)
    plt.show()
    print(f"Showing {num_samples}/{len(image_files)} images from {folder_path}")
