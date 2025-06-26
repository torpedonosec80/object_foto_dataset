# metrics_analysis.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
import os
from pathlib import Path

def plot_training_metrics(experiment_name="yolo_object_detection"):
    experiment_dir = Path("runs/detect") / experiment_name
    if not experiment_dir.exists():
        all_experiments = sorted(Path("runs/detect").glob(f"{experiment_name}*"))
        if all_experiments:
            experiment_dir = all_experiments[-1]
    
    results_file = experiment_dir / "results.csv"
    
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return None
    
    df = pd.read_csv(results_file)
    
    plt.figure(figsize=(15, 12))
    sns.set_style("whitegrid")
    
    # Losses
    plt.subplot(3, 1, 1)
    plt.plot(df['epoch'], df['train/box_loss'], label='Box Loss')
    plt.plot(df['epoch'], df['train/cls_loss'], label='Class Loss')
    plt.plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss')
    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Precision & Recall
    plt.subplot(3, 1, 2)
    plt.plot(df['epoch'], df['metrics/precision(B)'], label='Precision', marker='o')
    plt.plot(df['epoch'], df['metrics/recall(B)'], label='Recall', marker='o')
    plt.title('Precision & Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    
    # mAP Metrics
    plt.subplot(3, 1, 3)
    plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50', marker='s')
    plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95', marker='s')
    plt.title('mAP Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.legend()
    
    plt.tight_layout()
    save_path = 'training_metrics.png'
    plt.savefig(save_path)
    plt.show()
    return save_path

def analyze_detections(detections_df):
    # Анализ распределения детекций
    class_counts = detections_df['class_name'].value_counts()
    confidence_by_class = detections_df.groupby('class_name')['confidence'].mean()
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    class_counts.plot(kind='bar')
    plt.title('Detections per Class')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    confidence_by_class.sort_values().plot(kind='barh')
    plt.title('Average Confidence per Class')
    plt.xlabel('Confidence')
    
    plt.tight_layout()
    save_path = 'detection_analysis.png'
    plt.savefig(save_path)
    plt.show()
    
    # Возвращаем статистику
    return {
        'class_counts': class_counts.to_dict(),
        'confidence_by_class': confidence_by_class.to_dict()
    }
