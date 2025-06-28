import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import pandas as pd
import seaborn as sns

# Assuming CLASSES and DEVICE are imported from config
from .config import CLASSES, DEVICE, THRESHOLD, transform
from .datasets import CustomDataset, InferenceDataset, collate_fn

def visualize_predictions(model, dataset, num_samples=5, threshold=0.3, save_dir="prediction_visualizations"):
    """
    Visualizes model predictions on a few sample images from the dataset.

    Args:
        model: The trained PyTorch model.
        dataset: The dataset to sample images from (e.g., validation or test dataset).
        num_samples (int): The number of images to visualize.
        threshold (float): Confidence threshold for displaying predictions.
        save_dir (str): Directory to save the visualization images.
    """
    model.eval()
    # Ensure num_samples does not exceed dataset size
    num_samples = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    fig, axs = plt.subplots(num_samples, 1, figsize=(12, 6 * num_samples))

    # Handle case with only one subplot
    if num_samples == 1:
        axs = [axs]

    os.makedirs(save_dir, exist_ok=True)

    for i, idx in enumerate(indices):
        # Get image, target, and image name from the dataset
        # Note: CustomDataset returns image, target, img_name.
        # InferenceDataset returns image, img_name.
        # We need to handle both possibilities or ensure the dataset provides targets for visualization.
        # Assuming we are visualizing from a dataset *with* targets (like val_dataset or test_dataset)
        try:
             image, target, img_name = dataset[idx]
        except ValueError:
             print(f"Warning: Dataset {type(dataset).__name__} does not provide targets for visualization. Skipping ground truth boxes.")
             image, img_name = dataset[idx]
             target = None # No ground truth available

        ax = axs[i]

        with torch.no_grad():
            # Add batch dimension for inference
            prediction = model([image.to(DEVICE)])[0]

        # Denormalize and permute image for plotting
        img = image.cpu().permute(1, 2, 0).numpy()
        # Assuming standard ImageNet normalization values used in config
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1) # Clip values to [0, 1]

        ax.imshow(img)
        ax.set_title(f"Image: {img_name}", fontsize=12, pad=10)
        ax.axis('off') # Hide axes ticks

        # Plot predictions (Red boxes)
        for score, box, label in zip(prediction['scores'], prediction['boxes'], prediction['labels']):
            if score > threshold:
                box = box.cpu().numpy()
                rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                        linewidth=1.5, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                label_name = CLASSES.get(label.item(), f'unknown_{label.item()}') # Handle potential unknown labels
                ax.text(box[0] + 2, box[1] + 10,
                        f'{label_name} {score:.4f}',
                        color='white', fontsize=10,
                        bbox=dict(facecolor='red', alpha=0.8))

        # Plot ground truth (Green dotted boxes)
        if target is not None:
            for box, label in zip(target['boxes'], target['labels']):
                box = box.cpu().numpy()
                rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                       linewidth=1, edgecolor='g', facecolor='none', linestyle=':')
                ax.add_patch(rect)
                label_name = CLASSES.get(label.item(), f'unknown_{label.item()}') # Handle potential unknown labels
                ax.text(box[2] - 2, box[1] + 10,
                       f'TRUE: {label_name}',
                       color='green', fontsize=8,
                       bbox=dict(facecolor='white', alpha=0.7),
                       horizontalalignment='right')

        # Save individual image visualization
        img_save_path = os.path.join(save_dir, f'prediction_{img_name}')
        # Remove extension from image name for saving
        img_save_path_base, _ = os.path.splitext(img_save_path)
        plt.savefig(f'{img_save_path_base}.png', bbox_inches='tight')


    plt.tight_layout()
    # Optionally save the combined figure
    # plt.savefig(os.path.join(save_dir, 'all_predictions_sample.png'))
    plt.show()
    plt.close(fig) # Close the figure to free memory

def analyze_detection_results(df_results, inference_dataset, save_dir="analysis_results"):
    """
    Analyzes detection results from a DataFrame and generates visualizations and reports.

    Args:
        df_results (pd.DataFrame): DataFrame containing detection results (from run_inference).
        classes_dict (dict): Dictionary mapping class IDs to names.
        save_dir (str): Directory to save analysis results.
    """
    os.makedirs(save_dir, exist_ok=True)

    print("\n--- Analyzing Detection Results ---")

    # 1. Количество обнаружений по классам
    class_counts = df_results['class_name'].value_counts()
    plt.figure(figsize=(12, 6))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
    plt.title('Количество обнаружений по классам')
    plt.xlabel('Класс объекта')
    plt.ylabel('Количество обнаружений')
    plt.xticks(rotation=45)
    plt.tight_layout()
    class_counts_path = os.path.join(save_dir, 'class_counts.png')
    plt.savefig(class_counts_path)
    plt.show()
    plt.close()
    print("\n1. Количество обнаружений по классам:")
    print(class_counts)

    # 2. Средняя уверенность по классам
    confidence_by_class = df_results.groupby('class_name')['confidence'].mean().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=confidence_by_class.index, y=confidence_by_class.values, palette='coolwarm')
    plt.title('Средняя уверенность по классам')
    plt.xlabel('Класс объекта')
    plt.ylabel('Средняя уверенность')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    confidence_path = os.path.join(save_dir, 'confidence_by_class.png')
    plt.savefig(confidence_path)
    plt.show()
    plt.close()
    print("\n2. Средняя уверенность по классам:")
    print(confidence_by_class)

    # 3. Распределение размеров объектов
    df_results['width'] = df_results['xmax'] - df_results['xmin']
    df_results['height'] = df_results['ymax'] - df_results['ymin']

    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    sns.histplot(df_results['width'], bins=30, kde=True, color='skyblue')
    plt.title('Распределение ширины объектов')
    plt.xlabel('Ширина (пиксели)')
    plt.ylabel('Количество')

    plt.subplot(2, 2, 2)
    sns.histplot(df_results['height'], bins=30, kde=True, color='salmon')
    plt.title('Распределение высоты объектов')
    plt.xlabel('Высота (пиксели)')
    plt.ylabel('Количество')

    plt.subplot(2, 2, 3)
    sns.scatterplot(data=df_results, x='width', y='height', alpha=0.6)
    plt.title('Соотношение ширина-высота объектов')
    plt.xlabel('Ширина (пиксели)')
    plt.ylabel('Высота (пиксели)')

    plt.subplot(2, 2, 4)
    sns.boxplot(data=df_results, x='class_name', y='width', palette='Set3')
    plt.title('Ширина объектов по классам')
    plt.xlabel('Класс объекта')
    plt.ylabel('Ширина (пиксели)')
    plt.xticks(rotation=45)

    plt.tight_layout()
    size_dist_path = os.path.join(save_dir, 'object_size_distribution.png')
    plt.savefig(size_dist_path)
    plt.show()
    plt.close()

    print("\n3. Статистика размеров объектов:")
    print(df_results[['width', 'height']].describe())

    # 4. Изображения без детекций
    
    
    # Получаем список всех обработанных изображений
    all_images = set(inference_dataset.images)

    # Получаем изображения с детекциями
    detected_images = set(df_results['image_name'].unique())

    # Находим изображения без детекций
    no_detection_images = list(all_images - detected_images)

    # Создаем отчет
    no_detection_df = pd.DataFrame({
        'image_name': no_detection_images,
        'detection_status': 'No objects detected'
    })

    # Сохраняем в CSV
    no_detection_path = os.path.join(save_dir, 'images_without_detections.csv')
    no_detection_df.to_csv(no_detection_path, index=False)

    print(f"\n4. Изображения без детекций: {len(no_detection_images)}")
    print(f"Список сохранен в: {no_detection_path}")


    # Save full analysis report
    analysis_report = f"""
    Анализ результатов детекции:
    ------------------------------------
    1. Общее количество обнаруженных объектов: {len(df_results)}
    2. Количество обработанных изображений: {len(all_images)}
    3. Изображений с детекциями: {len(df_results['image_name'].unique())}
    4. Изображений без детекций: {len(no_detection_images)}
    5. Средняя уверенность по всем объектам: {df_results['confidence'].mean():.4f}
    6. Минимальная уверенность: {df_results['confidence'].min():.4f}
    7. Максимальная уверенность: {df_results['confidence'].max():.4f}

    Распределение по классам:
    {class_counts.to_string()}

    Средняя уверенность по классам:
    {confidence_by_class.to_string()}
    """

    report_path = os.path.join(save_dir, 'analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write(analysis_report)

    print(analysis_report)
    print(f"Анализ завершен! Все результаты сохранены в {save_dir}")

    return {
        'class_counts_plot': class_counts_path,
        'confidence_by_class_plot': confidence_path,
        'size_distribution_plot': size_dist_path,
        'analysis_report_txt': report_path,
        'images_without_detections_csv': no_detection_path
    }
