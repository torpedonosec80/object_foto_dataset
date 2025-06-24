import os
import cv2
import numpy as np
import torch
import torchvision
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.model_selection import train_test_split
import gc
from copy import deepcopy
from datetime import datetime
from google.colab import drive
import shutil
import yaml
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import seaborn as sns
from glob import glob
from PIL import Image
from sklearn.metrics import confusion_matrix


# Функция конвертации одного файла
def convert_single_annotation(img_name, subset):
    # Пути к файлам
    img_path = os.path.join(data_dir, img_name)
    xml_path = os.path.join(data_dir, os.path.splitext(img_name)[0] + '.xml')
    txt_path = os.path.join(output_dir, subset, 'labels', os.path.splitext(img_name)[0] + '.txt')

    # Копируем изображение
    shutil.copy(img_path, os.path.join(output_dir, subset, 'images', img_name))

    # Получаем размеры изображения из самого файла изображения
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not read image {img_path}. Skipping.")
        return
    height, width = img.shape[:2]

    # Проверяем существование XML-файла
    if not os.path.exists(xml_path):
        print(f"Warning: XML annotation not found for {img_name}. Skipping.")
        return

    # Парсим XML
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        print(f"Error parsing XML for {img_name}. Skipping.")
        return

    # Обрабатываем объекты
    with open(txt_path, 'w') as f_txt:
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in CLASS_NAME_TO_ID:
                continue

            class_id = CLASS_NAME_TO_ID[class_name] - 1  # YOLO class IDs start from 0

            bndbox = obj.find('bndbox')
            try:
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
            except (AttributeError, ValueError):
                print(f"Error reading bbox for {img_name}. Skipping object.")
                continue

            # Конвертация в YOLO формат
            x_center = (xmin + xmax) / (2.0 * width)
            y_center = (ymin + ymax) / (2.0 * height)
            bbox_width = (xmax - xmin) / width
            bbox_height = (ymax - ymin) / height

            # Проверка валидности координат
            if (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                0 < bbox_width <= 1 and 0 < bbox_height <= 1):
                f_txt.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
            else:
                print(f"Invalid bbox coordinates in {img_name}: "
                      f"x_center={x_center}, y_center={y_center}, "
                      f"width={bbox_width}, height={bbox_height}")



def convert_annotations_to_yolo_format(data_dir, output_dir, subsets):
    """
    Конвертирует XML аннотации в формат YOLO и создает структуру директорий
    """
    # Создаем директории
    os.makedirs(output_dir, exist_ok=True)
    for subset in subsets:
        os.makedirs(os.path.join(output_dir, subset, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, subset, 'labels'), exist_ok=True)

    # Собираем все изображения
    all_images = [f for f in os.listdir(data_dir) if f.endswith(('.jpeg', '.jpg', '.JPG', '.png'))]

    # Создаем dataset.yaml
    yaml_data = {
        'path': output_dir,
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {idx-1: name for idx, name in CLASSES.items() if idx != 0}  # исключаем background
    }

    with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
        yaml.dump(yaml_data, f)

    # Функция конвертации одного файла
    convert_single_annotation(img_name, subset)

    # Распределение по сетам
    indices = list(range(len(all_images)))
    train_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.176, random_state=42)

    # Конвертация
    for i, img_name in enumerate(tqdm(all_images)):
        if i in train_idx:
            convert_single_annotation(img_name, 'train')
        elif i in val_idx:
            convert_single_annotation(img_name, 'val')
        elif i in test_idx:
            convert_single_annotation(img_name, 'test')
