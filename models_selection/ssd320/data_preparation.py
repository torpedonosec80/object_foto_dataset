# data_preparation.py
import os
import shutil
import xml.etree.ElementTree as ET
import cv2
import yaml
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image
from pathlib import Path
from config import CLASS_NAME_TO_ID, CLASSES

def prepare_voc_structure(data_dir, output_dir):
    """
    Создает структуру папок VOC для обучения SSD
    """
    # Создаем директории
    voc_dir = os.path.join(output_dir, 'VOCdevkit', 'VOC2012')
    annotations_dir = os.path.join(voc_dir, 'Annotations')
    jpeg_dir = os.path.join(voc_dir, 'JPEGImages')
    imagesets_dir = os.path.join(voc_dir, 'ImageSets', 'Main')
    
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(jpeg_dir, exist_ok=True)
    os.makedirs(imagesets_dir, exist_ok=True)

    # Копируем изображения и аннотации
    image_files = [f for f in os.listdir(data_dir) 
                  if f.endswith(('.jpeg', '.jpg', '.JPG', '.png'))]
    
    for img_file in tqdm(image_files, desc="Preparing VOC dataset"):
        img_path = os.path.join(data_dir, img_file)
        xml_file = os.path.splitext(img_file)[0] + '.xml'
        xml_path = os.path.join(data_dir, xml_file)
        
        # Копируем изображение
        shutil.copy(img_path, jpeg_dir)
        
        # Копируем аннотацию (если существует)
        if os.path.exists(xml_path):
            shutil.copy(xml_path, annotations_dir)
    
    # Создаем списки для train/val/test
    all_files = [os.path.splitext(f)[0] for f in image_files]
    train_val, test = train_test_split(all_files, test_size=0.15, random_state=42)
    train, val = train_test_split(train_val, test_size=0.176, random_state=42)
    
    # Сохраняем списки
    def save_list(file_path, items):
        with open(file_path, 'w') as f:
            for item in items:
                f.write(f"{item}\n")
    
    save_list(os.path.join(imagesets_dir, 'train.txt'), train)
    save_list(os.path.join(imagesets_dir, 'val.txt'), val)
    save_list(os.path.join(imagesets_dir, 'test.txt'), test)
    
    return voc_dir

def load_annotations(data_dir):
    """
    Загружает XML аннотации в DataFrame
    """
    annotations = []
    
    for xml_file in Path(data_dir).glob("*.xml"):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            filename = root.find('filename').text
            img_path = Path(data_dir) / filename
            
            if not img_path.exists():
                continue
                
            with Image.open(img_path) as img:
                width, height = img.size
                
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                bbox = obj.find('bndbox')
                
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                
                annotations.append({
                    'image': filename,
                    'class': class_name,
                    'img_w': width,
                    'img_h': height,
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax
                })
                
        except Exception as e:
            print(f"Error processing {xml_file}: {str(e)}")
            
    df = pd.DataFrame(annotations)
    if not df.empty:
        df['bbox_w'] = df['xmax'] - df['xmin']
        df['bbox_h'] = df['ymax'] - df['ymin']
        df['bbox_area'] = df['bbox_w'] * df['bbox_h']
        
    return df
