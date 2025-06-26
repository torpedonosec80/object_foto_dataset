# model_training.py
import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as transforms
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np
from config import DEFAULT_SAVE_DIR, NUM_CLASSES, DEVICE, CLASS_NAME_TO_ID

class CustomVOCDataset(Dataset):
    def __init__(self, root_dir, image_set='train', transform=None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.annotation_dir = os.path.join(root_dir, 'annotations')
        self.transform = transform
        
        # Загрузка списка изображений
        split_file = os.path.join(root_dir, 'ImageSets', 'Main', f'{image_set}.txt')
        with open(split_file, 'r') as f:
            self.image_names = [line.strip() for line in f.readlines()]
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name + '.jpg')
        xml_path = os.path.join(self.annotation_dir, img_name + '.xml')
        
        # Загрузка изображения
        img = Image.open(img_path).convert('RGB')
        
        # Загрузка аннотаций
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in CLASS_NAME_TO_ID:
                continue
                
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(CLASS_NAME_TO_ID[class_name])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        
        if self.transform:
            img, target = self.transform(img, target)
            
        return img, target

def get_transform(train):
    transforms_list = [
        transforms.ToImageTensor(),
        transforms.ConvertDtype(torch.float32),
    ]
    
    if train:
        transforms_list.extend([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomZoomOut(fill=0),
            transforms.RandomIoUCrop(),
        ])
    
    transforms_list.append(transforms.Resize((320, 320), antialias=True))
    return transforms.Compose(transforms_list)

def collate_fn(batch):
    return tuple(zip(*batch))

def create_ssd_model(num_classes=NUM_CLASSES):
    weights = torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=weights)
    
    # Обновляем классификатор для нашего числа классов
    in_channels = model.head.classification_head.conv[0].in_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = torchvision.models.detection.ssd.SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes
    )
    
    return model.to(DEVICE)

def train_ssd_model(data_dir, epochs=30, batch_size=8):
    # Создаем датасеты
    train_dataset = CustomVOCDataset(
        root_dir=os.path.join(data_dir, 'train'),
        image_set='train',
        transform=get_transform(train=True))
    
    val_dataset = CustomVOCDataset(
        root_dir=os.path.join(data_dir, 'val'),
        image_set='val',
        transform=get_transform(train=False))
    
    # Создаем DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=2
    )
    
    # Инициализация модели
    model = create_ssd_model()
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=0.001, 
        momentum=0.9, 
        weight_decay=0.0005
    )
    
    # Цикл обучения
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for images, targets in train_loader:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            train_loss += losses.item()
        
        # Валидация
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(DEVICE) for img in images]
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets)
                val_loss += sum(loss for loss in loss_dict.values()).item()
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f}")
    
    return model

def save_ssd_model(model, save_dir=DEFAULT_SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'ssd_model.pth')
    torch.save(model.state_dict(), model_path)
    return model_path
