# model_training.py
import os
import torch
import torchvision
from torchvision.transforms import v2 as transforms
from torch.utils.data import DataLoader
from config import DEFAULT_SAVE_DIR, NUM_CLASSES, SSD_WEIGHTS, DEVICE

def create_ssd_model(num_classes=NUM_CLASSES):
    """Создает модель SSD с предобученными весами"""
    weights = getattr(torchvision.models.detection, SSD_WEIGHTS)
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
        weights=weights
    )
    
    # Заменяем классификатор для нашего числа классов
    in_channels = model.head.classification_head.conv[0].in_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = torchvision.models.detection.ssd.SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes
    )
    
    return model.to(DEVICE)

def get_transform(train):
    """Трансформации для данных"""
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32),
        transforms.Resize((320, 320)),
        transforms.RandomHorizontalFlip(0.5) if train else lambda x: x,
    ])

def collate_fn(batch):
    """Обработка батча для детекции"""
    return tuple(zip(*batch))

def train_ssd_model(voc_dir, epochs=30, batch_size=8):
    """Обучение модели SSD"""
    # Загрузка данных
    train_dataset = torchvision.datasets.VOCDetection(
        root=os.path.dirname(voc_dir),
        year='2012',
        image_set='train',
        transforms=get_transform(train=True)
    )
    
    val_dataset = torchvision.datasets.VOCDetection(
        root=os.path.dirname(voc_dir),
        year='2012',
        image_set='val',
        transforms=get_transform(train=False)
    )
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, 
        shuffle=False, collate_fn=collate_fn
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
        for images, targets in train_loader:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        
        # Валидация
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(DEVICE) for img in images]
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets)
                val_loss += sum(loss for loss in loss_dict.values()).item()
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {losses.item():.4f} | Val Loss: {val_loss/len(val_loader):.4f}")
    
    return model

def save_ssd_model(model, save_dir=DEFAULT_SAVE_DIR):
    """Сохранение модели SSD"""
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'ssd_model.pth')
    torch.save(model.state_dict(), model_path)
    return model_path
