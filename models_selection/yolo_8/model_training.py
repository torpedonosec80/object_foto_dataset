# model_training.py
import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import torch
from config import DEFAULT_SAVE_DIR

def train_yolo_model(data_yaml, epochs=30, batch_size=8, model_size='n', device=None):
    if device is None:
        device = '0' if torch.cuda.is_available() else 'cpu'
    
    model = YOLO(f'yolov8{model_size}.pt')
    
    train_params = {
        'data': data_yaml,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': 640,
        'device': device,
        'workers': 2,
        'name': 'yolo_object_detection'
    }
    
    results = model.train(**train_params)
    return model, results

def save_model(model, save_dir=DEFAULT_SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, 'best_yolo_model.pt')
    
    # Экспорт и сохранение
    model.export(format="torchscript", imgsz=640)
    model.save(best_model_path)
    
    # Попытка найти лучшую модель через trainer
    if hasattr(model, 'trainer'):
        trainer = model.trainer
        trainer_best_path = Path(trainer.save_dir) / 'weights' / 'best.pt'
        
        if trainer_best_path.exists():
            shutil.copy(str(trainer_best_path), best_model_path)
            print(f'Best model copied to {best_model_path}')
        else:
            last_model_path = Path(trainer.save_dir) / 'weights' / 'last.pt'
            if last_model_path.exists():
                shutil.copy(str(last_model_path), best_model_path)
                print(f'Used last.pt as best model (best.pt not found)')
    
    return best_model_path
