from ultralytics import YOLO
import os
import torch

def run_train_model(model_weights, YOLO_DATA_DIR, NUM_EPOCHS, BATCH_SIZE, DEVICE):
  # Инициализация модели. Используем детекционную модель
  model = YOLO(model_weights) 
    
  # Параметры обучения
  train_params = {
      'data': os.path.join(YOLO_DATA_DIR, 'dataset.yaml'),
      'epochs': NUM_EPOCHS,
      'batch': BATCH_SIZE,
      'imgsz': 640,
      'device': DEVICE,
      'workers': 2,
      'name': 'yolo_object_detection'
  }
  
  # Обучение модели
  results = model.train(**train_params)
  return results
