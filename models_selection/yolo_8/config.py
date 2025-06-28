# config.py
CLASSES = {
    0: 'background',
    1: 'shta-9m',
    2: 'shta-ps',
    3: 'shta-7m',
    4: 'e2-6u',
    5: 'shta-9',
    6: 'shta-3',
    7: 'sv-5'
}
CLASS_NAME_TO_ID = {v: k for k, v in CLASSES.items()}
NUM_CLASSES = len(CLASSES)

# Пути по умолчанию
DEFAULT_DATA_DIR = "/content/object_foto_dataset/boxes_dataset"
DEFAULT_YOLO_DIR = "/content/yolo_dataset"
DEFAULT_SAVE_DIR = "saved_models"
