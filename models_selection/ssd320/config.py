import torch
import torchvision

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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
BATCH_SIZE = 8
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
PRINT_FREQ = 20

# Path to data
DATA_DIR = r"C:\Users\MAFirsov\Nextcloud\python_projects\NETOLOGY\object_foto_dataset\boxes_dataset"

# Transformations
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Inference configuration
MODEL_PATH = r'C:\Users\MAFirsov\Nextcloud\python_projects\NETOLOGY\object_foto_dataset\models_selection\ssd320\saved_models\best_model.pth'  # Path to the best model
THRESHOLD = 0.5  
INFERENCE_DIR = DATA_DIR 
