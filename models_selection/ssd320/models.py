import torch
import torchvision
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead

# Assuming DEVICE is imported from config
from .config import DEVICE

def get_model(num_classes):
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=weights)
    model = model.to(DEVICE)

    model.backbone.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 320, 320).to(DEVICE)
        features = model.backbone(dummy_input)
        in_channels_list = [features[str(i)].shape[1] for i in range(len(features))]

    num_anchors = model.anchor_generator.num_anchors_per_location()

    model.head.classification_head = SSDLiteClassificationHead(
        in_channels=in_channels_list,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=torch.nn.BatchNorm2d
    ).to(DEVICE)

    for layer in model.head.classification_head.modules():
        if isinstance(layer, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(layer.weight)

    return model

def load_model_for_inference(model_path, num_classes, device):
    model = get_model(num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model
