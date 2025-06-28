import torch
import pandas as pd
from tqdm import tqdm
import os

# Assuming necessary components are imported from other modules
from .config import DEVICE, CLASSES, THRESHOLD, INFERENCE_DIR, MODEL_PATH, NUM_CLASSES, transform
from .datasets import InferenceDataset
from .models import load_model_for_inference

def run_inference(model, inference_loader, device, confidence_threshold, output_csv_path='detection_results.csv'):
    """
    Runs inference on a dataset and saves the detection results to a CSV file.

    Args:
        model: The PyTorch model to use for inference.
        inference_loader: DataLoader for the inference dataset.
        device: The device (CPU or GPU) to run inference on.
        confidence_threshold (float): The minimum confidence score for a detection to be included.
        output_csv_path (str): The path to save the detection results CSV.

    Returns:
        pandas.DataFrame: DataFrame containing the detection results.
    """
    model.eval()
    results = []

    with torch.no_grad():
        for images, image_names in tqdm(inference_loader, desc="Processing images"):
            # Ensure images is a list of tensors before moving to device
            images = [img.to(device) for img in images]
            predictions = model(images)

            for img_name, pred in zip(image_names, predictions):
                boxes = pred['boxes'].cpu().numpy()
                scores = pred['scores'].cpu().numpy()
                labels = pred['labels'].cpu().numpy()

                for i in range(len(boxes)):
                    if scores[i] > confidence_threshold:
                        results.append({
                            'image_name': img_name,
                            'class_id': labels[i],
                            'class_name': CLASSES.get(labels[i], f'unknown_{labels[i]}'), # Handle potential unknown labels
                            'xmin': boxes[i][0],
                            'ymin': boxes[i][1],
                            'xmax': boxes[i][2],
                            'ymax': boxes[i][3],
                            'confidence': scores[i]
                        })

    df_results = pd.DataFrame(results)

    output_dir = os.path.dirname(output_csv_path)
    if output_dir: # Create directory if output_csv_path includes a directory
        os.makedirs(output_dir, exist_ok=True)

    df_results.to_csv(output_csv_path, index=False)

    print(f"Detection completed! Results saved to {output_csv_path}")
    print(f"Total objects detected: {len(df_results)}")
    # Note: Cannot easily print images processed count here without access to the original dataset length

    return df_results
