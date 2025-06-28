import torch
import gc
from copy import deepcopy
import matplotlib.pyplot as plt
import os

# Assuming necessary components are imported from other modules
from .config import DEVICE, NUM_EPOCHS, PRINT_FREQ, CLASSES, NUM_CLASSES, DATA_DIR, LEARNING_RATE
from .datasets import CustomDataset, collate_fn
from .models import get_model
from .evaluate import calculate_map # Import calculate_map for validation

def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device, print_freq, saved_models_dir='saved_models'):
    """
    Trains the object detection model.

    Args:
        model: The PyTorch model to train.
        train_loader: DataLoader for the training dataset.
        val_loader: DataLoader for the validation dataset.
        optimizer: The optimizer for training.
        scheduler: The learning rate scheduler.
        num_epochs (int): The total number of training epochs.
        device: The device (CPU or GPU) to train on.
        print_freq (int): Frequency of printing training loss.
        saved_models_dir (str): Directory to save model checkpoints.
    """
    os.makedirs(saved_models_dir, exist_ok=True)

    best_map = 0.0
    best_model_weights = None
    train_losses = []
    val_maps = []

    for epoch in range(num_epochs):
        # === Training Phase ===
        model.train()
        epoch_loss = 0.0
        for i, (images, targets, _) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            if i % print_freq == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {i+1}/{len(train_loader)}, Loss: {losses.item():.4f}')

        epoch_avg_loss = epoch_loss / len(train_loader)
        train_losses.append(epoch_avg_loss)
        print(f'Epoch {epoch+1} Train Loss: {epoch_avg_loss:.4f}')

        # === Validation Phase ===
        # Pass the correct classes dictionary to calculate_map if needed internally
        val_metrics = calculate_map(model, val_loader, device)
        val_map = val_metrics['map'].item()
        val_maps.append(val_map)

        print(f'Epoch {epoch+1} Validation mAP: {val_map:.4f}')
        # Check if 'map_50' and 'map_75' exist in the metrics dictionary
        map_50 = val_metrics.get('map_50', torch.tensor(0.0)).item()
        map_75 = val_metrics.get('map_75', torch.tensor(0.0)).item()
        print(f'Validation metrics: mAP@50={map_50:.4f}, mAP@75={map_75:.4f}')


        # Update scheduler
        scheduler.step(val_map)

        # Free up memory
        torch.cuda.empty_cache()
        gc.collect()

        # === Saving Models ===
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(saved_models_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_avg_loss,
                'val_map': val_map,
            }, checkpoint_path)
            print(f'Saved checkpoint: {checkpoint_path}')

        if val_map > best_map:
            best_map = val_map
            best_model_weights = deepcopy(model.state_dict())
            best_model_path = os.path.join(saved_models_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_weights,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_avg_loss,
                'val_map': val_map,
            }, best_model_path)
            print(f'New best model saved with mAP {best_map:.4f} at {best_model_path}')

    # Save the final model
    final_model_path = os.path.join(saved_models_dir, 'final_model.pth')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': epoch_avg_loss,
        'val_map': val_map, # Use the val_map from the last epoch
    }, final_model_path)
    print(f'Training complete. Final model saved at {final_model_path}')

    # --- Moved final test and plotting to main or evaluation module ---
    # The training function should primarily handle the training loop and saving.

    return model, train_losses, val_maps, best_model_weights
