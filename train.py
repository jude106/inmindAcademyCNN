import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import os

from model import SimpleNet

# Load config (YAML for easy editing)
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# CIFAR-10 specific Mean and Standard Deviation for mathematically sound normalization
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)

# Training transforms: includes Data Augmentation to prevent overfitting
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    # ---> STATE OF THE ART AUGMENTATION <---
    # Automatically applies a randomized mix of extreme distortions
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    # ---> THE SECRET WEAPON: CUTOUT / RANDOM ERASING <---
    transforms.RandomErasing(p=0.5, scale=(0.05, 0.2)),
    transforms.Normalize(cifar10_mean, cifar10_std)
])

# Testing/Validation transforms: NO augmentation, just normalization
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)
])

def get_loaders():
    # Ensure train and test data directories exist
    os.makedirs(config['paths']['train_dir'], exist_ok=True)
    os.makedirs(config['paths']['test_dir'], exist_ok=True)

    # Load the base training dataset TWICE to apply different transforms
    dataset_train_full = datasets.CIFAR10(
        root=config['paths']['train_dir'], train=True, download=True, transform=transform_train
    )
    dataset_val_full = datasets.CIFAR10(
        root=config['paths']['train_dir'], train=True, download=True, transform=transform_test
    )

    dataset_test = datasets.CIFAR10(
        root=config['paths']['test_dir'], train=False, download=True, transform=transform_test
    )

    # Calculate split sizes
    val_split = config['hyperparameters'].get('val_split', 0.1)
    n_total = len(dataset_train_full)
    n_val = int(n_total * val_split)

    # Create fixed indices for a clean train/val split
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(n_total, generator=generator).tolist()
    train_idx = indices[:-n_val]
    val_idx = indices[-n_val:]

    # Map the subsets to the correctly transformed datasets
    dataset_train = Subset(dataset_train_full, train_idx)
    dataset_val = Subset(dataset_val_full, val_idx)

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=config['hyperparameters']['batch_size'],
        shuffle=True,
        num_workers=config['hyperparameters']['num_workers']
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=config['hyperparameters']['batch_size'],
        shuffle=False,  # No need to shuffle validation data
        num_workers=config['hyperparameters']['num_workers']
    )
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=config['hyperparameters']['batch_size'],
        shuffle=False,
        num_workers=config['hyperparameters']['num_workers']
    )
    return dataloader_train, dataloader_val, dataloader_test


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def train(model, dataloader_train, dataloader_val, criterion, optimizer, scheduler, device):
    model.train()
    epochs = config['hyperparameters']['epochs']
    for epoch in range(epochs):
        running_loss = 0.0
        with tqdm(
            dataloader_train,
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=True,
            unit="batch"
        ) as progress_bar:
            for i, (inputs, labels) in enumerate(progress_bar):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # ---> ONE CYCLE POLICY: Step the scheduler AFTER EVERY BATCH <---
                scheduler.step()
                
                running_loss += loss.item()
                avg_loss = running_loss / (i + 1)
                progress_bar.set_postfix({'loss': avg_loss})
        
        avg_train_loss = running_loss / len(dataloader_train)
        val_loss, val_acc = evaluate(model, dataloader_val, criterion, device)
        
        # Get current learning rate to monitor the One Cycle curve
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1} finished. Train loss: {avg_train_loss:.3f} | Val loss: {val_loss:.3f} | Val acc: {val_acc:.2f}% | LR: {current_lr:.6f}")
    print('Finished Training')


# Function to test the neural network and print accuracy
def test(model, dataloader_test, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0  # Count of correct predictions
    total = 0  # Total number of samples
    with torch.no_grad():  # No need to compute gradients during testing (save memory)
        for images, labels in dataloader_test:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # Get model predictions
            _, predicted = torch.max(outputs, 1)  # Get predicted class
            total += labels.size(0)  # Update total count
            correct += (predicted == labels).sum().item()  # Update correct count
    # Print accuracy as a percentage
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f} %')


def main():
    # Select device: use GPU if available, otherwise use CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Get data loaders for training and testing
    dataloader_train, dataloader_val, dataloader_test = get_loaders()

    # Create the neural network and move it to the selected device
    model = SimpleNet().to(device)

    # ---> LABEL SMOOTHING <---
    # Prevents overconfidence by making the target 90% instead of 100%
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Define the optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['hyperparameters']['lr'], # This gets overridden by OneCycleLR
        weight_decay=config['hyperparameters'].get('weight_decay', 0.01)
    )

    # ---> Define the One Cycle Learning Rate Scheduler for Super-Convergence <---
    epochs = config['hyperparameters']['epochs']
    steps_per_epoch = len(dataloader_train)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config['hyperparameters']['lr'], 
        epochs=epochs, 
        steps_per_epoch=steps_per_epoch
    )

    # Pass the scheduler to the train function
    train(model, dataloader_train, dataloader_val, criterion, optimizer, scheduler, device)
    
    test_loss, test_acc = evaluate(model, dataloader_test, criterion, device)
    print(f'Test loss: {test_loss:.3f} | Test acc: {test_acc:.2f}%')

    # Save the trained model's parameters to a file
    os.makedirs(os.path.dirname(config['paths']['model_path']), exist_ok=True)  # Create weights dir if missing
    torch.save(model.state_dict(), config['paths']['model_path']) # Save the model's parameters to a pth file
    print(f"Model saved to {config['paths']['model_path']}")

if __name__ == '__main__':
    main()
