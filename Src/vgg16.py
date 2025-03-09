import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import models, datasets, transforms
from torchvision.models import VGG16_Weights
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.utils.data import DataLoader, random_split

# Constants
IMG_SIZE = (224, 224)  # VGG16 expects 224x224 images
BATCH_SIZE = 32
NUM_EPOCHS = 55
NUM_WORKERS = 2

# Path to dataset
folder_path = '/kaggle/input/persian-food-20mehr/vn'

# Image transformations
train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomRotation(40),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# No augmentation for validation and test, only resize and normalize
eval_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load full dataset
full_dataset = datasets.ImageFolder(folder_path)

# Calculate splits (60% train, 20% validation, 20% test)
total_size = len(full_dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

# Split dataset
train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)  # For reproducibility
)


# Apply appropriate transforms to each dataset
class TransformDataset:
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return self.transform(image), label


# Wrap datasets with appropriate transforms
train_dataset = TransformDataset(train_dataset, train_transform)
val_dataset = TransformDataset(val_dataset, eval_transform)
test_dataset = TransformDataset(test_dataset, eval_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# Load the pre-trained VGG16 model
base_model = models.vgg16(weights=VGG16_Weights.DEFAULT)

# Freeze the feature layers
for param in base_model.features.parameters():
    param.requires_grad = False

# Modify the classifier
num_classes = len(full_dataset.classes)
base_model.classifier[6] = nn.Sequential(
    nn.Linear(base_model.classifier[6].in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes)
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = base_model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(base_model.classifier.parameters(), lr=1E-2, momentum=0.9, nesterov=True)
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

# Training history
history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': [],
}

# Early stopping parameters
best_val_loss = float('inf')
patience = 5
patience_counter = 0

# Training loop
for epoch in range(NUM_EPOCHS):
    # Training phase
    base_model.train()
    train_loss = 0.0
    train_predictions = []
    train_targets = []

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = base_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        train_predictions.extend(preds.cpu().numpy())
        train_targets.extend(labels.cpu().numpy())

    train_loss = train_loss / len(train_loader.dataset)
    train_accuracy = accuracy_score(train_targets, train_predictions)

    # Validation phase
    base_model.eval()
    val_loss = 0.0
    val_predictions = []
    val_targets = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = base_model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            val_predictions.extend(preds.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())

    val_loss = val_loss / len(val_loader.dataset)
    val_accuracy = accuracy_score(val_targets, val_predictions)

    # Update history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_accuracy)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_accuracy)

    print(f'Epoch {epoch + 1}/{NUM_EPOCHS}:')
    print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
    print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    print('-' * 50)

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save(base_model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    scheduler.step()

# Load best model for testing
base_model.load_state_dict(torch.load('best_model.pth'))

# Final evaluation on test set
base_model.eval()
test_predictions = []
test_targets = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = base_model(inputs)
        _, preds = torch.max(outputs, 1)
        test_predictions.extend(preds.cpu().numpy())
        test_targets.extend(labels.cpu().numpy())

# Calculate and print test metrics
test_accuracy = accuracy_score(test_targets, test_predictions)
print(f'\nFinal Test Accuracy: {test_accuracy:.4f}')
print('\nClassification Report:')
print(classification_report(test_targets, test_predictions, target_names=full_dataset.classes))

# Plot training history
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history['train_acc'], label='Train Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Plot confusion matrix
cm = confusion_matrix(test_targets, test_predictions)
plt.figure(figsize=(15, 15))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=full_dataset.classes,
            yticklabels=full_dataset.classes)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.show()

# Save final model
torch.save(base_model.state_dict(), 'final_vgg16_model.pth')
