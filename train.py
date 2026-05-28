import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

from model import AnimalCNN


# -------------------------------------------------------
# Dataset class - filters CIFAR-10 to bird, cat, dog only
# (same logic as in the notebook)
# -------------------------------------------------------

class AnimalDataset(Dataset):
    def __init__(self, cifar_data, animal_classes, label_map, transform=None):
        self.images = cifar_data.data
        self.labels = cifar_data.targets
        self.transform = transform
        self.label_map = label_map

        # collect the indices for images that belong to our 3 classes
        self.indices = []
        for i in range(len(self.labels)):
            if self.labels[i] in animal_classes:
                self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image = self.images[real_idx]
        label = self.labels[real_idx]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        new_label = self.label_map[label]
        return image, new_label


# -------------------------------------------------------
# Training and validation functions
# -------------------------------------------------------

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    # put model in training mode - this enables dropout and batchnorm training behaviour
    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_images = 0

    # loop through each batch in training data
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # clear gradients from the previous step
        optimizer.zero_grad()

        # forward pass - get the model's predictions
        outputs = model(images)

        # calculate how wrong the predictions are
        loss = criterion(outputs, labels)

        # backward pass - figure out how to adjust each weight
        loss.backward()

        # update the weights
        optimizer.step()

        running_loss += loss.item()

        # find the class with the highest score for each image
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_images += labels.size(0)

    average_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct_predictions / total_images
    return average_loss, accuracy


def validate(model, val_loader, criterion, device):
    # switch to eval mode - turns off dropout so results are consistent
    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_images = 0

    # no_grad means we skip gradient calculations - faster and uses less memory
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_images += labels.size(0)

    average_loss = running_loss / len(val_loader)
    accuracy = 100.0 * correct_predictions / total_images
    return average_loss, accuracy


# -------------------------------------------------------
# Settings
# -------------------------------------------------------

num_epochs = 20
learning_rate = 0.001
batch_size = 64
early_stopping_patience = 5   # stop training if val accuracy doesnt improve for this many epochs

# use GPU if one is available, otherwise fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# -------------------------------------------------------
# Data loading
# -------------------------------------------------------

animal_classes = [2, 3, 5]
class_names = ['bird', 'cat', 'dog']
label_map = {2: 0, 3: 1, 5: 2}

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

val_test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

print('Loading CIFAR-10 data...')

full_train_raw = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True)
full_test_raw  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

# two filtered versions of the training data - one with augmentation, one without
train_and_val_augmented = AnimalDataset(full_train_raw, animal_classes, label_map, transform=train_transforms)
train_and_val_plain     = AnimalDataset(full_train_raw, animal_classes, label_map, transform=val_test_transforms)
test_dataset            = AnimalDataset(full_test_raw,  animal_classes, label_map, transform=val_test_transforms)

total      = len(train_and_val_augmented)
val_size   = int(0.2 * total)
train_size = total - val_size

# same seed so the same images end up in each split
generator = torch.Generator().manual_seed(42)
train_dataset, _ = random_split(train_and_val_augmented, [train_size, val_size], generator=generator)

generator = torch.Generator().manual_seed(42)
_, val_dataset   = random_split(train_and_val_plain,     [train_size, val_size], generator=generator)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

print(f'Training images:   {len(train_dataset)}')
print(f'Validation images: {len(val_dataset)}')
print(f'Test images:       {len(test_dataset)}')


# -------------------------------------------------------
# Model, loss function, optimiser, scheduler
# -------------------------------------------------------

model     = AnimalCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# reduce learning rate by 10x every 7 epochs to help fine-tune later in training
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)


# -------------------------------------------------------
# Training loop
# -------------------------------------------------------

# these lists keep track of metrics so we can plot them later
train_losses     = []
val_losses       = []
train_accuracies = []
val_accuracies   = []

best_accuracy          = 0.0
early_stopping_counter = 0

os.makedirs('model',  exist_ok=True)
os.makedirs('static', exist_ok=True)

print('Starting training...')
print('-' * 50)

for epoch in range(1, num_epochs + 1):

    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss,   val_acc   = validate(model, val_loader, criterion, device)

    # step the scheduler after each epoch
    scheduler.step()

    # store metrics for plotting
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(f'Epoch {epoch}/{num_epochs}')
    print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
    print(f'  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%')

    # save the model if this is the best validation accuracy we have seen so far
    if val_acc > best_accuracy:
        best_accuracy = val_acc
        torch.save(model.state_dict(), 'model/best_model.pth')
        print('  Model improved! Saving...')
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        print(f'  No improvement. ({early_stopping_counter}/{early_stopping_patience})')

    print('-' * 50)

    # if there has been no improvement for several epochs in a row, stop early
    if early_stopping_counter >= early_stopping_patience:
        print(f'Early stopping triggered at epoch {epoch}. No improvement for {early_stopping_patience} epochs.')
        break

print(f'Training complete! Best validation accuracy: {best_accuracy:.2f}%')
print('Saving model...')  # left this in from when i was debugging the save path


# -------------------------------------------------------
# Plot and save training curves
# -------------------------------------------------------

epochs_ran = len(train_losses)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# loss plot
axes[0].plot(range(1, epochs_ran + 1), train_losses, label='Train Loss', marker='o', markersize=3)
axes[0].plot(range(1, epochs_ran + 1), val_losses,   label='Val Loss',   marker='o', markersize=3)
axes[0].set_title('Loss over epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

# accuracy plot
axes[1].plot(range(1, epochs_ran + 1), train_accuracies, label='Train Accuracy', marker='o', markersize=3)
axes[1].plot(range(1, epochs_ran + 1), val_accuracies,   label='Val Accuracy',   marker='o', markersize=3)
axes[1].set_title('Accuracy over epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].legend()

plt.tight_layout()
plt.savefig('static/training_curves.png')
print('Training curves saved to static/training_curves.png')
plt.show()
