import torch
from torch import nn, optim
import torchvision
from torchvision import datasets, transforms, models
import argparse
import os

# Define argument parser
parser = argparse.ArgumentParser(description="Train a neural network on a dataset.")
parser.add_argument("data_dir", type=str, help="Path to dataset directory")
parser.add_argument("--save_dir", type=str, default="checkpoints/", help="Directory to save checkpoints")
parser.add_argument("--arch", type=str, default="vgg16", choices=["vgg16", "resnet50"], help="Model architecture")
parser.add_argument("--learning_rate", type=float, default=0.0003, help="Learning rate")
parser.add_argument("--hidden_units", type=int, default=512, help="Number of hidden units in classifier")
parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
parser.add_argument("--gpu", action="store_true", help="Use GPU for training")

args = parser.parse_args()

# Set device
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transforms with improved augmentation
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets
train_dir = os.path.join(args.data_dir, "train")
valid_dir = os.path.join(args.data_dir, "valid")

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64)

# Load pre-trained model
if args.arch == "vgg16":
    model = models.vgg16(pretrained=True)
    input_size = 25088
elif args.arch == "resnet50":
    model = models.resnet50(pretrained=True)
    input_size = 2048

# Unfreeze last 10 layers for fine-tuning
for param in list(model.parameters())[-10:]:
    param.requires_grad = True

# Define classifier
classifier = nn.Sequential(
    nn.Linear(input_size, args.hidden_units),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(args.hidden_units, 102),
    nn.LogSoftmax(dim=1)
)

# Attach classifier to model
if args.arch == "vgg16":
    model.classifier = classifier
elif args.arch == "resnet50":
    model.fc = classifier

model.to(device)

# Define loss function and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam((model.classifier.parameters() if args.arch == "vgg16" else model.fc.parameters()), lr=args.learning_rate)

# Ensure save directory exists
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

best_checkpoint_path = os.path.join(args.save_dir, "best_checkpoint.pth")

# Training loop
best_acc = 0.0
print("Starting training...")
for epoch in range(args.epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total

    print(f"Epoch {epoch+1}/{args.epochs}.. "
          f"Train Loss: {running_loss/len(train_loader):.3f}.. "
          f"Train Accuracy: {train_acc:.2f}%.. "
          f"Validation Loss: {val_loss/len(valid_loader):.3f}.. "
          f"Validation Accuracy: {val_acc:.2f}%")

    # Save best checkpoint
    if val_acc > best_acc:
        best_acc = val_acc
        checkpoint_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch+1}.pth")
        
        torch.save({
            "arch": args.arch,
            "state_dict": model.state_dict(),
            "class_to_idx": train_data.class_to_idx,
            "hidden_units": args.hidden_units,
            "optimizer_state_dict": optimizer.state_dict()
        }, checkpoint_path)

        print(f"Checkpoint saved to {checkpoint_path}")

print("Training complete!")
print("Evaluating Test Set...")

# Test evaluation
test_dir = os.path.join(args.data_dir, "test")
test_data = datasets.ImageFolder(test_dir, transform=valid_transforms)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)

model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)

test_acc = 100 * test_correct / test_total
print(f"Final Test Accuracy: {test_acc:.2f}%")
