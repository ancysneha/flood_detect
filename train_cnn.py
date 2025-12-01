import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "dataset")

MODEL_PATH = "model.pth"
BATCH_SIZE = 4
EPOCHS = 10
LR = 0.0001

# Data Augmentation (IMPORTANT for your small dataset)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Datasets
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, ""), transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Classes found:", train_dataset.classes)

# Load Pretrained ResNet-50
model = models.resnet50(weights="IMAGENET1K_V1")

# Modify final layer (2 classes: flooded / non_flooded)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training
print("\nTraining started...\n")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(), MODEL_PATH)
print("\n🎉 Model saved as model.pth")
