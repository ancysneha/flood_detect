import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from cnn_model import FloodCNN
from lstm_model import LSTMForecast

IMG_SIZE = 128
BATCH_SIZE = 4
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------
# DATASET CLASS
# ---------------------------------------
class FloodDataset(Dataset):
    def __init__(self, root):
        self.images = []
        self.labels = []

        for label, folder in enumerate(["flooded", "non_flooded"]):
            path = os.path.join(root, folder)
            for img_name in os.listdir(path):
                img_path = os.path.join(path, img_name)
                self.images.append(img_path)
                self.labels.append(label)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)

        seq = torch.tensor(np.random.rand(10, 1), dtype=torch.float32)

        return img, seq, label

    def __len__(self):
        return len(self.images)


# ---------------------------------------
# COMBINED MODEL
# ---------------------------------------
class CombinedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = FloodCNN()
        self.lstm = LSTMForecast()
        self.fc = nn.Linear(2, 2)  # final 2-class output

    def forward(self, img, seq):
        cnn_out = self.cnn(img)      # (B,2)
        lstm_val = self.lstm(seq)    # (B,1)

        # convert lstm output to (B,2)
        lstm_out = torch.cat([lstm_val, lstm_val], dim=1)

        # combine signals
        out = cnn_out + lstm_out
        return out


# ---------------------------------------
# TRAINING LOOP
# ---------------------------------------
def train():
    dataset = FloodDataset("dataset")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = CombinedModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for imgs, seqs, labels in loader:
            imgs = imgs.to(DEVICE)
            seqs = seqs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(imgs, seqs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("\n✔ Model saved as model.pth")


if __name__ == "__main__":
    train()
