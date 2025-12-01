import torch
from PIL import Image
import numpy as np
from cnn_model import FloodCNN

# Path of image to test
IMAGE_PATH = input("Enter image path: ")  # e.g. dataset/flooded/img1.jpg
   # Change to your test image
   # <-- change this to your image name

# Load image
image = Image.open(IMAGE_PATH).resize((128,128))
arr = np.array(image) / 255.0
arr = torch.tensor(arr).permute(2,0,1).unsqueeze(0).float()

# Load CNN model
model = FloodCNN()
model.eval()

# Predict flood probability
with torch.no_grad():
    pred = model(arr)

print("Flood Probability:", float(pred))
