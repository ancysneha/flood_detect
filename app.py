https://github.com/ancysneha/flood_detect# app.py (final)
import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import sys, os
import sqlite3
from datetime import datetime

# allow local imports
sys.path.append(os.path.dirname(__file__))

# import local weather helper (weather_api.py must be in same folder)
from weather_api import get_weather

from cnn_model import FloodCNN
from lstm_model import LSTMForecast

# ======================================================
#  Combined Model
# ======================================================
class CombinedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = FloodCNN()                     # outputs (B,2)
        self.lstm = LSTMForecast()                # outputs (B,1)
        self.fc = torch.nn.Linear(2, 2)

    def forward(self, img, seq):
        cnn_out = self.cnn(img)                   # shape = (B,2)
        lstm_val = self.lstm(seq)                 # shape = (B,1)
        lstm_out = torch.cat([lstm_val, lstm_val], dim=1)
        return cnn_out + lstm_out                 # final logits


# ======================================================
#  Load Model
# ======================================================
DEVICE = "cpu"
model = CombinedModel()
model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ======================================================
#  Database Setup
# ======================================================
def init_db():
    conn = sqlite3.connect("flood_logs.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            prediction TEXT,
            confidence REAL,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_log(filename, prediction, confidence):
    conn = sqlite3.connect("flood_logs.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO logs (filename, prediction, confidence, timestamp)
        VALUES (?, ?, ?, ?)
    """, (filename, prediction, confidence, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

# Create DB on start
init_db()

# ======================================================
#  Streamlit UI
# ======================================================
st.set_page_config(page_title="Flood Detection", layout="centered")
st.title("🌊 Flood Detection (CNN + LSTM)")
st.write("Upload an image to check flood probability. Live weather for Coimbatore is used as LSTM input.")

# ---------------------------------------
# Weather Section (Coimbatore by default)
# ---------------------------------------
st.subheader("🌦 Live Weather (Coimbatore)")

weather = get_weather("Coimbatore")
if weather:
    # weather_api returns metric values already
    temp = weather["temperature"]
    humidity = weather["humidity"]
    pressure = weather["pressure"]
    desc = weather["description"]

    st.write(f"**Temperature:** {temp:.1f} °C")
    st.write(f"**Humidity:** {humidity} %")
    st.write(f"**Pressure:** {pressure} hPa")
    st.write(f"**Conditions:** {desc}")
else:
    st.warning("⚠️ Unable to fetch weather data. Check your API key or internet connection.")

st.markdown("---")

# ---------------------------------------
# Image Upload & Prediction
# ---------------------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_tensor = transform(img).unsqueeze(0)  # shape (1,3,128,128)

    # Build LSTM input sequence from live weather (length 10)
    # We'll make a simple 10-step sequence where first step contains rain/humidity/temperature info and rest are zeros.
    # LSTM expects shape (batch, seq_len, feature_dim). Your LSTM was built for input_size=1, so we feed a single scalar.
    # Here we use humidity as a proxy signal (you can change to rainfall if you have time-series).
    if weather:
        # choose a scalar from weather (humidity scaled 0-1)
        humidity_val = weather["humidity"] / 100.0
        # create a 10-step sequence with humidity as first value and small decays
        seq_array = np.zeros((1, 10, 1), dtype=np.float32)
        seq_array[0, 0, 0] = humidity_val
        # small decay values so LSTM has some pattern
        for i in range(1, 10):
            seq_array[0, i, 0] = max(0.0, humidity_val * (0.9 ** i))
        seq = torch.tensor(seq_array, dtype=torch.float32)
    else:
        # fallback to random if weather not available
        seq = torch.tensor(np.random.rand(1, 10, 1), dtype=torch.float32)

    with torch.no_grad():
        out = model(img_tensor, seq)
        probs = torch.softmax(out, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    # ------------ Label Mapping ----------------
    label = "Flood Detected" if pred == 0 else "No Flood"

    st.subheader("📌 Result:")
    if pred == 0:
        st.success(f"🚨 Flood Detected — Confidence: {confidence:.4f}")
    else:
        st.info(f"✅ No Flood — Confidence: {confidence:.4f}")

    st.write("📈 *(LSTM rainfall/weather signal also influenced the result)*")

    # ------------ Save to DB ----------------
    save_log(uploaded_file.name, label, confidence)
    st.success("🗃️ Prediction saved to database!")

# ======================================================
#  Show Logs Table
# ======================================================
st.markdown("---")
if st.checkbox("Show Prediction History"):
    conn = sqlite3.connect("flood_logs.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM logs ORDER BY id DESC")
    data = cursor.fetchall()
    conn.close()

    st.write("### 📜 Prediction History:")
    for row in data:
        st.write(
            f"**ID:** {row[0]} | "
            f"**File:** {row[1]} | "
            f"**Prediction:** {row[2]} | "
            f"**Confidence:** {row[3]:.4f} | "
            f"**Time:** {row[4]}"
        )
