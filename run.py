import streamlit as st
from PIL import Image
import cv2
import numpy as np
import pickle

# Title
st.title("Flood Detection Project")

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert to array (for your model)
    img_array = np.array(image)
    
    # Load your trained model
    model = pickle.load(open("model.pkl", "rb"))
    
    # Example: preprocess image
    # feature = preprocess(img_array)  # Your preprocessing function
    # prediction = model.predict([feature])
    
    st.write("Prediction: Flooded / Non-Flooded (demo)")
