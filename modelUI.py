import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np

# =========================
# Config & Class Names
# =========================
CLASS_NAMES = ["Abnormal", "Normal", "STE"]
MODEL_PATH = "best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Load Model
# =========================
@st.cache_resource  # Cache model to avoid reloading on every interaction
def load_model():
    model = torch.load(MODEL_PATH, map_location=DEVICE)
    model.eval()
    return model

model = load_model()

# =========================
# Preprocessing Function
# =========================
def preprocess_ecg_image_bytes(image_bytes):
    # Convert bytes -> numpy
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    # --- Apply preprocessing ---
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    kernel = np.ones((1, 3), np.uint8)
    denoised = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.bitwise_not(denoised)

    # Convert OpenCV → PIL
    pil_img = Image.fromarray(cleaned)

    # Same normalization as training
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform(pil_img).unsqueeze(0).to(DEVICE)

# =========================
# Streamlit UI
# =========================
st.title("ECG Classification Demo")
st.write("Upload an ECG image to classify it.")

uploaded_file = st.file_uploader("Choose an ECG image...", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded ECG", use_column_width=True)

    if st.button("Predict"):
        input_tensor = preprocess_ecg_image_bytes(uploaded_file.getvalue())
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_idx = outputs.argmax(1).item()

        st.subheader("Prediction Result")
        st.write(f"**Predicted Class:** {CLASS_NAMES[pred_idx]}")
        st.json({CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))})




