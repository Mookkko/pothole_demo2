import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("YOLO Image Detection App :)")

# Load YOLO model
model = YOLO("best.pt")  # ใช้โมเดลที่คุณเทรนเอง

# Upload image
uploaded_image = st.file_uploader("Upload an image (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    image_np = np.array(image)

    st.info("Running YOLO object detection...")
    results = model.predict(image_np, conf=0.2, imgsz=512)

    # ✅ เพิ่มส่วนนี้เพื่อตรวจ
    st.write("🧠 Model class names:", model.names)
    st.write("📦 Detected boxes:", results[0].boxes)
    st.write("📊 Detection probabilities:", results[0].probs)

    # Plot result
    result_image = results[0].plot()
    st.image(result_image, caption="YOLO Detection Result", use_container_width=True)

    st.success("Detection completed!")
