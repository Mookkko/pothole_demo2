import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("YOLOv11 Pothole Detection App 🚧") 

# 1️⃣ โหลดโมเดล (ที่เทรนแล้ว)
model = YOLO("runs/detect/train6/weights/best.pt")  # path ของ best.pt

# 2️⃣ อัปโหลดรูปภาพ
uploaded_image = st.file_uploader("Upload an image (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # 3️⃣ แสดงภาพต้นฉบับ
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

    # 4️⃣ แปลงภาพเป็น numpy array
    image = Image.open(uploaded_image)
    image_np = np.array(image)

    # 5️⃣ รัน YOLO inference
    st.info("Running YOLO detection...")
    results = model.predict(image_np, conf=0.4)

    # 6️⃣ แสดงภาพผลลัพธ์
    result_image = results[0].plot()[:, :, ::-1]  # convert BGR→RGB
    st.image(result_image, caption="Detection Result", use_container_width=True)
    st.success("Detection completed!")

    # 7️⃣ แสดงจำนวน pothole ที่เจอ
    boxes = results[0].boxes
    class_ids = boxes.cls.cpu().numpy().astype(int)
    class_names = [model.names[i] for i in class_ids]
    pothole_count = class_names.count("pothole")

    st.write(f"**Number of potholes detected:** {pothole_count}")
