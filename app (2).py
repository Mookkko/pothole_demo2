import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("YOLOv11 Pothole Detection App 🕳️")

# 1️⃣ โหลดโมเดล (ที่เทรนแล้ว)
model = YOLO("best.pt")   # ✅ เปลี่ยนจาก yolo11n.pt เป็น best.pt ที่คุณเทรนเอง

# 2️⃣ อัปโหลดรูปภาพ
uploaded_image = st.file_uploader("Upload an image (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # แปลงภาพและแสดง
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    image_np = np.array(image)

    # 3️⃣ รันโมเดล
    st.info("Running YOLO object detection...")
    results = model.predict(image_np, conf=0.1, imgsz=512)

    # 4️⃣ ดึงผลลัพธ์
    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        confs = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        class_names = [model.names[i] for i in class_ids]

        st.success(f"✅ Detected {len(class_names)} object(s): {class_names}")
        for name, conf in zip(class_names, confs):
            st.write(f"- {name} ({conf:.2f})")

        # 5️⃣ แสดงภาพที่มีกรอบวัตถุ
        result_image = results[0].plot()
        st.image(result_image, caption="Detection Result", use_container_width=True)

    else:
        st.warning("⚠️ No objects detected.")
