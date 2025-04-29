import streamlit as st
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np

# Load the model
model = YOLO('best.pt')
names_file = 'names.csv'
with open(names_file, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Streamlit interface
st.title('Car Model Detection')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)
    #cv2 use BGR
    img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # Run inference
    results = model.predict(img, imgsz=640)

    boxes = []
    confidences = []
    class_ids = []

    # Plot results
    for result in results:
        for data in result.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = data
            boxes.append([int(x1), int(y1), int(x2), int(y2)])
            confidences.append(float(confidence))
            class_ids.append(int(class_id))
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.3, nms_threshold=0.4)
    # Draw filtered bounding boxes
    for i in indices:
        x1, y1, x2, y2 = boxes[i]
        confidence = confidences[i]
        class_id = class_ids[i]
        class_name = class_names[class_id]
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 0, 255), 3)
        st.write(f"Detected class: {class_name}, Confidence: {confidence:.2f}")
    #streamlit use RGB
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    st.image(img_np, caption='Detected Image', use_container_width=True)
