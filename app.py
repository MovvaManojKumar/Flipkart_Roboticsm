import streamlit as st
import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import re
from datetime import datetime, timedelta

# Load YOLO models
brand_model = YOLO('b.pt')  # Replace 'b.pt' with the correct path to your YOLO model for brands
ocr = PaddleOCR(lang='en')  # Initialize PaddleOCR
fruit_model = load_model('DenseNet20_model.h5')  # Replace with the correct path to your fruit freshness model

# Class names for freshness detection
class_names = {
    0: 'Banana_Bad', 1: 'Banana_Good', 2: 'Fresh', 3: 'FreshCarrot', 4: 'FreshCucumber',
    5: 'FreshMango', 6: 'FreshTomato', 7: 'Guava_Bad', 8: 'Guava_Good', 9: 'Lime_Bad',
    10: 'Lime_Good', 11: 'Rotten', 12: 'RottenCarrot', 13: 'RottenCucumber',
    14: 'RottenMango', 15: 'RottenTomato', 16: 'freshBread', 17: 'rottenBread'
}

# Helper function: Extract expiry dates
def extract_expiry_dates(text):
    patterns = [
        r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})',  # 20/07/2024
        r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2})',  # 20/07/24
        r'(\d{1,2}\s*[A-Za-z]{3,}\s*\d{4})',  # 20 MAY 2024
    ]
    dates = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        dates.extend(matches)
    return dates

# Helper function: Preprocess image for fruit freshness
def preprocess_image(image):
    img = cv2.resize(image, (128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Streamlit App
st.title("Live Detection App")

# Sidebar options
app_mode = st.sidebar.selectbox(
    "Choose the mode",
    ["Home", "Live Brand & Text Detection", "Live Fruit Freshness Detection"]
)

if app_mode == "Home":
    st.markdown("""
    ## Welcome to the Live Detection App
    Use the sidebar to choose between:
    - **Live Brand & Text Detection**: Detect brands, extract text, and identify expiry dates.
    - **Live Fruit Freshness Detection**: Detect and classify the freshness of fruits.
    """)

elif app_mode == "Live Brand & Text Detection":
    st.header("Live Brand & Text Detection")
    run_detection = st.button("Start Detection")
    stop_detection = st.button("Stop Detection")

    if run_detection:
        # Open webcam
        cap = cv2.VideoCapture(0)
        st_frame = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Brand detection
            results = brand_model.predict(source=frame, stream=False)
            detected_frame = results[0].plot()

            # OCR for text extraction
            _, img_buffer = cv2.imencode('.jpg', frame)
            ocr_result = ocr.ocr(img_buffer.tobytes())
            if ocr_result and isinstance(ocr_result[0], list) and len(ocr_result[0]) > 0:
                extracted_text = ' '.join([line[1][0] for line in ocr_result[0]])
                expiry_dates = extract_expiry_dates(extracted_text)
            else:
                extracted_text = "No text detected"
                expiry_dates = []

            # Count objects
            object_counts = {}
            for box in results[0].boxes.data.cpu().numpy():
                label = results[0].names[int(box[5])]
                object_counts[label] = object_counts.get(label, 0) + 1

            # Overlay information
            y_offset = 50
            cv2.putText(detected_frame, f"Extracted Text: {extracted_text}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 20
            cv2.putText(detected_frame, f"Expiry Dates: {', '.join(expiry_dates) if expiry_dates else 'None'}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 20

            for label, count in object_counts.items():
                cv2.putText(detected_frame, f"{label}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 20

            # Display the frame
            st_frame.image(cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB), channels="RGB")
            if stop_detection:
                break

        cap.release()

elif app_mode == "Live Fruit Freshness Detection":
    st.header("Live Fruit Freshness Detection")
    run_detection = st.button("Start Detection")
    stop_detection = st.button("Stop Detection")

    if run_detection:
        # Open webcam
        cap = cv2.VideoCapture(0)
        st_frame = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess and predict
            img_array = preprocess_image(frame)
            predictions = fruit_model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            label = class_names[predicted_class]
            confidence = predictions[0][predicted_class] * 100

            # Overlay predictions on the frame
            cv2.putText(frame, f"Label: {label}, Confidence: {confidence:.2f}%", 
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display the frame
            st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            if stop_detection:
                break

        cap.release()
