# Import All the Required Libraries
import json
import cv2
from ultralytics import YOLO
import numpy as np
import math
import re
import os
from datetime import datetime
from paddleocr import PaddleOCR
import traceback

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# List of valid Indian state codes
STATE_CODES = [
    "AN", "AP", "AR", "AS", "BR", "CH", "DN", "DD", "DL", "GA", "GJ", "HR", "HP",
    "JK", "KA", "KL", "LD", "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "PY", "PN",
    "RJ", "SK", "TN", "TR", "UK", "UP", "WB"
]

# Create a Video Capture Object
cap = cv2.VideoCapture("E:/ANPR_YOLOv10/Resources/carLicence5.mp4")
if not cap.isOpened():
    print("Error: Could not open video file. Verify the path: E:/ANPR_YOLOv10/Resources/carLicence6.mp4")
    exit()

# Initialize the YOLOv8 Model
try:
    model = YOLO("weights/best.pt")
    print("YOLOv8 model loaded successfully")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

# Initialize the frame count
count = 0

# Class Names
className = ["License"]

# Initialize the Paddle OCR
try:
    ocr = PaddleOCR(use_angle_cls=True, use_gpu=False)
    print("PaddleOCR initialized successfully")
except Exception as e:
    print(f"Error initializing PaddleOCR: {e}")
    exit()

def paddle_ocr(frame, x1, y1, x2, y2):
    try:
        frame = frame[y1:y2, x1:x2]
        result = ocr.ocr(frame, det=False, rec=True, cls=False)
        text = ""
        for r in result:
            scores = r[0][1]
            if np.isnan(scores):
                scores = 0
            else:
                scores = int(scores * 100)
            if scores > 60:
                text = r[0][0]
        # Remove non-ASCII characters (e.g., Unicode characters like \u5dde)
        text = ''.join(char for char in text if ord(char) < 128)
        # Remove non-alphanumeric characters
        pattern = re.compile(r'[^A-Z0-9]')
        text = pattern.sub('', text)
        text = text.replace("???", "")
        text = text.replace("O", "0")
        text = text.replace("粤", "")

        # Post-processing: Validate the Indian license plate format
        # Format: XXNNX...NNNN (state code, RTO code, series, number)
        # - State code: 2 letters (already in STATE_CODES)
        # - RTO code: 2 digits
        # - Series: 1, 2, or 3 letters
        # - Number: 4 digits (0001–9999)

        # Minimum length check (2 + 2 + 1 + 4 = 9, max 2 + 2 + 3 + 4 = 11)
        if len(text) < 9 or len(text) > 11:
            print(f"Invalid length: {text} (length {len(text)}, expected 9–11)")
            return ""

        # Validate state code (first two characters)
        state_code = text[:2].upper()
        if state_code not in STATE_CODES:
            print(f"Invalid state code detected: {state_code} in {text}")
            return ""

        # Validate RTO code (next two characters, digits)
        rto_code = text[2:4]
        if not rto_code.isdigit():
            print(f"Invalid RTO code (not digits): {rto_code} in {text}")
            return ""

        # Validate series (1, 2, or 3 letters)
        series_start = 4
        series_length = len(text) - 8  # Total length - (state + RTO + number)
        if series_length not in [1, 2, 3]:
            print(f"Invalid series length: {series_length} in {text} (expected 1–3 letters)")
            return ""
        series = text[series_start:series_start + series_length]
        if not series.isalpha():
            print(f"Invalid series (not all letters): {series} in {text}")
            return ""

        # Validate number (last four characters, digits between 0001 and 9999)
        number = text[-4:]
        if not number.isdigit():
            print(f"Invalid number (not digits): {number} in {text}")
            return ""
        num_value = int(number)
        if not (1 <= num_value <= 9999):
            print(f"Invalid number range: {number} in {text} (expected 0001–9999)")
            return ""

        return str(text)
    except Exception as e:
        print(f"Error in paddle_ocr: {e}")
        return ""

def save_json(license_plates):
    try:
        # Save all license plates to a single JSON file
        data = {
            "Timestamp": datetime.now().isoformat(),
            "License Plates": list(license_plates)
        }
        os.makedirs("json", exist_ok=True)  # Ensure json folder exists
        with open("json/license_plates.json", 'w') as f:
            json.dump(data, f, indent=2)
        print("License plates saved to json/license_plates.json")
    except Exception as e:
        print(f"Error saving JSON: {e}")

license_plates = set()

try:
    while True:
        ret, frame = cap.read()
        if ret:
            count += 1
            print(f"Frame Number: {count}")
            results = model.predict(frame, conf=0.45)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    # Extract license plate text
                    label = paddle_ocr(frame, x1, y1, x2, y2)
                    # Add to set if label is non-empty
                    if label:
                        license_plates.add(label)
                        print(f"Detected License Plate: {label}")
                    # Display the label (even if empty, this won't cause issues)
                    textSize = cv2.getTextSize(label if label else "Unknown", 0, fontScale=0.5, thickness=2)[0]
                    c2 = x1 + textSize[0], y1 - textSize[1] - 3
                    cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
                    cv2.putText(frame, label if label else "Unknown", (x1, y1 - 2), 0, 0.5, [255,255,255], thickness=1, lineType=cv2.LINE_AA)
            cv2.imshow("Video", frame)
            if cv2.waitKey(1) & 0xFF == ord('1'):
                save_json(license_plates)
                break
        else:
            print("End of video or error reading frame")
            save_json(license_plates)
            break
except Exception as e:
    print(f"Error in main loop: {e}")
    traceback.print_exc()
finally:
    save_json(license_plates)  # Save any detected plates on exit
    cap.release()
    cv2.destroyAllWindows()
    print("Video capture released and windows closed")