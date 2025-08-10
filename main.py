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

# Create a Video Capture Object
cap = cv2.VideoCapture("E:/ANPR_YOLOv10/Resources/carLicence5.mp4")
if not cap.isOpened():
    print("Error: Could not open video file. Verify the path: E:/ANPR_YOLOv10/Resources/carLicence4.mp4")
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
        pattern = re.compile(r'[\W]')  # Fixed regex
        text = pattern.sub('', text)
        text = text.replace("???", "")
        text = text.replace("O", "0")
        text = text.replace("粤", "")
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
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    classNameInt = int(box.cls[0])
                    clsName = className[classNameInt]
                    conf = math.ceil(box.conf[0]*100)/100
                    label = paddle_ocr(frame, x1, y1, x2, y2)
                    if label:
                        license_plates.add(label)
                        print(f"Detected License Plate: {label}")
                    textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                    c2 = x1 + textSize[0], y1 - textSize[1] - 3
                    cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
                    cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255,255,255], thickness=1, lineType=cv2.LINE_AA)
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