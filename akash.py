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
import time
import psycopg2  # For Supabase database connection
import requests  # For Supabase storage upload

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Adjustable Thresholds
AREA_CHANGE_THRESHOLD = 0.02  # Lowered for better sensitivity (2% change)
MIN_FRAMES_FOR_STATIONARY = 5
TRACKING_MATCH_DISTANCE = 50
PLATE_CONFIDENCE_THRESHOLD = 0.1
SKIP_FRAMES = 3

# List of valid Indian state codes
STATE_CODES = [
    "AN", "AP", "AR", "AS", "BR", "CH", "DN", "DD", "DL", "GA", "GJ", "HR", "HP",
    "JK", "KA", "KL", "LD", "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "PY", "PN",
    "RJ", "SK", "TN", "TR", "UK", "UP", "WB"
]

# Supabase database and storage details
DB_HOST = "aws-0-ap-southeast-1.pooler.supabase.com"
DB_NAME = "postgres"
DB_USER = "postgres.zwlzviyircbejbjaggol"
DB_PASS = "fI2mCiweTssxPsVg"
DB_PORT = "5432"

SUPABASE_URL = "https://zwlzviyircbejbjaggol.supabase.co/storage/v1/object"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inp3bHp2aXlpcmNiZWpiamFnZ29sIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0MDU0Nzc1MSwiZXhwIjoyMDU2MTIzNzUxfQ.vGHatMjI24kfE5STZbheYhSXHG-KOf8g2YENITDM-I0"

# Live stream URL from IP Webcam
stream_url = "rtsp://admin:123@192.168.0.46:8080/h264_ulaw.sdp"  # Use the URL that worked in VLC
cap = cv2.VideoCapture(stream_url)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Video writer setup
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Stream dimensions: {frame_width}x{frame_height}")
if frame_width <= 0 or frame_height <= 0:
    print("Error: Invalid stream dimensions.")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))
if not out.isOpened():
    print("Warning: Video writer failed to initialize. Proceeding without video output.")
    out = None
else:
    print("Video writer initialized successfully.")

# Load YOLO Models
try:
    vehicle_model = YOLO(r"E:\ANPR_YOLOv10\weights\vehicle.pt")
    plate_model = YOLO(r"E:\ANPR_YOLOv10\weights\best.pt")
    print("YOLO models loaded successfully")
except Exception as e:
    print(f"Error loading YOLO models: {e}")
    exit()

# Initialize Paddle OCR
try:
    ocr = PaddleOCR(use_angle_cls=True, use_gpu=False)
    print("PaddleOCR initialized successfully")
except Exception as e:
    print(f"Error initializing PaddleOCR: {e}")
    exit()

# Output directory for saved frames
OUTPUT_DIR = "saved_frames"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to upload an image to Supabase storage
def upload_image(frame, timestamp, vid, plate):
    # Convert the frame to JPEG bytes
    _, buffer = cv2.imencode('.jpg', frame)
    image_data = buffer.tobytes()
    
    # Create a filename using timestamp, vehicle ID, and plate number
    filename = f"violations/violation_{timestamp}_vehicle_{vid}_plate_{plate}.jpg"
    url = f"{SUPABASE_URL}/{filename}"
    
    headers = {
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "image/jpeg",
        "x-upsert": "true"
    }
    
    response = requests.post(url, data=image_data, headers=headers)
    if response.status_code in [200, 201]:
        image_url = f"https://zwlzviyircbejbjaggol.supabase.co/storage/v1/object/public/{filename}"
        print(f"Image uploaded to Supabase: {image_url}")
        return image_url
    print(f"Upload error: {response.status_code} - {response.text}")
    return None

# Function to log violation into Supabase database
def log_violation(plate_number, image_url, timestamp):
    try:
        conn = psycopg2.connect(host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASS, port=DB_PORT)
        cursor = conn.cursor()
        lat, lon = 12.9716, 77.5946  # Default coordinates (Bangalore)
        cursor.execute("SELECT plate FROM violations WHERE plate = %s;", (plate_number,))
        if cursor.fetchone():
            cursor.execute("UPDATE violations SET frame = %s, lat = %s, lon = %s, images = %s, fine = %s, paid = %s WHERE plate = %s;",
                           (timestamp, lat, lon, f'["{image_url}"]', 500.00, False, plate_number))
        else:
            cursor.execute("INSERT INTO violations (plate, frame, lat, lon, images, fine, paid) VALUES (%s, %s, %s, %s, %s, %s, %s);",
                           (plate_number, timestamp, lat, lon, f'["{image_url}"]', 500.00, False))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"Violation logged/updated for plate: {plate_number}")
    except Exception as e:
        print(f"Database error: {e}")

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
        # Remove non-ASCII characters
        text = ''.join(char for char in text if ord(char) < 128)
        # Remove non-alphanumeric characters
        pattern = re.compile(r'[^A-Z0-9]')
        text = pattern.sub('', text)
        text = text.replace("???", "")
        text = text.replace("O", "0")
        text = text.replace("粤", "")

        # Validate Indian license plate format
        if len(text) < 9 or len(text) > 11:
            print(f"Invalid length: {text} (length {len(text)}, expected 9–11)")
            return ""

        # Validate state code
        state_code = text[:2].upper()
        if state_code not in STATE_CODES:
            print(f"Invalid state code detected: {state_code} in {text}")
            return ""

        # Validate RTO code
        rto_code = text[2:4]
        if not rto_code.isdigit():
            print(f"Invalid RTO code (not digits): {rto_code} in {text}")
            return ""

        # Validate series
        series_start = 4
        series_length = len(text) - 8
        if series_length not in [1, 2, 3]:
            print(f"Invalid series length: {series_length} in {text} (expected 1–3 letters)")
            return ""
        series = text[series_start:series_start + series_length]
        if not series.isalpha():
            print(f"Invalid series (not all letters): {series} in {text}")
            return ""

        # Validate number
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
        data = {
            "Timestamp": datetime.now().isoformat(),
            "License Plates": list(license_plates)
        }
        os.makedirs("json", exist_ok=True)
        with open("json/license_plates.json", 'w') as f:
            json.dump(data, f, indent=2)
        print("License plates saved to json/license_plates.json")
    except Exception as e:
        print(f"Error saving JSON: {e}")

# Process video stream
frame_count = 0
tracked_vehicles = {}
last_yellow_line_x = None
license_plates = set()

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from stream. Retrying...")
            cap.release()
            cap = cv2.VideoCapture(stream_url)
            time.sleep(2)  # Wait before retrying
            continue
        
        frame_count += 1
        timestamp = int(time.time())
        display_frame = frame.copy()
        height, width = frame.shape[:2]
        
        if frame_count % SKIP_FRAMES != 0:
            if out:
                out.write(display_frame)
            continue
        
        # Yellow lane detection
        roi_width = int(width * 0.3)
        roi_x_start = (width - roi_width) // 2
        roi_x_end = roi_x_start + roi_width
        roi = frame[:, roi_x_start:roi_x_end]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([15, 80, 80])
        upper_yellow = np.array([35, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        edges = cv2.Canny(mask, 30, 100)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=100)
        
        yellow_line_x = None
        if lines is not None:
            x_coords = [(x1 + x2) / 2 + roi_x_start for x1, _, x2, _ in lines[:, 0]]
            yellow_line_x = int(np.mean(x_coords))
            print(f"Yellow line detected at x = {yellow_line_x}")
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(display_frame, (x1 + roi_x_start, y1), (x2 + roi_x_start, y2), (0, 255, 255), 5)
        
        if yellow_line_x is None and last_yellow_line_x is not None:
            yellow_line_x = last_yellow_line_x
            print(f"Using last yellow line x = {yellow_line_x}")
        elif yellow_line_x is None:
            print("Yellow Line Not Detected")
            if out:
                out.write(display_frame)
            continue
        else:
            last_yellow_line_x = yellow_line_x
        
        # Detect vehicles
        vehicle_results = vehicle_model(frame)
        current_vehicles = []
        for result in vehicle_results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
            class_id = int(result.cls[0])
            class_name = vehicle_model.names[class_id]
            vehicle_center_x = (x1 + x2) // 2
            if vehicle_center_x > yellow_line_x:
                continue  # Skip vehicles on the right side of the yellow line
            area = (x2 - x1) * (y2 - y1)
            current_vehicles.append({
                'bbox': (x1, y1, x2, y2),
                'area': area,
                'center': (vehicle_center_x, (y1 + y2) // 2),
                'class_name': class_name
            })
        
        # Update tracked vehicles
        new_tracked_vehicles = {}
        for vid, vdata in tracked_vehicles.items():
            best_match = min(current_vehicles, key=lambda v: np.hypot(v['center'][0] - vdata['center'][0], v['center'][1] - vdata['center'][1]), default=None)
            if best_match and np.hypot(best_match['center'][0] - vdata['center'][0], best_match['center'][1] - vdata['center'][1]) < TRACKING_MATCH_DISTANCE:
                frames = vdata['frames'] + 1
                prev_area = vdata['area']
                new_area = best_match['area']
                
                # Calculate area change percentage
                if prev_area > 0:
                    area_change = (new_area - prev_area) / prev_area
                else:
                    area_change = 0
                
                # Determine motion status
                if frames >= MIN_FRAMES_FOR_STATIONARY:
                    if area_change > AREA_CHANGE_THRESHOLD or abs(area_change) <= AREA_CHANGE_THRESHOLD:
                        stationary = True  # Area increasing (approaching) or minimal change (not moving)
                    else:  # area_change < -AREA_CHANGE_THRESHOLD
                        stationary = False  # Area decreasing (moving away)
                else:
                    stationary = vdata['stationary']
                
                max_area = max(vdata['max_area'], new_area)
                
                # Expand bounding box
                x1, y1, x2, y2 = best_match['bbox']
                expand_factor = 0.1
                width_exp = int((x2 - x1) * expand_factor)
                height_exp = int((y2 - y1) * expand_factor)
                x1 = max(0, x1 - width_exp)
                y1 = max(0, y1 - height_exp)
                x2 = min(width, x2 + width_exp)
                y2 = min(height, y2 + height_exp)
                
                best_frame = vdata['best_frame'] if vdata['max_area'] > new_area else frame.copy()
                
                new_tracked_vehicles[vid] = {
                    'bbox': (x1, y1, x2, y2),
                    'center': best_match['center'],
                    'area': new_area,
                    'frames': frames,
                    'max_area': max_area,
                    'best_frame': best_frame,
                    'stationary': stationary,
                    'plate': vdata.get('plate', None),
                    'timestamp': vdata['timestamp'],
                    'class_name': vdata['class_name']
                }
        
        # Add new vehicles
        for vehicle in current_vehicles:
            if not any(np.hypot(vehicle['center'][0] - v['center'][0], vehicle['center'][1] - v['center'][1]) < TRACKING_MATCH_DISTANCE for v in new_tracked_vehicles.values()):
                vid = len(new_tracked_vehicles)
                
                x1, y1, x2, y2 = vehicle['bbox']
                expand_factor = 0.1
                width_exp = int((x2 - x1) * expand_factor)
                height_exp = int((y2 - y1) * expand_factor)
                x1 = max(0, x1 - width_exp)
                y1 = max(0, y1 - height_exp)
                x2 = min(width, x2 + width_exp)
                y2 = min(height, y2 + height_exp)
                
                new_tracked_vehicles[vid] = {
                    'bbox': (x1, y1, x2, y2),
                    'center': vehicle['center'],
                    'area': vehicle['area'],
                    'frames': 1,
                    'max_area': vehicle['area'],
                    'best_frame': frame.copy(),
                    'stationary': False,
                    'plate': None,
                    'timestamp': timestamp,
                    'class_name': vehicle['class_name']
                }
        
        tracked_vehicles = new_tracked_vehicles
        
        # Process cars for license plate detection
        for vid, vdata in list(tracked_vehicles.items()):
            if vdata['stationary'] and vdata['class_name'] == 'car' and not vdata['plate']:
                # Double-check the vehicle is on the left side of the yellow line
                x1, y1, x2, y2 = vdata['bbox']
                vehicle_center_x = (x1 + x2) // 2
                if vehicle_center_x > yellow_line_x:
                    continue  # Skip vehicles on the right side of the yellow line
                
                best_frame = vdata['best_frame']
                plate_results = plate_model(best_frame, conf=PLATE_CONFIDENCE_THRESHOLD)
                for result in plate_results:
                    for box in result.boxes:
                        px1, py1, px2, py2 = map(int, box.xyxy[0])
                        label = paddle_ocr(best_frame, px1, py1, px2, py2)
                        if label:
                            vdata['plate'] = label
                            license_plates.add(label)
                            print(f"Detected License Plate for ID {vid}: {label}")
                            # Save the entire frame locally
                            frame_path = os.path.join(OUTPUT_DIR, f"frame_{frame_count}_vehicle_{vid}_plate_{label}.jpg")
                            cv2.imwrite(frame_path, display_frame)
                            print(f"Saved frame with valid plate: {frame_path}")
                            # Upload the frame to Supabase storage and log to database
                            image_url = upload_image(display_frame, timestamp, vid, label)
                            if image_url:
                                log_violation(label, image_url, timestamp)
        
        # Draw vehicles and plates
        for vid, vdata in tracked_vehicles.items():
            x1, y1, x2, y2 = vdata['bbox']
            color = (0, 255, 0) if vdata['stationary'] else (0, 0, 255)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 5)
            label = f"ID: {vid} ({vdata['class_name']}) {'Stationary' if vdata['stationary'] else 'Not Stationary'}"
            if vdata['plate']:
                label += f" Plate: {vdata['plate']}"
            cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        
        if out:
            out.write(display_frame)

except Exception as e:
    print(f"Error in main loop: {e}")
    traceback.print_exc()
finally:
    if out:
        out.release()
    cap.release()
    save_json(license_plates)
    print("Processing complete.")