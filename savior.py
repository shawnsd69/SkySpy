# ## Import All the Required Libraries
# # import json
# # import cv2
# # from ultralytics import YOLO
# # import numpy as np
# # import math
# # import re
# # import os
# # from datetime import datetime
# # from paddleocr import PaddleOCR
# # import traceback

# # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# # # List of valid Indian state codes
# # STATE_CODES = [
# #     "AN", "AP", "AR", "AS", "BR", "CH", "DN", "DD", "DL", "GA", "GJ", "HR", "HP",
# #     "JK", "KA", "KL", "LD", "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "PY", "PN",
# #     "RJ", "SK", "TN", "TR", "UK", "UP", "WB"
# # ]

# # # Create a Video Capture Object
# # cap = cv2.VideoCapture("E:/ANPR_YOLOv10/Resources/carLicence5.mp4")
# # if not cap.isOpened():
# #     print("Error: Could not open video file. Verify the path: E:/ANPR_YOLOv10/Resources/carLicence5.mp4")
# #     exit()

# # # Initialize the YOLOv8 Model
# # try:
# #     model = YOLO("weights/best.pt")
# #     print("YOLOv8 model loaded successfully")
# # except Exception as e:
# #     print(f"Error loading YOLO model: {e}")
# #     exit()

# # # Initialize the frame count
# # count = 0

# # # Class Names
# # className = ["License"]

# # # Initialize the Paddle OCR
# # try:
# #     ocr = PaddleOCR(use_angle_cls=True, use_gpu=False)
# #     print("PaddleOCR initialized successfully")
# # except Exception as e:
# #     print(f"Error initializing PaddleOCR: {e}")
# #     exit()

# # def paddle_ocr(frame, x1, y1, x2, y2):
# #     try:
# #         frame = frame[y1:y2, x1:x2]
# #         result = ocr.ocr(frame, det=False, rec=True, cls=False)
# #         text = ""
# #         for r in result:
# #             scores = r[0][1]
# #             if np.isnan(scores):
# #                 scores = 0
# #             else:
# #                 scores = int(scores * 100)
# #             if scores > 60:
# #                 text = r[0][0]
# #         # Remove non-ASCII characters (e.g., Unicode characters like \u5dde)
# #         text = ''.join(char for char in text if ord(char) < 128)
# #         # Remove non-alphanumeric characters
# #         pattern = re.compile(r'[^A-Z0-9]')
# #         text = pattern.sub('', text)
# #         text = text.replace("???", "")
# #         text = text.replace("O", "0")
# #         text = text.replace("粤", "")

# #         # Post-processing: Validate state code (first two characters)
# #         if len(text) >= 2:
# #             state_code = text[:2].upper()
# #             if state_code not in STATE_CODES:
# #                 print(f"Invalid state code detected: {state_code} in {text}")
# #                 return ""  # Return empty string to reject invalid state codes
# #         else:
# #             print(f"Text too short to validate state code: {text}")
# #             return ""  # Return empty string to reject short text

# #         return str(text)
# #     except Exception as e:
# #         print(f"Error in paddle_ocr: {e}")
# #         return ""

# # def save_json(license_plates):
# #     try:
# #         # Save all license plates to a single JSON file
# #         data = {
# #             "Timestamp": datetime.now().isoformat(),
# #             "License Plates": list(license_plates)
# #         }
# #         os.makedirs("json", exist_ok=True)  # Ensure json folder exists
# #         with open("json/license_plates.json", 'w') as f:
# #             json.dump(data, f, indent=2)
# #         print("License plates saved to json/license_plates.json")
# #     except Exception as e:
# #         print(f"Error saving JSON: {e}")

# # license_plates = set()

# # try:
# #     while True:
# #         ret, frame = cap.read()
# #         if ret:
# #             count += 1
# #             print(f"Frame Number: {count}")
# #             results = model.predict(frame, conf=0.45)
# #             for result in results:
# #                 boxes = result.boxes
# #                 for box in boxes:
# #                     x1, y1, x2, y2 = box.xyxy[0]
# #                     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
# #                     cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
# #                     classNameInt = int(box.cls[0])
# #                     clsName = className[classNameInt]
# #                     conf = math.ceil(box.conf[0]*100)/100
# #                     label = paddle_ocr(frame, x1, y1, x2, y2)
# #                     if label:
# #                         license_plates.add(label)
# #                         print(f"Detected License Plate: {label}")
# #                     textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
# #                     c2 = x1 + textSize[0], y1 - textSize[1] - 3
# #                     cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
# #                     cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255,255,255], thickness=1, lineType=cv2.LINE_AA)
# #             cv2.imshow("Video", frame)
# #             if cv2.waitKey(1) & 0xFF == ord('1'):
# #                 save_json(license_plates)
# #                 break
# #         else:
# #             print("End of video or error reading frame")
# #             save_json(license_plates)
# #             break
# # except Exception as e:
# #     print(f"Error in main loop: {e}")
# #     traceback.print_exc()
# # finally:
# #     save_json(license_plates)  # Save any detected plates on exit
# #     cap.release()
# #     cv2.destroyAllWindows()
# #     print("Video capture released and windows closed")














#     # Import All the Required Libraries
# import json
# import cv2
# from ultralytics import YOLO
# import numpy as np
# import math
# import re
# import os
# from datetime import datetime
# from paddleocr import PaddleOCR
# import traceback

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# # List of valid Indian state codes
# STATE_CODES = [
#     "AN", "AP", "AR", "AS", "BR", "CH", "DN", "DD", "DL", "GA", "GJ", "HR", "HP",
#     "JK", "KA", "KL", "LD", "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "PY", "PN",
#     "RJ", "SK", "TN", "TR", "UK", "UP", "WB"
# ]

# # Create a Video Capture Object
# cap = cv2.VideoCapture("E:/ANPR_YOLOv10/Resources/carLicence7.mp4")
# if not cap.isOpened():
#     print("Error: Could not open video file. Verify the path: E:/ANPR_YOLOv10/Resources/carLicence5.mp4")
#     exit()

# # Initialize the YOLOv8 Model
# try:
#     model = YOLO("weights/best.pt")
#     print("YOLOv8 model loaded successfully")
# except Exception as e:
#     print(f"Error loading YOLO model: {e}")
#     exit()

# # Initialize the frame count
# count = 0

# # Class Names
# className = ["License"]

# # Initialize the Paddle OCR
# try:
#     ocr = PaddleOCR(use_angle_cls=True, use_gpu=False)
#     print("PaddleOCR initialized successfully")
# except Exception as e:
#     print(f"Error initializing PaddleOCR: {e}")
#     exit()

# def paddle_ocr(frame, x1, y1, x2, y2):
#     try:
#         frame = frame[y1:y2, x1:x2]
#         result = ocr.ocr(frame, det=False, rec=True, cls=False)
#         text = ""
#         for r in result:
#             scores = r[0][1]
#             if np.isnan(scores):
#                 scores = 0
#             else:
#                 scores = int(scores * 100)
#             if scores > 60:
#                 text = r[0][0]
#         # Remove non-ASCII characters (e.g., Unicode characters like \u5dde)
#         text = ''.join(char for char in text if ord(char) < 128)
#         # Remove non-alphanumeric characters
#         pattern = re.compile(r'[^A-Z0-9]')
#         text = pattern.sub('', text)
#         text = text.replace("???", "")
#         text = text.replace("O", "0")
#         text = text.replace("粤", "")

#         # Post-processing: Validate state code (first two characters)
#         if len(text) >= 2:
#             state_code = text[:2].upper()
#             if state_code not in STATE_CODES:
#                 print(f"Invalid state code detected: {state_code} in {text}")
#                 return ""  # Return empty string to reject invalid state codes
#         else:
#             print(f"Text too short to validate state code: {text}")
#             return ""  # Return empty string to reject short text

#         return str(text)
#     except Exception as e:
#         print(f"Error in paddle_ocr: {e}")
#         return ""

# def save_json(license_plates):
#     try:
#         # Save all license plates to a single JSON file
#         data = {
#             "Timestamp": datetime.now().isoformat(),
#             "License Plates": list(license_plates)
#         }
#         os.makedirs("json", exist_ok=True)  # Ensure json folder exists
#         with open("json/license_plates.json", 'w') as f:
#             json.dump(data, f, indent=2)
#         print("License plates saved to json/license_plates.json")
#     except Exception as e:
#         print(f"Error saving JSON: {e}")

# license_plates = set()

# try:
#     while True:
#         ret, frame = cap.read()
#         if ret:
#             count += 1
#             print(f"Frame Number: {count}")
#             results = model.predict(frame, conf=0.45)
#             for result in results:
#                 boxes = result.boxes
#                 for box in boxes:
#                     x1, y1, x2, y2 = box.xyxy[0]
#                     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                     # Draw bounding box
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#                     # Extract license plate text
#                     label = paddle_ocr(frame, x1, y1, x2, y2)
#                     # Add to set if label is non-empty
#                     if label:
#                         license_plates.add(label)
#                         print(f"Detected License Plate: {label}")
#                     # Display the label (even if empty, this won't cause issues)
#                     textSize = cv2.getTextSize(label if label else "Unknown", 0, fontScale=0.5, thickness=2)[0]
#                     c2 = x1 + textSize[0], y1 - textSize[1] - 3
#                     cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
#                     cv2.putText(frame, label if label else "Unknown", (x1, y1 - 2), 0, 0.5, [255,255,255], thickness=1, lineType=cv2.LINE_AA)
#             cv2.imshow("Video", frame)
#             if cv2.waitKey(1) & 0xFF == ord('1'):
#                 save_json(license_plates)
#                 break
#         else:
#             print("End of video or error reading frame")
#             save_json(license_plates)
#             break
# except Exception as e:
#     print(f"Error in main loop: {e}")
#     traceback.print_exc()
# finally:
#     save_json(license_plates)  # Save any detected plates on exit
#     cap.release()
#     cv2.destroyAllWindows()
#     print("Video capture released and windows closed")




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

# Video path
video_path = r"E:\ANPR_YOLOv10\Resources\carLicence6.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Video not loaded.")
    exit()

# Video writer setup
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video dimensions: {frame_width}x{frame_height}")
if frame_width <= 0 or frame_height <= 0:
    print("Error: Invalid video dimensions.")
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

# Process video
frame_count = 0
tracked_vehicles = {}
last_yellow_line_x = None
license_plates = set()

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
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
                            # Save the entire frame
                            frame_path = os.path.join(OUTPUT_DIR, f"frame_{frame_count}_vehicle_{vid}_plate_{label}.jpg")
                            cv2.imwrite(frame_path, display_frame)
                            print(f"Saved frame with valid plate: {frame_path}")
        
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