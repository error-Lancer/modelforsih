import os
import re
import cv2
import numpy as np
from collections import Counter
from ultralytics import YOLO
import easyocr

# =============================================================
# 1. CONFIGURATION
# =============================================================
VIDEO_SOURCE = r"C:\Users\sandh\PycharmProjects\mlcore\open model\dataset\WhatsApp Video 2026-09-12 at 4.35.30 PM.mp4"   # Use 0 for webcam or path to .mp4
TARGET_PLATE = "KA02MM9091"                     # e.g., "DL01AB1234" (leave empty to view all)

WEIGHTS_PATH = r"C:\Users\sandh\PycharmProjects\mlcore\open model\notebook\runs\detect\runs\detect\plate_detector-4\weights\best.pt"

CONF_VEHICLE = 0.25
CONF_PLATE   = 0.25
OCR_INTERVAL = 3    # Run OCR every 3rd frame to keep FPS high

# =============================================================
# 2. COLOR EXTRACTION MODULE (HSV Space)
# =============================================================
# HSV definitions: Hue (0-180), Saturation (0-255), Value (0-255)
COLOR_RANGES = {
    "Red": [
        ((0, 70, 50), (10, 255, 255)),
        ((170, 70, 50), (180, 255, 255))
    ],
    "Yellow": [((20, 100, 100), (35, 255, 255))],
    "Green":  [((36, 50, 50), (85, 255, 255))],
    "Blue":   [((90, 50, 50), (130, 255, 255))],
    "White":  [((0, 0, 180), (180, 45, 255))],
    "Black":  [((0, 0, 0), (180, 255, 45))],
    "Silver/Grey": [((0, 0, 45), (180, 45, 180))]
}

def extract_vehicle_color(v_crop):
    """
    Extracts dominant paint color by sampling the center 40% area of the vehicle.
    Avoids tires, road surface, and windshield glass.
    """
    h, w = v_crop.shape[:2]
    if h < 30 or w < 30:
        return "Unknown"

    # Crop the central region of the vehicle (door panels / bonnet)
    y1, y2 = int(h * 0.35), int(h * 0.75)
    x1, x2 = int(w * 0.20), int(w * 0.80)
    body_patch = v_crop[y1:y2, x1:x2]

    if body_patch.size == 0:
        return "Unknown"

    hsv = cv2.cvtColor(body_patch, cv2.COLOR_BGR2HSV)
    max_pts = 0
    best_color = "Unknown"

    for c_name, bounds in COLOR_RANGES.items():
        total_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (low, high) in bounds:
            mask = cv2.inRange(hsv, np.array(low), np.array(high))
            total_mask = cv2.bitwise_or(total_mask, mask)

        pts = cv2.countNonZero(total_mask)
        if pts > max_pts:
            max_pts = pts
            best_color = c_name

    return best_color

# =============================================================
# 3. HELPER FUNCTIONS (OCR & PREPROCESSING)
# =============================================================
def clean_text(raw):
    return re.sub(r'[^A-Z0-9]', '', raw.upper()) if raw else ""

def enhance_plate(crop_bgr):
    h, w = crop_bgr.shape[:2]
    if h == 0 or w == 0:
        return None
    # Upscale height to 90px so strokes are clean for OCR
    target_h = 90
    scale = target_h / float(h)
    target_w = max(int(w * scale), 120)
    resized = cv2.resize(crop_bgr, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

# =============================================================
# 4. INITIALIZE MODELS & MEMORY BUFFERS
# =============================================================
print("[1/3] Loading YOLO Vehicle Tracker (COCO: Cars, Bikes, Buses, Trucks)...")
vehicle_model = YOLO("yolov8n.pt")

print("[2/3] Loading Your Custom Plate Model...")
plate_model = YOLO(WEIGHTS_PATH)

print("[3/3] Initializing EasyOCR on CUDA...")
ocr_reader = easyocr.Reader(['en'], gpu=True)

target_clean = clean_text(TARGET_PLATE)

# Tracking state buffers
vehicle_colors = {}  # vid -> "White", "Red", etc.
plate_history  = {}  # vid -> [reads...]
stable_plates  = {}  # vid -> consensus text

# =============================================================
# 5. VIDEO PROCESSING LOOP
# =============================================================
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"[ERROR] Could not open video file: '{VIDEO_SOURCE}'")
    exit()

window_name = "Real-Time ANPR + Vehicle Color Tracker"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)

frame_count = 0
print(f"\n[STREAMING] Active: {VIDEO_SOURCE}")
print(f"[SEARCHING] Target: '{target_clean or 'ALL VEHICLES'}'")
print("Press 'q' in the playback window to stop.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[INFO] Video playback completed.")
        break

    frame_count += 1
    h_img, w_img = frame.shape[:2]

    # --- Step A: Vehicle Detection & Multi-Object Tracking ---
    # COCO Classes: 2: car, 3: motorcycle, 5: bus, 7: truck
    v_results = vehicle_model.track(
        frame,
        classes=[2, 3, 5, 7],
        tracker="bytetrack.yaml",
        persist=True,
        conf=CONF_VEHICLE,
        device=0,
        verbose=False
    )[0]

    tracked_vehicles = []
    if v_results.boxes and v_results.boxes.id is not None:
        boxes = v_results.boxes.xyxy.cpu().numpy().astype(int)
        tids = v_results.boxes.id.cpu().numpy().astype(int)
        for b, tid in zip(boxes, tids):
            tracked_vehicles.append({"id": int(tid), "box": b})

    # --- Step B: Determine Color for Each Vehicle ---
    for veh in tracked_vehicles:
        vid = veh["id"]
        vx1, vy1, vx2, vy2 = veh["box"]
        # Extract once per vehicle ID to save processing power
        if vid not in vehicle_colors:
            v_crop = frame[max(0, vy1):min(h_img, vy2), max(0, vx1):min(w_img, vx2)]
            vehicle_colors[vid] = extract_vehicle_color(v_crop)

    # --- Step C: Detect Plates & Run Periodic OCR ---
    p_results = plate_model.predict(
        frame,
        conf=CONF_PLATE,
        device=0,
        verbose=False
    )[0]

    should_run_ocr = (frame_count % OCR_INTERVAL == 0)

    for pbox in p_results.boxes:
        px1, py1, px2, py2 = map(int, pbox.xyxy[0].cpu().numpy())
        p_cx, p_cy = (px1 + px2) // 2, (py1 + py2) // 2

        # Associate plate with vehicle via centroid containment
        parent_id = None
        for veh in tracked_vehicles:
            vx1, vy1, vx2, vy2 = veh["box"]
            if (vx1 - 15) <= p_cx <= (vx2 + 15) and (vy1 - 15) <= p_cy <= (vy2 + 15):
                parent_id = veh["id"]
                break

        # Run OCR when triggered
        if should_run_ocr and parent_id is not None:
            pad_w = int((px2 - px1) * 0.05)
            pad_h = int((py2 - py1) * 0.05)
            crop_x1 = max(0, px1 - pad_w)
            crop_y1 = max(0, py1 - pad_h)
            crop_x2 = min(w_img - 1, px2 + pad_w)
            crop_y2 = min(h_img - 1, py2 + pad_h)

            crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            enhanced = enhance_plate(crop)
            if enhanced is not None:
                ocr_out = ocr_reader.readtext(
                    enhanced,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                    detail=1,
                    paragraph=False
                )
                raw_text = "".join([clean_text(item[1]) for item in ocr_out])
                if len(raw_text) >= 3:
                    if parent_id not in plate_history:
                        plate_history[parent_id] = []
                    plate_history[parent_id].append(raw_text)
                    stable_plates[parent_id] = Counter(plate_history[parent_id]).most_common(1)[0][0]

        # Draw Cyan Box around the Plate
        cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 255, 0), 2)

    # --- Step D: Render Overlays & Target Alerts ---
    for veh in tracked_vehicles:
        vx1, vy1, vx2, vy2 = veh["box"]
        vid = veh["id"]
        plate_str = stable_plates.get(vid, "")
        car_color = vehicle_colors.get(vid, "Vehicle")

        is_target = bool(target_clean and plate_str and target_clean in plate_str)

        if is_target:
            # TARGET ALERT: Red Box + Warning Banner
            cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 0, 255), 4)
            cv2.rectangle(frame, (vx1, max(0, vy1 - 45)), (vx1 + 420, vy1), (0, 0, 255), -1)
            cv2.putText(frame, f"TARGET MATCH: {plate_str}", (vx1 + 8, vy1 - 24),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Color: {car_color} | ID: {vid}", (vx1 + 8, vy1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        else:
            # NORMAL VEHICLE: Green Box + ID + Color + Plate
            label = f"ID:{vid} | {car_color}"
            if plate_str:
                label += f" | [{plate_str}]"

            cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 255, 0), 2)
            cv2.putText(frame, label, (vx1, max(20, vy1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # Top Status Bar
    cv2.rectangle(frame, (0, 0), (w_img, 36), (20, 20, 20), -1)
    hud = f"Frame: {frame_count} | Active: {len(tracked_vehicles)} | Target: {target_clean or 'ALL'}"
    cv2.putText(frame, hud, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[FINISHED] Process terminated cleanly.")