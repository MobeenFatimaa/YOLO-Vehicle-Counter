from pathlib import Path
from collections import defaultdict
from datetime import datetime
import csv
import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

VIDEO_PATH = BASE_DIR / "videos" / "input.mp4"
OUTPUT_PATH = BASE_DIR / "output" / "counted_video.mp4"
CSV_PATH = BASE_DIR / "output" / "vehicle_report.csv"
EVENTS_PATH = BASE_DIR / "output" / "vehicle_events.csv"

MODEL_NAME = "yolo11n.pt"

CONFIDENCE = 0.25
IMAGE_SIZE = 960
IOU_THRESHOLD = 0.5

# Horizontal counting line.
LINE_Y = 0.75

# Vertical divider between the two roads.
ROAD_SPLIT_X = 0.50

# Vehicles close to the divider are treated as ambiguous.
DIVIDER_MARGIN = 0.025

# Preview window dimensions.
PREVIEW_WIDTH = 1280
PREVIEW_HEIGHT = 800

# Allow a vehicle to disappear briefly between observations.
MAX_CROSSING_GAP = 30

VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}

# ------------------------------------------------------------
# SPEED ESTIMATION CONFIGURATION
# ------------------------------------------------------------
# Calibration factor: approximate meters per pixel for overhead highway view
METERS_PER_PIXEL = 0.05  

# OpenCV uses BGR color order.
BLUE = (255, 0, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
CYAN = (255, 255, 0)


# ============================================================
# 2. PREPARE OUTPUT FILES
# ============================================================

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Input video not found:\n{VIDEO_PATH}"
    )

print("=" * 60)
print("YOLO VEHICLE COUNTER & SPEED ESTIMATOR - HIGHWAY MODE")
print("=" * 60)

print("\nLoading YOLO model...")

model = YOLO(MODEL_NAME)

print("YOLO model loaded successfully.")


# ============================================================
# 3. OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video:\n{VIDEO_PATH}"
    )

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if fps <= 0:
    fps = 30.0

if width <= 0 or height <= 0:
    cap.release()
    raise RuntimeError("The video has invalid dimensions.")

line_y = int(height * LINE_Y)
split_x = int(width * ROAD_SPLIT_X)

writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError(
        f"Could not create output video:\n{OUTPUT_PATH}"
    )

print(f"Video resolution: {width} x {height}")
print(f"Video FPS: {fps:.2f}")
print(f"Total frames: {total_frames}")
print(f"Counting line Y: {line_y}")
print(f"Road divider X: {split_x}")


# ============================================================
# 4. TRACKING AND COUNTING VARIABLES
# ============================================================

last_line_side = {}
last_seen_frame = {}
vehicle_lanes = {}
counted_ids = set()
vehicle_directions = {}

# Speed tracking dictionaries
previous_positions = {}  # track_id -> (center_x, center_y, frame_number)
vehicle_speeds = {}      # track_id -> calculated speed in km/h

incoming_counts = defaultdict(int)
outgoing_counts = defaultdict(int)

incoming_total = 0
outgoing_total = 0

vehicle_events = []
frame_number = 0


# ============================================================
# 5. DASHBOARD
# ============================================================

def draw_dashboard(frame, incoming_counts, outgoing_counts):

    frame_h, frame_w = frame.shape[:2]

    scale = min(frame_w / 1000, frame_h / 600)

    panel_width = min(
        int(370 * scale),
        frame_w - 20,
    )

    panel_height = min(
        int(300 * scale),
        frame_h - 20,
    )

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (10 + panel_width, 10 + panel_height),
        (20, 20, 20),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.82,
        frame,
        0.18,
        0,
        frame,
    )

    def put_text(
        text,
        x,
        y,
        font_scale=0.65,
        color=WHITE,
        thickness=2,
    ):
        cv2.putText(
            frame,
            text,
            (int(x * scale), int(y * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * scale,
            color,
            max(1, int(thickness * scale)),
            cv2.LINE_AA,
        )

    incoming = sum(incoming_counts.values())
    outgoing = sum(outgoing_counts.values())

    put_text(
        "VEHICLE ANALYTICS",
        20,
        40,
        0.85,
    )

    put_text(
        f"Incoming: {incoming}",
        20,
        80,
        0.75,
        BLUE,
    )

    put_text(
        f"Outgoing: {outgoing}",
        20,
        118,
        0.75,
        RED,
    )

    put_text(
        f"Total Vehicles: {incoming + outgoing}",
        20,
        155,
        0.70,
    )

    put_text(
        "VEHICLE BREAKDOWN",
        20,
        192,
        0.60,
        CYAN,
    )

    y = 222

    for name in ["Car", "Motorcycle", "Bus", "Truck"]:

        count_in = incoming_counts.get(name, 0)
        count_out = outgoing_counts.get(name, 0)

        put_text(
            f"{name}: IN {count_in} | OUT {count_out}",
            20,
            y,
            0.53,
            (235, 235, 235),
            1,
        )

        y += 25

    return frame


# ============================================================
# 6. COUNT A VEHICLE
# ============================================================

def record_vehicle(
    track_id,
    vehicle_name,
    direction,
    frame_number,
):
    global incoming_total, outgoing_total

    if track_id in counted_ids:
        return

    counted_ids.add(track_id)
    vehicle_directions[track_id] = direction

    if direction == "INCOMING":

        incoming_counts[vehicle_name] += 1
        incoming_total += 1

    elif direction == "OUTGOING":

        outgoing_counts[vehicle_name] += 1
        outgoing_total += 1

    event = {
        "Vehicle_ID": int(track_id),
        "Vehicle_Type": vehicle_name,
        "Direction": direction,
        "Frame_Number": frame_number,
        "Video_Time_Seconds": round(
            frame_number / fps,
            2,
        ),
        "Timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    }

    vehicle_events.append(event)

    print(
        f"[COUNTED] ID={track_id} | "
        f"Type={vehicle_name} | "
        f"Direction={direction} | "
        f"Incoming={incoming_total} | "
        f"Outgoing={outgoing_total}"
    )


# ============================================================
# 7. MAIN PROCESSING LOOP
# ============================================================

print("\nStarting video processing with speed estimation...")
print("LEFT ROAD: Incoming vehicles (BLUE)")
print("RIGHT ROAD: Outgoing vehicles (RED)")
print("DIVIDER AREA: Yellow")
print("Incoming direction: DOWNWARD")
print("Outgoing direction: UPWARD")
print("Press Q to stop processing.\n")

try:

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # ----------------------------------------------------
        # DETECT AND TRACK VEHICLES
        # ----------------------------------------------------

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            classes=list(VEHICLE_CLASSES.keys()),
            verbose=False,
            iou=IOU_THRESHOLD,
        )

        # ----------------------------------------------------
        # DRAW COUNTING LINE AND ROAD DIVIDER
        # ----------------------------------------------------

        cv2.line(
            frame,
            (0, line_y),
            (split_x, line_y),
            GREEN,
            3,
        )

        cv2.line(
            frame,
            (split_x, line_y),
            (width - 1, line_y),
            (0, 165, 255),
            3,
        )

        cv2.line(
            frame,
            (split_x, 0),
            (split_x, height - 1),
            WHITE,
            2,
        )

        # ----------------------------------------------------
        # PROCESS TRACKED VEHICLES & SPEED ESTIMATION
        # ----------------------------------------------------

        active_track_ids = set()

        if (
            results
            and results[0].boxes is not None
            and results[0].boxes.id is not None
        ):

            boxes = (
                results[0]
                .boxes.xyxy
                .cpu()
                .numpy()
            )

            class_ids = (
                results[0]
                .boxes.cls
                .cpu()
                .numpy()
                .astype(int)
            )

            track_ids = (
                results[0]
                .boxes.id
                .cpu()
                .numpy()
                .astype(int)
            )

            for box, class_id, track_id in zip(
                boxes,
                class_ids,
                track_ids,
            ):

                active_track_ids.add(track_id)
                x1, y1, x2, y2 = map(int, box)

                vehicle_name = VEHICLE_CLASSES.get(
                    class_id,
                    "Vehicle",
                )

                center_x = (x1 + x2) // 2
                bottom_y = y2

                # ------------------------------------------------
                # SPEED ESTIMATION LOGIC
                # ------------------------------------------------
                if track_id in previous_positions:
                    prev_x, prev_y, prev_frame = previous_positions[track_id]
                    frame_diff = frame_number - prev_frame
                    
                    if frame_diff > 0:
                        # Euclidean distance in pixels
                        pixel_distance = np.sqrt((center_x - prev_x)**2 + (bottom_y - prev_y)**2)
                        # Distance in real-world meters
                        meter_distance = pixel_distance * METERS_PER_PIXEL
                        # Time elapsed in seconds
                        time_elapsed = frame_diff / fps
                        
                        if time_elapsed > 0:
                            speed_mps = meter_distance / time_elapsed
                            speed_kmh = speed_mps * 3.6  # convert meters per second to km/h
                            
                            # Smooth speed fluctuations
                            if track_id in vehicle_speeds:
                                vehicle_speeds[track_id] = int(0.7 * vehicle_speeds[track_id] + 0.3 * speed_kmh)
                            else:
                                vehicle_speeds[track_id] = int(speed_kmh)

                previous_positions[track_id] = (center_x, bottom_y, frame_number)
                vehicle_speed = vehicle_speeds.get(track_id, 0)

                # ------------------------------------------------
                # IDENTIFY THE VEHICLE'S LANE
                # ------------------------------------------------

                if track_id not in vehicle_lanes:

                    if center_x < split_x:
                        vehicle_lanes[track_id] = "LEFT"
                    else:
                        vehicle_lanes[track_id] = "RIGHT"

                lane = vehicle_lanes[track_id]

                # ------------------------------------------------
                # DETERMINE THE CURRENT SIDE OF THE LINE
                # ------------------------------------------------

                current_side = (
                    -1 if bottom_y < line_y else 1
                )

                previous_side = last_line_side.get(
                    track_id
                )

                previous_frame = last_seen_frame.get(
                    track_id
                )

                # ------------------------------------------------
                # COUNT A CROSSING
                # ------------------------------------------------

                if previous_side is not None:

                    frame_gap = (
                        frame_number - previous_frame
                        if previous_frame is not None
                        else MAX_CROSSING_GAP + 1
                    )

                    crossed_line = (
                        previous_side != current_side
                        and 0 < frame_gap <= MAX_CROSSING_GAP
                    )

                    if (
                        crossed_line
                        and track_id not in counted_ids
                    ):

                        event_direction = None

                        if (
                            previous_side == -1
                            and current_side == 1
                        ):
                            event_direction = "INCOMING"

                        elif (
                            previous_side == 1
                            and current_side == -1
                        ):
                            event_direction = "OUTGOING"

                        if event_direction is not None:

                            record_vehicle(
                                track_id=track_id,
                                vehicle_name=vehicle_name,
                                direction=event_direction,
                                frame_number=frame_number,
                            )

                last_line_side[track_id] = current_side
                last_seen_frame[track_id] = frame_number

                # ------------------------------------------------
                # ASSIGN BOX COLOR & SPEED LABEL
                # ------------------------------------------------

                if track_id in counted_ids:

                    direction = vehicle_directions[track_id]

                    if direction == "INCOMING":
                        color = BLUE
                    else:
                        color = RED

                    label = (
                        f"{vehicle_name} ID:{track_id} "
                        f"{direction} | {vehicle_speed} km/h"
                    )

                else:

                    divider_margin = int(
                        width * DIVIDER_MARGIN
                    )

                    if abs(center_x - split_x) <= divider_margin:

                        color = YELLOW
                        label = (
                            f"{vehicle_name} ID:{track_id} "
                            f"TRACKING | {vehicle_speed} km/h"
                        )

                    elif lane == "LEFT":

                        color = YELLOW

                        label = (
                            f"{vehicle_name} ID:{track_id} "
                            f"OUTGOING | {vehicle_speed} km/h"
                        )

                    else:

                        color = YELLOW

                        label = (
                            f"{vehicle_name} ID:{track_id} "
                            f"INCOMING | {vehicle_speed} km/h"
                        )

                # ------------------------------------------------
                # DRAW BOUNDING BOX
                # ------------------------------------------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2,
                )

                text_y = max(y1 - 10, 20)

                cv2.putText(
                    frame,
                    label,
                    (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        # Clean up old tracked positions for memory optimization
        for tid in list(previous_positions.keys()):
            if tid not in active_track_ids and frame_number - previous_positions[tid][2] > 60:
                del previous_positions[tid]

        # ----------------------------------------------------
        # DRAW PROGRESS
        # ----------------------------------------------------

        if total_frames > 0:

            progress = (
                frame_number / total_frames
            ) * 100

        else:

            progress = 0.0

        progress_text = f"Progress: {progress:.1f}%"

        cv2.putText(
            frame,
            progress_text,
            (15, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            WHITE,
            2,
            cv2.LINE_AA,
        )

        # ----------------------------------------------------
        # SAVE ANNOTATED VIDEO
        # ----------------------------------------------------

        output_frame = draw_dashboard(
            frame.copy(),
            incoming_counts,
            outgoing_counts,
        )

        writer.write(output_frame)

        # ----------------------------------------------------
        # SHOW LIVE PREVIEW
        # ----------------------------------------------------

        preview_scale = min(
            PREVIEW_WIDTH / width,
            PREVIEW_HEIGHT / height,
            1.0,
        )

        preview_width = max(
            1,
            int(width * preview_scale),
        )

        preview_height = max(
            1,
            int(height * preview_scale),
        )

        preview_frame = cv2.resize(
            frame,
            (preview_width, preview_height),
            interpolation=cv2.INTER_AREA,
        )

        preview_frame = draw_dashboard(
            preview_frame,
            incoming_counts,
            outgoing_counts,
        )

        preview_progress = (
            f"Progress: {progress:.1f}%"
        )

        cv2.putText(
            preview_frame,
            preview_progress,
            (15, preview_height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            WHITE,
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(
            "YOLO Vehicle Counter & Speed Estimator",
            preview_frame,
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):

            print("\nProcessing stopped by user.")
            break

finally:

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


# ============================================================
# 8. EXPORT VEHICLE EVENTS
# ============================================================

event_columns = [
    "Vehicle_ID",
    "Vehicle_Type",
    "Direction",
    "Frame_Number",
    "Video_Time_Seconds",
    "Timestamp",
]

with open(
    EVENTS_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as file:

    csv_writer = csv.DictWriter(
        file,
        fieldnames=event_columns,
    )

    csv_writer.writeheader()
    csv_writer.writerows(vehicle_events)


# ============================================================
# 9. EXPORT SUMMARY REPORT
# ============================================================

all_vehicle_types = [
    "Car",
    "Motorcycle",
    "Bus",
    "Truck",
]

with open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as file:

    csv_writer = csv.writer(file)

    csv_writer.writerow([
        "Vehicle_Type",
        "Incoming",
        "Outgoing",
        "Total",
    ])

    for vehicle_type in all_vehicle_types:

        count_in = incoming_counts.get(
            vehicle_type,
            0,
        )

        count_out = outgoing_counts.get(
            vehicle_type,
            0,
        )

        csv_writer.writerow([
            vehicle_type,
            count_in,
            count_out,
            count_in + count_out,
        ])

    csv_writer.writerow([
        "ALL VEHICLES",
        incoming_total,
        outgoing_total,
        incoming_total + outgoing_total,
    ])


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL VEHICLE REPORT")
print("=" * 60)

print(f"Incoming vehicles: {incoming_total}")
print(f"Outgoing vehicles: {outgoing_total}")
print(
    f"Total vehicles: "
    f"{incoming_total + outgoing_total}"
)

print("\nVehicle breakdown:")

for vehicle_type in all_vehicle_types:

    print(
        f"{vehicle_type}: "
        f"IN={incoming_counts.get(vehicle_type, 0)} | "
        f"OUT={outgoing_counts.get(vehicle_type, 0)}"
    )

print("\nGenerated files:")

print(f"Processed video: {OUTPUT_PATH}")
print(f"Summary report: {CSV_PATH}")
print(f"Events report: {EVENTS_PATH}")

print("\nProcessing completed.")
