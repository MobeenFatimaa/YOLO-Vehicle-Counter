
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime
import csv
import cv2
from ultralytics import YOLO

# ================= CONFIGURATION =================

BASE_DIR = Path(__file__).resolve().parent.parent

VIDEO_PATH = BASE_DIR / "videos" / "input7.mp4"
OUTPUT_PATH = BASE_DIR / "output" / "counted_video.mp4"
CSV_PATH = BASE_DIR / "output" / "vehicle_report.csv"
EVENTS_PATH = BASE_DIR / "output" / "vehicle_events.csv"

MODEL_NAME = "yolo11n.pt"

CONFIDENCE = 0.35
IMAGE_SIZE = 640

LINE_Y = 0.75
ROAD_SPLIT_X = 0.50

VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}
PREVIEW_WIDTH = 1000
PREVIEW_HEIGHT = 650
# =================================================

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print("Loading YOLO model...")
model = YOLO(MODEL_NAME)

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30.0

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Could not create output video.")

line_y = int(height * LINE_Y)
split_x = int(width * ROAD_SPLIT_X)

# ================= TRACKING DATA =================

position_history = defaultdict(lambda: deque(maxlen=8))
counted_ids = set()

incoming_counts = defaultdict(int)
outgoing_counts = defaultdict(int)

incoming_total = 0
outgoing_total = 0

# Records for Phase 9
vehicle_events = []

frame_number = 0

# ================= DASHBOARD FUNCTION =================

def draw_dashboard(frame, incoming_counts, outgoing_counts, frame_number):
    """Draw a professional vehicle analytics dashboard."""

    incoming_total = sum(incoming_counts.values())
    outgoing_total = sum(outgoing_counts.values())
    total = incoming_total + outgoing_total

    overlay = frame.copy()

    panel_width = min(390, frame.shape[1] - 20)
    panel_height = min(320, frame.shape[0] - 20)

    cv2.rectangle(
        overlay,
        (10, 10),
        (10 + panel_width, 10 + panel_height),
        (20, 20, 20),
        -1
    )

    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    def put_text(text, x, y, scale=0.65, color=(255, 255, 255)):
        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            2,
            cv2.LINE_AA
        )

    put_text("VEHICLE ANALYTICS", 25, 40, 0.75)
    put_text(f"Incoming: {incoming_total}", 25, 78, 0.68, (0, 255, 0))
    put_text(f"Outgoing: {outgoing_total}", 25, 113, 0.68, (0, 165, 255))
    put_text(f"Total: {total}", 25, 148, 0.68)

    put_text("VEHICLE BREAKDOWN", 25, 185, 0.55, (255, 255, 0))

    y = 215

    for vehicle in ["Car", "Motorcycle", "Bus", "Truck"]:
        incoming = incoming_counts.get(vehicle, 0)
        outgoing = outgoing_counts.get(vehicle, 0)

        put_text(
            f"{vehicle}: IN {incoming} | OUT {outgoing}",
            25,
            y,
            0.48,
            (220, 220, 220)
        )

        y += 25

    return frame


# ================= MAIN PROCESSING LOOP =================

print("Phase 8: Live dashboard enabled.")
print("Phase 9: Vehicle event logging enabled.")
print("Press Q in the video window to stop processing.")

try:
    while True:
        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            classes=list(VEHICLE_CLASSES.keys()),
            verbose=False
        )

        # Draw counting lines and road divider
        cv2.line(
            frame,
            (0, line_y),
            (split_x, line_y),
            (0, 255, 0),
            3
        )

        cv2.line(
            frame,
            (split_x, line_y),
            (width - 1, line_y),
            (0, 165, 255),
            3
        )

        cv2.line(
            frame,
            (split_x, 0),
            (split_x, height - 1),
            (255, 255, 255),
            2
        )

        # ================= DETECTION =================

        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, class_id, track_id in zip(
                boxes, class_ids, track_ids
            ):
                x1, y1, x2, y2 = map(int, box)

                vehicle_name = VEHICLE_CLASSES.get(class_id, "Vehicle")

                center_x = (x1 + x2) // 2
                bottom_y = y2

                history = position_history[track_id]

                previous_y = history[-1][1] if history else None

                history.append((center_x, bottom_y))

                direction = "Tracking"

                # Estimate vertical movement
                if len(history) >= 3:
                    movement = history[-1][1] - history[-3][1]

                    if movement > 5:
                        direction = "Down"
                    elif movement < -5:
                        direction = "Up"

                # Detect crossing of counting line
                if previous_y is not None and track_id not in counted_ids:

                    crossed_line = (
                        previous_y < line_y <= bottom_y
                        or previous_y > line_y >= bottom_y
                    )

                    if crossed_line and bottom_y != previous_y:

                        if bottom_y > previous_y:
                            event_direction = "INCOMING"

                            incoming_total += 1
                            incoming_counts[vehicle_name] += 1

                        else:
                            event_direction = "OUTGOING"

                            outgoing_total += 1
                            outgoing_counts[vehicle_name] += 1

                        counted_ids.add(track_id)
                        direction = event_direction

                        # Record event timestamp and vehicle details
                        video_seconds = frame_number / fps

                        event = {
                            "Vehicle_ID": track_id,
                            "Vehicle_Type": vehicle_name,
                            "Direction": event_direction,
                            "Frame_Number": frame_number,
                            "Video_Time": round(video_seconds, 2),
                            "Timestamp": datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            )
                        }

                        vehicle_events.append(event)

                # Detection colors
                if direction == "INCOMING":
                    color = (0, 255, 0)
                elif direction == "OUTGOING":
                    color = (0, 165, 255)
                else:
                    color = (255, 255, 0)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                label = f"{vehicle_name} ID:{track_id} {direction}"

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

        # ================= LIVE DASHBOARD =================

        frame = draw_dashboard(
            frame,
            incoming_counts,
            outgoing_counts,
            frame_number
        )

        # Show processing progress
        progress = (
            frame_number / total_frames * 100
            if total_frames > 0 else 0
        )

        cv2.putText(
            frame,
            f"Progress: {progress:.1f}%",
            (15, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # Save and display frame
# Save the original, full-resolution frame without cropping

        # Save the original, full-resolution frame without cropping
        writer.write(frame)

        # Create a smaller preview while preserving the complete video
        preview_scale = min(
            PREVIEW_WIDTH / width,
            PREVIEW_HEIGHT / height,
            1.0
        )

        preview_width = int(width * preview_scale)
        preview_height = int(height * preview_scale)

        preview_frame = cv2.resize(
            frame,
            (preview_width, preview_height),
            interpolation=cv2.INTER_AREA
        )

        cv2.imshow("YOLO Vehicle Counter - Full Video", preview_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Processing stopped by user.")
            break


finally:
    cap.release()
    writer.release()
    cv2.destroyAllWindows()


# ================= PHASE 9: EVENT REPORT =================

event_columns = [
    "Vehicle_ID",
    "Vehicle_Type",
    "Direction",
    "Frame_Number",
    "Video_Time",
    "Timestamp"
]

with open(EVENTS_PATH, "w", newline="", encoding="utf-8") as file:
    csv_writer = csv.DictWriter(file, fieldnames=event_columns)
    csv_writer.writeheader()
    csv_writer.writerows(vehicle_events)


# ================= PHASE 8: SUMMARY REPORT =================

all_types = sorted(
    set(incoming_counts.keys()) |
    set(outgoing_counts.keys()) |
    set(VEHICLE_CLASSES.values())
)

with open(CSV_PATH, "w", newline="", encoding="utf-8") as file:
    csv_writer = csv.writer(file)

    csv_writer.writerow([
        "Vehicle_Type",
        "Incoming",
        "Outgoing",
        "Total"
    ])

    for vehicle_type in all_types:
        incoming = incoming_counts.get(vehicle_type, 0)
        outgoing = outgoing_counts.get(vehicle_type, 0)

        csv_writer.writerow([
            vehicle_type,
            incoming,
            outgoing,
            incoming + outgoing
        ])

    csv_writer.writerow([
        "ALL VEHICLES",
        incoming_total,
        outgoing_total,
        incoming_total + outgoing_total
    ])


# ================= FINAL RESULTS =================

print("\n========== FINAL VEHICLE REPORT ==========")
print(f"Incoming vehicles: {incoming_total}")
print(f"Outgoing vehicles: {outgoing_total}")
print(f"Total vehicles: {incoming_total + outgoing_total}")

print("\nIncoming by type:")
for name in all_types:
    print(f"{name}: {incoming_counts.get(name, 0)}")

print("\nOutgoing by type:")
for name in all_types:
    print(f"{name}: {outgoing_counts.get(name, 0)}")

print(f"\nProcessed video: {OUTPUT_PATH}")
print(f"Summary CSV: {CSV_PATH}")
print(f"Vehicle events CSV: {EVENTS_PATH}")
print("Phases 8 and 9 completed!")
