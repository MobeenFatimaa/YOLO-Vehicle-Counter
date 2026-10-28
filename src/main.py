from pathlib import Path
from collections import defaultdict, deque
import cv2
from ultralytics import YOLO

# ================= CONFIGURATION =================

BASE_DIR = Path(__file__).resolve().parent.parent

VIDEO_PATH = BASE_DIR / "videos" / "input7.mp4"
OUTPUT_PATH = BASE_DIR / "output" / "counted_video.mp4"

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

# =================================================

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

model = YOLO(MODEL_NAME)

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

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

# Track recent vehicle positions
position_history = defaultdict(lambda: deque(maxlen=8))

counted_ids = set()

incoming_counts = defaultdict(int)
outgoing_counts = defaultdict(int)

incoming_total = 0
outgoing_total = 0

print("Phase 6: Direction-aware vehicle counting started.")

while True:
    success, frame = cap.read()

    if not success:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        classes=list(VEHICLE_CLASSES.keys()),
        verbose=False
    )

    # Counting lines
    cv2.line(
        frame, (0, line_y), (split_x, line_y),
        (0, 255, 0), 3
    )

    cv2.line(
        frame, (split_x, line_y), (width, line_y),
        (255, 165, 0), 3
    )

    cv2.line(
        frame, (split_x, 0), (split_x, height),
        (255, 255, 255), 2
    )

    if results and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, class_id, track_id in zip(
            boxes, class_ids, track_ids
        ):
            x1, y1, x2, y2 = map(int, box)

            vehicle_name = VEHICLE_CLASSES.get(
                class_id, "Vehicle"
            )

            center_x = (x1 + x2) // 2
            bottom_y = y2

            history = position_history[track_id]
            previous_y = history[-1][1] if history else None

            history.append((center_x, bottom_y))

            direction = "Tracking"

            # Determine direction from vertical movement.
            if len(history) >= 3:
                movement = history[-1][1] - history[-3][1]

                if movement > 5:
                    direction = "Down"
                elif movement < -5:
                    direction = "Up"

            # Determine which road side the vehicle occupies.
            is_left_side = center_x < split_x

            # Count only when the vehicle crosses the line.
            if previous_y is not None and track_id not in counted_ids:
                crossed_line = (
                    previous_y < line_y <= bottom_y
                    or previous_y > line_y >= bottom_y
                )

                if crossed_line:
                    # Movement down the image: incoming.
                    # Movement up the image: outgoing.
                    if bottom_y > previous_y:
                        incoming_total += 1
                        incoming_counts[vehicle_name] += 1
                        direction = "INCOMING"
                        counted_ids.add(track_id)

                    elif bottom_y < previous_y:
                        outgoing_total += 1
                        outgoing_counts[vehicle_name] += 1
                        direction = "OUTGOING"
                        counted_ids.add(track_id)

            # Draw vehicle detection
            color = (
                (0, 255, 0)
                if direction == "INCOMING"
                else (0, 165, 255)
                if direction == "OUTGOING"
                else (255, 255, 0)
            )

            cv2.rectangle(
                frame, (x1, y1), (x2, y2), color, 2
            )

            label = f"{vehicle_name} ID:{track_id} {direction}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

    # Dashboard
    cv2.rectangle(frame, (10, 10), (360, 145), (25, 25, 25), -1)

    cv2.putText(
        frame,
        f"Incoming: {incoming_total}",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Outgoing: {outgoing_total}",
        (25, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 165, 255),
        2
    )

    cv2.putText(
        frame,
        f"Total: {incoming_total + outgoing_total}",
        (25, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    writer.write(frame)
    cv2.imshow("YOLO Vehicle Counter - Phase 6", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break    

cap.release()
writer.release()
cv2.destroyAllWindows()

print("\n========== FINAL RESULTS ==========")
print(f"Incoming vehicles: {incoming_total}")
print(f"Outgoing vehicles: {outgoing_total}")
print(f"Total vehicles: {incoming_total + outgoing_total}")

print("\nIncoming by type:")
for name, count in incoming_counts.items():
    print(f"{name}: {count}")

print("\nOutgoing by type:")
for name, count in outgoing_counts.items():
    print(f"{name}: {count}")

print(f"\nOutput saved to: {OUTPUT_PATH}")
print("Phase 6 completed!")
