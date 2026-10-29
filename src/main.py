from pathlib import Path
from collections import defaultdict
from datetime import datetime
import csv
import cv2
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
print("YOLO VEHICLE COUNTER - HIGHWAY MODE")
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

# Remember the previous side of the counting line for each ID.
# -1 = above the line
#  1 = on or below the line

last_line_side = {}

# Remember the frame in which the vehicle was last observed.
last_seen_frame = {}

# Remember the lane assigned when the vehicle was first seen.
vehicle_lanes = {}

# Prevent the same tracking ID from being counted repeatedly.
counted_ids = set()

# Remember the direction assigned to counted vehicles.
vehicle_directions = {}

# Counts by vehicle type.
incoming_counts = defaultdict(int)
outgoing_counts = defaultdict(int)

incoming_total = 0
outgoing_total = 0

# Event report.
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

    # This function should only be called for uncounted IDs.
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

print("\nStarting video processing...")
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

        # Left half of the horizontal counting line.
        cv2.line(
            frame,
            (0, line_y),
            (split_x, line_y),
            GREEN,
            3,
        )

        # Right half of the horizontal counting line.
        cv2.line(
            frame,
            (split_x, line_y),
            (width - 1, line_y),
            (0, 165, 255),
            3,
        )

        # Vertical divider between the traffic directions.
        cv2.line(
            frame,
            (split_x, 0),
            (split_x, height - 1),
            WHITE,
            2,
        )

        # ----------------------------------------------------
        # PROCESS TRACKED VEHICLES
        # ----------------------------------------------------

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

                x1, y1, x2, y2 = map(int, box)

                vehicle_name = VEHICLE_CLASSES.get(
                    class_id,
                    "Vehicle",
                )

                center_x = (x1 + x2) // 2
                bottom_y = y2

                # ------------------------------------------------
                # IDENTIFY THE VEHICLE'S LANE
                # ------------------------------------------------

                # Keep the initial lane assignment for this ID.
                # Do not let small bounding-box movements around
                # the divider repeatedly change the lane.

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

                        # Downward crossing = incoming.
                        if (
                            previous_side == -1
                            and current_side == 1
                        ):
                            event_direction = "INCOMING"

                        # Upward crossing = outgoing.
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

                # Update the previous observation after checking
                # whether a crossing occurred.
                last_line_side[track_id] = current_side
                last_seen_frame[track_id] = frame_number

                # ------------------------------------------------
                # ASSIGN BOX COLOR
                # ------------------------------------------------

                if track_id in counted_ids:

                    # Counted vehicles retain their counted color.
                    direction = vehicle_directions[track_id]

                    if direction == "INCOMING":
                        color = BLUE
                    else:
                        color = RED

                    label = (
                        f"{vehicle_name} ID:{track_id} "
                        f"{direction}"
                    )

                else:

                    # Color vehicles immediately using the road.
                    # This does not wait for the counting line.
                    divider_margin = int(
                        width * DIVIDER_MARGIN
                    )

                    if abs(center_x - split_x) <= divider_margin:

                        color = YELLOW
                        label = (
                            f"{vehicle_name} ID:{track_id} "
                            "TRACKING"
                        )

                    elif lane == "LEFT":

                        color = YELLOW

                        label = (
                            f"{vehicle_name} ID:{track_id} "
                            "OUTGOING"
                        )

                    else:

                        color = YELLOW

                        label = (
                            f"{vehicle_name} ID:{track_id} "
                            "INCOMING"
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
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

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
            "YOLO Vehicle Counter - Phase 10",
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
