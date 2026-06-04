import time
import cv2
from ultralytics import YOLO

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "runs/detect/runs/sku110k_yolov8n_gpu_test-3/weights/best.pt"
IMAGE_PATH = "test_images/vegetables.jpeg" 

# -----------------------------
# Load model
# -----------------------------
model = YOLO(MODEL_PATH)

# -----------------------------
# Run prediction
# -----------------------------
start_time = time.time()

results = model.predict(
    source=IMAGE_PATH,
    conf=0.35,
    save=True
)

end_time = time.time()

# -----------------------------
# Extract results
# -----------------------------
result = results[0]
boxes = result.boxes

product_count = len(boxes)

# Image dimensions
img = cv2.imread(IMAGE_PATH)
height, width, _ = img.shape
image_area = width * height

# -----------------------------
# Confidence metrics
# -----------------------------
if product_count > 0:
    confidences = boxes.conf.cpu().numpy()
    avg_confidence = confidences.mean()
    low_confidence_count = (confidences < 0.50).sum()
else:
    avg_confidence = 0
    low_confidence_count = 0

# -----------------------------
# Shelf occupancy calculation
# -----------------------------
total_box_area = 0

for box in boxes.xyxy.cpu().numpy():
    x1, y1, x2, y2 = box
    box_area = max(0, x2 - x1) * max(0, y2 - y1)
    total_box_area += box_area

shelf_occupancy = (total_box_area / image_area) * 100
empty_space = 100 - shelf_occupancy

# Cap occupancy at 100%
shelf_occupancy = min(shelf_occupancy, 100)
empty_space = max(empty_space, 0)

# -----------------------------
# Restock status logic
# -----------------------------
if shelf_occupancy < 40:
    status = "Restock Required"
elif shelf_occupancy < 70:
    status = "Monitor Shelf"
else:
    status = "Healthy Shelf"

# -----------------------------
# Zone-wise product count
# -----------------------------
top_count = 0
middle_count = 0
bottom_count = 0

for box in boxes.xyxy.cpu().numpy():
    x1, y1, x2, y2 = box
    center_y = (y1 + y2) / 2

    if center_y < height / 3:
        top_count += 1
    elif center_y < 2 * height / 3:
        middle_count += 1
    else:
        bottom_count += 1

# -----------------------------
# Inference time
# -----------------------------
total_time_ms = (end_time - start_time) * 1000

# -----------------------------
# Print business metrics
# -----------------------------
print("\n===== Shelf Monitoring Report =====")
print(f"Image analyzed: {IMAGE_PATH}")
print(f"Products detected: {product_count}")
print(f"Average confidence: {avg_confidence:.2f}")
print(f"Low-confidence detections (<0.50): {low_confidence_count}")
print(f"Shelf occupancy: {shelf_occupancy:.2f}%")
print(f"Estimated empty space: {empty_space:.2f}%")
print(f"Inventory status: {status}")
print(f"Top shelf products: {top_count}")
print(f"Middle shelf products: {middle_count}")
print(f"Bottom shelf products: {bottom_count}")
print(f"Total processing time: {total_time_ms:.2f} ms")
print("===================================")