import os
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="preprocessing/data.yaml",
        epochs=10,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        project="runs",
        name="sku110k_yolov8n_gpu_test"
    )


if __name__ == "__main__":
    main()