from ultralytics import YOLO


def main():
    model = YOLO(
        "runs/detect/runs/sku110k_yolov8n_gpu_test-3/weights/best.pt"
    )

    metrics = model.val(
        data="preprocessing/data.yaml",
        split="test",
        device=0,
        workers=0
    )

    print(metrics)


if __name__ == "__main__":
    main()