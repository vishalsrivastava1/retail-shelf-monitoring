import time
import tempfile
from pathlib import Path

import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Retail Shelf Monitoring",
    page_icon="🛒",
    layout="wide"
)


# -----------------------------
# Constants
# -----------------------------
MODEL_PATH = "runs/detect/runs/sku110k_yolov8n_gpu_test-3/weights/best.pt"

SAMPLE_IMAGES = {
    "Ice Cream Shelf — Low Inventory": "test_images/icecream.jpeg",
    "Vegetable Shelf — Monitor": "test_images/vegetables.jpeg"
}


# -----------------------------
# Custom CSS
# -----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #1e293b 100%);
        color: #f8fafc;
    }

    .hero-card {
        background: rgba(15, 23, 42, 0.95);
        padding: 1.8rem;
        border-radius: 20px;
        border: 1px solid rgba(148, 163, 184, 0.25);
        box-shadow: 0 20px 35px rgba(0,0,0,0.30);
        margin-bottom: 1.5rem;
    }

    .main-title {
        font-size: 3.2rem;
        font-weight: 850;
        color: #ffffff;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.15rem;
        color: #cbd5e1;
        margin-bottom: 0.8rem;
    }

    .small-note {
        color: #94a3b8;
        font-size: 0.95rem;
    }

    .section-card {
        background: rgba(30, 41, 59, 0.92);
        padding: 1.4rem;
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.18);
        margin-bottom: 1rem;
    }

    .metric-card {
        background: rgba(15, 23, 42, 0.92);
        padding: 1.15rem;
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.25);
        text-align: center;
        min-height: 125px;
        box-shadow: 0 12px 22px rgba(0,0,0,0.18);
    }

    .metric-label {
        font-size: 0.88rem;
        color: #94a3b8;
        margin-bottom: 0.45rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 850;
        color: #ffffff;
    }

    .metric-sub {
        font-size: 0.78rem;
        color: #cbd5e1;
        margin-top: 0.25rem;
    }

    .status-restock {
        background: linear-gradient(135deg, #7f1d1d, #dc2626);
        padding: 1.3rem;
        border-radius: 16px;
        color: white;
        font-weight: 800;
        font-size: 1.35rem;
        text-align: center;
        box-shadow: 0 14px 28px rgba(220, 38, 38, 0.25);
    }

    .status-monitor {
        background: linear-gradient(135deg, #78350f, #f59e0b);
        padding: 1.3rem;
        border-radius: 16px;
        color: white;
        font-weight: 800;
        font-size: 1.35rem;
        text-align: center;
        box-shadow: 0 14px 28px rgba(245, 158, 11, 0.25);
    }

    .status-healthy {
        background: linear-gradient(135deg, #064e3b, #10b981);
        padding: 1.3rem;
        border-radius: 16px;
        color: white;
        font-weight: 800;
        font-size: 1.35rem;
        text-align: center;
        box-shadow: 0 14px 28px rgba(16, 185, 129, 0.25);
    }

    .recommendation-card {
        background: rgba(15, 23, 42, 0.92);
        padding: 1.3rem;
        border-radius: 16px;
        border-left: 5px solid #38bdf8;
        border-top: 1px solid rgba(148, 163, 184, 0.20);
        border-right: 1px solid rgba(148, 163, 184, 0.20);
        border-bottom: 1px solid rgba(148, 163, 184, 0.20);
        min-height: 115px;
    }

    .stImage img {
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.25);
        box-shadow: 0 12px 24px rgba(0,0,0,0.25);
    }

    [data-testid="stSidebar"] {
        background: #020617;
    }

    [data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.85);
        border-radius: 16px;
        padding: 1rem;
        border: 1px dashed rgba(148, 163, 184, 0.35);
    }

    h1, h2, h3, h4 {
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()


# -----------------------------
# Helper functions
# -----------------------------
def analyze_shelf(image_path, conf_threshold):
    start_time = time.time()

    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        save=False
    )

    end_time = time.time()

    result = results[0]
    boxes = result.boxes
    product_count = int(len(boxes))

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    height, width, _ = img.shape
    image_area = float(width * height)

    # Confidence metrics
    if product_count > 0:
        confidences = boxes.conf.cpu().numpy()
        avg_confidence = float(confidences.mean())
        low_confidence_count = int((confidences < 0.50).sum())
    else:
        avg_confidence = 0.0
        low_confidence_count = 0

    # Occupancy calculation based on detected bounding-box area
    total_box_area = 0.0

    for box in boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = box
        box_area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
        total_box_area += box_area

    shelf_occupancy = float((total_box_area / image_area) * 100.0)
    shelf_occupancy = min(shelf_occupancy, 100.0)
    empty_space = float(max(100.0 - shelf_occupancy, 0.0))

    # Business status logic
    if shelf_occupancy < 40:
        status = "🚨 Restock Required"
        status_class = "status-restock"
        recommendation = (
            "Detected product coverage is low. This shelf should be prioritized "
            "for replenishment in the next restocking cycle."
        )
    elif shelf_occupancy < 70:
        status = "⚠️ Monitor Shelf"
        status_class = "status-monitor"
        recommendation = (
            "Detected product coverage is moderate. Staff should monitor this shelf "
            "and prepare for potential replenishment soon."
        )
    else:
        status = "✅ Healthy Shelf"
        status_class = "status-healthy"
        recommendation = (
            "Detected product coverage appears healthy. No immediate restocking action "
            "is required based on current shelf visibility."
        )

    # Zone-wise product count
    top_count = 0
    middle_count = 0
    bottom_count = 0

    for box in boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = box
        center_y = float((y1 + y2) / 2)

        if center_y < height / 3:
            top_count += 1
        elif center_y < 2 * height / 3:
            middle_count += 1
        else:
            bottom_count += 1

    total_time_ms = float((end_time - start_time) * 1000.0)

    # Annotated image
    annotated_image = result.plot()
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    metrics = {
        "product_count": product_count,
        "avg_confidence": avg_confidence,
        "low_confidence_count": low_confidence_count,
        "shelf_occupancy": shelf_occupancy,
        "empty_space": empty_space,
        "status": status,
        "status_class": status_class,
        "recommendation": recommendation,
        "top_count": int(top_count),
        "middle_count": int(middle_count),
        "bottom_count": int(bottom_count),
        "total_time_ms": total_time_ms
    }

    return annotated_image, metrics


def save_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.read())
        return temp_file.name


def metric_card(label, value, subtext):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.markdown("## 🛒 Retail Shelf Monitoring")
st.sidebar.markdown("Retail Operations Dashboard")

st.sidebar.markdown("---")

conf_threshold = st.sidebar.slider(
    "Detection confidence threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.35,
    step=0.05
)

st.sidebar.markdown("---")

st.sidebar.markdown("### Model")
st.sidebar.markdown(
    """
    **Architecture:** YOLOv8n  
    **Framework:** PyTorch / Ultralytics  
    **Dataset:** SKU110K subset  
    **Task:** Product object detection
    """
)

st.sidebar.markdown("### Independent Test Results")
st.sidebar.markdown(
    """
    - **Precision:** 88.8%  
    - **Recall:** 82.1%  
    - **mAP50:** 88.2%  
    - **mAP50-95:** 52.8%
    """
)


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="hero-card">
        <div class="main-title">Retail Shelf Monitoring</div>
        <div class="subtitle">
            Real-time retail shelf monitoring using YOLOv8 computer vision.
        </div>
        <div class="small-note">
            Detect products, estimate shelf occupancy, and generate restock recommendations from shelf images.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(
    ["📸 Shelf Scan", "📊 Model Performance", "🏗️ System Design"]
)


# -----------------------------
# Tab 1: Shelf Scan
# -----------------------------
with tab1:
    st.markdown("## Shelf Scan Demo")

    input_mode = st.radio(
        "Choose input mode",
        ["Use sample image", "Upload custom image"],
        horizontal=True
    )

    selected_image_path = None
    original_image = None

    if input_mode == "Use sample image":
        sample_choice = st.selectbox(
            "Choose a demo shelf image",
            list(SAMPLE_IMAGES.keys())
        )

        selected_image_path = SAMPLE_IMAGES[sample_choice]
        original_image = Image.open(selected_image_path)

        st.caption(
            "Tip: Use the ice cream shelf during the presentation because it clearly produces a restock recommendation."
        )

    else:
        uploaded_file = st.file_uploader(
            "Upload a retail shelf image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            selected_image_path = save_uploaded_file(uploaded_file)
            original_image = Image.open(selected_image_path)

    analyze_button = st.button("Run Shelf Analysis", type="primary")

    if selected_image_path is not None and analyze_button:
        with st.spinner("Running YOLOv8 product detection..."):
            annotated_image, metrics = analyze_shelf(selected_image_path, conf_threshold)

        st.markdown("---")
        st.markdown("## Executive Shelf Report")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            metric_card(
                "Products detected",
                metrics["product_count"],
                "YOLO bounding boxes"
            )

        with col2:
            metric_card(
                "Shelf occupancy",
                f"{metrics['shelf_occupancy']:.1f}%",
                "Estimated product coverage"
            )

        with col3:
            metric_card(
                "Empty space",
                f"{metrics['empty_space']:.1f}%",
                "Estimated available shelf area"
            )

        with col4:
            metric_card(
                "Avg confidence",
                f"{metrics['avg_confidence']:.2f}",
                "Model confidence score"
            )

        st.markdown("### Occupancy Gauge")

        progress_value = float(metrics["shelf_occupancy"]) / 100.0
        progress_value = max(0.0, min(progress_value, 1.0))

        st.progress(progress_value)

        st.caption(
            "Occupancy is estimated using total detected bounding-box area divided by total image area."
        )

        st.markdown("---")
        st.markdown("## Detection Output")

        image_col1, image_col2 = st.columns(2)

        with image_col1:
            st.markdown("### Original Image")
            st.image(original_image, use_container_width=True)

        with image_col2:
            st.markdown("### YOLO Detection")
            st.image(annotated_image, use_container_width=True)

        st.markdown("---")
        st.markdown("## Inventory Decision")

        status_col, rec_col = st.columns([1, 2])

        with status_col:
            st.markdown(
                f"""
                <div class="{metrics["status_class"]}">
                    {metrics["status"]}
                </div>
                """,
                unsafe_allow_html=True
            )

        with rec_col:
            st.markdown(
                f"""
                <div class="recommendation-card">
                    <h4>Business Recommendation</h4>
                    <p>{metrics["recommendation"]}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("## Shelf Zone Analysis")

        zone1, zone2, zone3 = st.columns(3)

        with zone1:
            metric_card(
                "Top shelf",
                metrics["top_count"],
                "Detected products"
            )

        with zone2:
            metric_card(
                "Middle shelf",
                metrics["middle_count"],
                "Detected products"
            )

        with zone3:
            metric_card(
                "Bottom shelf",
                metrics["bottom_count"],
                "Detected products"
            )

        st.markdown("---")
        st.markdown("## System Performance")

        perf1, perf2 = st.columns(2)

        with perf1:
            metric_card(
                "Processing time",
                f"{metrics['total_time_ms']:.1f} ms",
                "End-to-end local inference"
            )

        with perf2:
            metric_card(
                "Low confidence detections",
                metrics["low_confidence_count"],
                "Boxes below 0.50 confidence"
            )

    elif selected_image_path is not None:
        st.info("Click **Run Shelf Analysis** to analyze the selected image.")
    else:
        st.info("Select or upload a shelf image to begin.")


# -----------------------------
# Tab 2: Model Performance
# -----------------------------
with tab2:
    st.markdown("## Model Performance")

    st.markdown(
        """
        <div class="section-card">
            <h4>Evaluation Summary</h4>
            <p>
            The YOLOv8n model was trained on a subset of the SKU110K retail shelf dataset
            and evaluated on an independent test set.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        metric_card("Precision", "88.8%", "Correctness of detections")

    with m2:
        metric_card("Recall", "82.1%", "Products successfully found")

    with m3:
        metric_card("mAP50", "88.2%", "Main detection metric")

    with m4:
        metric_card("mAP50-95", "52.8%", "Strict localization metric")

    st.markdown("### Dataset Split")

    st.table(
        {
            "Split": ["Train", "Validation", "Test"],
            "Images": ["~8,192", "584", "2,920"],
            "Purpose": [
                "Model learning",
                "Monitoring during training",
                "Independent final evaluation"
            ]
        }
    )

    st.markdown("### Interpretation")

    st.markdown(
        """
        - **Precision 88.8%:** Most detected boxes correspond to real products.
        - **Recall 82.1%:** The model finds most visible products in dense shelf scenes.
        - **mAP50 88.2%:** Strong object detection performance for a prototype system.
        - **mAP50-95 52.8%:** Stricter localization score, reasonable for crowded retail shelf imagery.
        """
    )


# -----------------------------
# Tab 3: System Design
# -----------------------------
with tab3:
    st.markdown("## System Architecture")

    st.markdown(
        """
        <div class="section-card">
            <h4>Prototype Architecture</h4>
            <p>
            Shelf image → YOLOv8 detection → analytics layer → dashboard → inventory decision
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.code(
        """
Retail Shelf Image
        ↓
YOLOv8 Object Detection Model
        ↓
Detected Product Bounding Boxes
        ↓
Analytics Layer
  - Product count
  - Average confidence
  - Shelf occupancy
  - Empty space estimate
  - Zone-wise count
        ↓
Business Rule Engine
  - Restock Required
  - Monitor Shelf
  - Healthy Shelf
        ↓
Streamlit Dashboard
        """,
        language="text"
    )

    st.markdown("## Production Deployment Path")

    st.markdown(
        """
        For the prototype, the system runs locally using Streamlit and a YOLOv8 model.
        For production deployment, the system can be expanded into:
        
        - **React frontend** for a more interactive retail operations dashboard
        - **FastAPI backend** for model inference
        - **GPU cloud service** for scalable prediction
        - **Database layer** for historical shelf monitoring
        - **Camera integration** for real-time shelf image capture
        """
    )

    with st.expander("Limitations and Risks"):
        st.markdown(
            """
            - The current model detects generic products, not exact SKU identities.
            - Shelf occupancy is estimated from bounding-box area, not true physical shelf capacity.
            - Dense overlapping products can lead to missed detections.
            - Reflections, blur, unusual camera angles, or poor lighting may reduce detection quality.
            - The current prototype processes uploaded images rather than live camera streams.
            """
        )

    with st.expander("Future Improvements"):
        st.markdown(
            """
            - Add product classification using SKU-level labels or CLIP-based visual search.
            - Add time-series monitoring to compare shelf conditions across the day.
            - Add multi-image batch upload for store-wide shelf audits.
            - Integrate alerts through email, Slack, or store operations systems.
            - Deploy React + FastAPI version for a production-style application.
            """
        )