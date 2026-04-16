import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import io
import json
import tempfile

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from src.utils.config import (
    MODELS_DIR,
    RUNS_DIR,
    EXPERIMENT_NAME,
    TOP_K,
)

BEST_MODEL_PATH = RUNS_DIR / EXPERIMENT_NAME / "weights" / "best.pt"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"


@st.cache_resource
def load_model(model_path: Path) -> YOLO:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return YOLO(str(model_path))


@st.cache_data
def load_class_names(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Class names file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    if not isinstance(class_names, list) or not class_names:
        raise ValueError("Invalid class names file format.")

    return class_names


def predict_image(
        model: YOLO,
        image_path: Path,
        class_names: list[str],
        top_k: int = TOP_K,
) -> dict:
    results = model.predict(
        source=str(image_path),
        verbose=False,
    )

    result = results[0]
    probs = result.probs

    top1_index = int(probs.top1)
    top1_conf = float(probs.top1conf)

    top_indices = probs.top5[:top_k]
    top_confidences = probs.top5conf[:top_k]

    top_predictions = []
    for idx, conf in zip(top_indices, top_confidences):
        idx = int(idx)
        conf = float(conf)
        top_predictions.append(
            {
                "class_id": idx,
                "class_name": class_names[idx],
                "confidence": conf,
            }
        )

    return {
        "predicted_class_id": top1_index,
        "predicted_class_name": class_names[top1_index],
        "confidence": top1_conf,
        "top_predictions": top_predictions,
    }


def format_class_name(class_name: str) -> tuple[str, str]:
    if "___" in class_name:
        plant, disease = class_name.split("___", 1)
    else:
        plant, disease = "Unknown", class_name

    plant = plant.replace("_", " ").replace(",", "")
    disease = disease.replace("_", " ")

    return plant, disease


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .main-title {
                font-size: 2.2rem;
                font-weight: 700;
                margin-bottom: 0.2rem;
            }
            .subtitle {
                color: #6b7280;
                font-size: 1rem;
                margin-bottom: 1.5rem;
            }
            .result-card {
                border: 1px solid rgba(120,120,120,0.18);
                border-radius: 18px;
                padding: 18px 20px;
                margin-top: 12px;
                background: rgba(255,255,255,0.02);
            }
            .metric-label {
                color: #6b7280;
                font-size: 0.9rem;
                margin-bottom: 0.2rem;
            }
            .metric-value {
                font-size: 1.2rem;
                font-weight: 600;
            }
            .healthy-badge {
                display: inline-block;
                padding: 0.35rem 0.7rem;
                border-radius: 999px;
                background: rgba(34,197,94,0.15);
                color: #16a34a;
                font-weight: 600;
                font-size: 0.9rem;
            }
            .disease-badge {
                display: inline-block;
                padding: 0.35rem 0.7rem;
                border-radius: 999px;
                background: rgba(239,68,68,0.15);
                color: #dc2626;
                font-weight: 600;
                font-size: 0.9rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_top_predictions_table(top_predictions: list[dict]) -> pd.DataFrame:
    rows = []

    for i, pred in enumerate(top_predictions, start=1):
        plant, disease = format_class_name(pred["class_name"])
        rows.append(
            {
                "Rank": i,
                "Plant": plant,
                "Class": disease,
                "Confidence": round(pred["confidence"], 4),
            }
        )

    return pd.DataFrame(rows)


def image_from_uploaded_file(uploaded_file) -> Image.Image:
    return Image.open(uploaded_file).convert("RGB")


def image_from_camera_file(camera_file) -> Image.Image:
    bytes_data = camera_file.getvalue()
    return Image.open(io.BytesIO(bytes_data)).convert("RGB")


def save_temp_image(image: Image.Image) -> Path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        image.save(tmp_file.name)
        return Path(tmp_file.name)


def main() -> None:
    st.set_page_config(
        page_title="Plant Disease Identification",
        page_icon="🌿",
        layout="wide",
    )

    inject_styles()

    with st.sidebar:
        st.header("Model info")
        st.write("**Architecture:** YOLOv11 classification")
        st.write(f"**Experiment:** {EXPERIMENT_NAME}")
        st.write("**Weights:** best.pt")
        st.write(f"**Top-K shown:** {TOP_K}")
        st.write("**Classes:** 38")
        st.caption(
            "This interface demonstrates the developed method for plant disease "
            "identification from leaf images."
        )

    st.markdown(
        '<div class="main-title">🌿 Plant Disease Identification</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">Upload a leaf image or take a photo and the model will identify the most probable disease class.</div>',
        unsafe_allow_html=True,
    )

    try:
        model = load_model(BEST_MODEL_PATH)
        class_names = load_class_names(CLASS_NAMES_PATH)
    except Exception as e:
        st.error(f"Initialization error: {e}")
        st.stop()

    col1, col2 = st.columns([1.05, 1], gap="large")

    selected_image = None
    source_label = None

    with col1:
        st.subheader("Image source")

        source_option = st.radio(
            "Choose image source",
            ["Upload image", "Take photo"],
            horizontal=True,
        )

        if source_option == "Upload image":
            uploaded_file = st.file_uploader(
                "Upload a leaf image",
                type=["jpg", "jpeg", "png"],
            )

            st.caption(
                "Tip: in many browsers you can also drag and drop an image into the upload area."
            )

            if uploaded_file is not None:
                selected_image = image_from_uploaded_file(uploaded_file)
                source_label = "Uploaded image"

        elif source_option == "Take photo":
            camera_file = st.camera_input("Take a photo of the leaf")

            if camera_file is not None:
                selected_image = image_from_camera_file(camera_file)
                source_label = "Captured image"

        if selected_image is not None:
            st.image(selected_image, caption=source_label, use_container_width=True)
        else:
            st.info("Please upload an image or take a photo to begin identification.")

    with col2:
        st.subheader("Prediction panel")

        if selected_image is not None:
            if st.button("Identify disease", use_container_width=True):
                temp_image_path = None
                with st.spinner("Processing image..."):
                    try:
                        temp_image_path = save_temp_image(selected_image)

                        prediction = predict_image(
                            model=model,
                            image_path=temp_image_path,
                            class_names=class_names,
                            top_k=TOP_K,
                        )

                        plant_name, disease_name = format_class_name(
                            prediction["predicted_class_name"]
                        )

                        st.success("Identification completed successfully.")

                        badge_html = (
                            '<span class="healthy-badge">Healthy</span>'
                            if disease_name.lower() == "healthy"
                            else '<span class="disease-badge">Disease detected</span>'
                        )
                        st.markdown(badge_html, unsafe_allow_html=True)

                        st.markdown('<div class="result-card">', unsafe_allow_html=True)

                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown(
                                '<div class="metric-label">Plant</div>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f'<div class="metric-value">{plant_name}</div>',
                                unsafe_allow_html=True,
                            )

                        with c2:
                            st.markdown(
                                '<div class="metric-label">Predicted class</div>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f'<div class="metric-value">{disease_name}</div>',
                                unsafe_allow_html=True,
                            )

                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown(
                            '<div class="metric-label">Confidence</div>',
                            unsafe_allow_html=True,
                        )
                        st.progress(float(prediction["confidence"]))
                        st.write(f"**{prediction['confidence']:.4f}**")

                        st.markdown("</div>", unsafe_allow_html=True)

                        st.subheader(f"Top-{TOP_K} predictions")
                        top_df = build_top_predictions_table(
                            prediction["top_predictions"]
                        )
                        st.dataframe(top_df, use_container_width=True, hide_index=True)

                        st.subheader("Interpretation")
                        if disease_name.lower() == "healthy":
                            st.info(
                                "The model predicts that the uploaded leaf belongs to a healthy plant class."
                            )
                        else:
                            st.warning(
                                "The model predicts that the uploaded leaf contains visible signs of plant disease."
                            )

                    except Exception as e:
                        st.error(f"Prediction error: {e}")
                    finally:
                        if temp_image_path is not None and temp_image_path.exists():
                            temp_image_path.unlink(missing_ok=True)
        else:
            st.write("The prediction result will appear here after image selection.")

    st.markdown("---")
    st.caption(
        "The application is intended for demonstration of the developed method for plant disease identification by leaf image."
    )


if __name__ == "__main__":
    main()