import sys
from pathlib import Path
import streamlit as st
import pandas as pd

# --- Ensure src/ is importable ---
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

st.set_page_config(page_title="Deepfake Detector", layout="wide")

# Always show header (never blank)
st.markdown(
    "<h1 style='text-align:center; color: grey;'>DEEPFAKE DETECTION IN SOCIAL MEDIA CONTENT</h1>",
    
    unsafe_allow_html=True
)

# --- Import package (failsafe) ---
try:
    from deepfake import (
        AppConfig,
        ImagePreprocessor,
        KerasModelLoader,
        DeepfakePredictor,
        FaceCropper,
        decode_uploaded_image,
    )
except Exception as e:
    st.error("Import error: could not load src/deepfake package")
    st.exception(e)
    st.stop()

CONFIG = AppConfig()

# Cover image (optional)
cover_path = ROOT / "assets" / "coverpage.png"
if cover_path.exists():
    st.image(str(cover_path), use_container_width=True)

# Sidebar settings
with st.sidebar:
    st.title("Settings")
    threshold = st.slider("Fake threshold", 0.50, 0.95, 0.60, 0.01)
    use_face_crop = st.checkbox("Use face-crop (recommended)", True)

    st.divider()
    st.write("Model path")
    st.code(str(CONFIG.model_path))

# Tabs
tab_detect, tab_model, tab_about, tab_debug = st.tabs(["Detect", "Model", "About", "Debug"])

# Build predictor (failsafe)
predictor = None
model_error = None

@st.cache_resource
def build_predictor():
    model = KerasModelLoader(CONFIG.model_path).load()
    pre = ImagePreprocessor(target_size=CONFIG.input_size)
    cropper = FaceCropper()
    return DeepfakePredictor(model=model, preprocessor=pre, class_names=CONFIG.class_names, face_cropper=cropper)

try:
    predictor = build_predictor()
except Exception as e:
    model_error = e

# --- Detect tab ---
with tab_detect:
    st.subheader("Upload one or multiple images")

    if model_error is not None:
        st.warning("Model is not loaded. UI works, but predictions won't run.")
        st.write("Reason:")
        st.exception(model_error)
        st.info("Run `python train.py` to create a dummy model, or add a dataset and train a real one.")
    else:
        st.success("Model loaded ✅")

    uploaded_files = st.file_uploader(
        "Choose image(s)...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("Upload an image to get a prediction.")
    else:
        if predictor is None:
            st.error("Predictor unavailable (model not loaded).")
        else:
            rows = []
            for f in uploaded_files:
                try:
                    img_bgr = decode_uploaded_image(f.read())
                    res = predictor.predict_bgr(img_bgr, threshold=threshold, use_face_crop=use_face_crop)

                    c1, c2 = st.columns([1, 1])
                    with c1:
                        st.image(img_bgr, channels="BGR", caption=f.name, use_container_width=True)

                    with c2:
                        color = "red" if res.is_fake else "green"
                        st.markdown(
                            f"<h2 style='color:{color}; margin-bottom:0;'>Verdict: {res.label}</h2>",
                            unsafe_allow_html=True
                        )
                        st.metric("Fake probability", f"{res.fake_prob:.2%}")
                        st.metric("Real probability", f"{res.real_prob:.2%}")
                        st.caption(f"Threshold: {res.threshold:.2f}")
                        if use_face_crop:
                            st.caption(f"Face-crop used: {'Yes' if res.used_face_crop else 'No (fallback)'}")

                        if res.is_fake:
                            st.warning("Classified as **Fake** at this threshold.")
                        else:
                            st.success("Classified as **Real** at this threshold.")

                    st.divider()

                    rows.append({
                        "file": f.name,
                        "verdict": res.label,
                        "fake_prob": res.fake_prob,
                        "real_prob": res.real_prob,
                        "threshold": res.threshold,
                        "face_crop": res.used_face_crop if use_face_crop else False
                    })

                except Exception as e:
                    st.error(f"{f.name}: error")
                    st.exception(e)

            if rows:
                st.subheader("Batch summary")
                df = pd.DataFrame(rows).sort_values("fake_prob", ascending=False)
                st.dataframe(df, use_container_width=True)

# --- Model tab ---
with tab_model:
    st.subheader("Training graphs & model files")

    st.write(f"Expected model path: `{CONFIG.model_path}`")
    st.write(f"Exists: `{CONFIG.model_path.exists()}`")

    fig1 = ROOT / "assets" / "Figure_1.png"
    fig2 = ROOT / "assets" / "Figure_2.png"

    colA, colB = st.columns(2)
    with colA:
        st.markdown("### Loss")
        if fig1.exists():
            st.image(str(fig1), use_container_width=True)
        else:
            st.info("Put your loss plot at assets/Figure_1.png")

    with colB:
        st.markdown("### Accuracy")
        if fig2.exists():
            st.image(str(fig2), use_container_width=True)
        else:
            st.info("Put your accuracy plot at assets/Figure_2.png")

# --- About tab ---
with tab_about:
    st.header("Understanding Deepfakes")
    st.write(
        "Deepfakes are AI-generated or AI-manipulated images/videos that can imitate real people. "
        "This tool is a classifier that estimates whether an image looks manipulated."
    )

    st.subheader("Ethics & limitations")
    st.write(
        "- This tool is not perfect.\n"
        "- A 'Real' label is not proof of authenticity.\n"
        "- Use as a decision-support signal, not a final verdict."
    )

# --- Debug tab ---
with tab_debug:
    st.subheader("Debug info")
    st.write("Python executable:")
    st.code(sys.executable)
    st.write("First sys.path entries:")
    st.code("\n".join(sys.path[:6]))
