"""Minimal Streamlit UI: upload a cell image → Yes / No cancer."""

from pathlib import Path

import streamlit as st
from PIL import Image

from model_utils import MODEL_PATH, load_cancer_model, predict_cancer

st.set_page_config(
    page_title="Cancer Cell Detection",
    page_icon="🔬",
    layout="centered",
)

st.markdown(
    """
    <style>
      .block-container { max-width: 520px; padding-top: 3rem; }
      .result-yes {
        background: #fde8e8;
        color: #9b1c1c;
        border: 1px solid #f8b4b4;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        font-size: 1.75rem;
        font-weight: 700;
        margin-top: 1rem;
      }
      .result-no {
        background: #def7ec;
        color: #03543f;
        border: 1px solid #84e1bc;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        font-size: 1.75rem;
        font-weight: 700;
        margin-top: 1rem;
      }
      .hint { color: #6b7280; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Cancer Cell Detection")
st.markdown(
    '<p class="hint">Upload a histopathology image. You get a simple Yes or No.</p>',
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model(_mtime: float):
    # _mtime busts cache when the .keras file is retrained/replaced
    return load_cancer_model()


uploaded = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"],
    label_visibility="collapsed",
)

if uploaded is None:
    st.caption("Supported: PNG, JPG, TIFF, BMP, WEBP")
    if not MODEL_PATH.exists():
        st.warning(
            "Model file missing. Create one with:\n\n"
            "`python3.11 train_bootstrap.py`"
        )
else:
    image = Image.open(uploaded)
    st.image(image, width="stretch", caption="Uploaded image")

    try:
        mtime = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else 0.0
        model = get_model(mtime)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    with st.spinner("Analyzing…"):
        result = predict_cancer(model, image)

    if result["has_cancer"]:
        st.markdown(
            '<div class="result-yes">Yes — Cancer detected</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="result-no">No — Cancer not detected</div>',
            unsafe_allow_html=True,
        )

    st.caption(
        f"Confidence: {result['confidence']:.1%} · Boolean: `{result['has_cancer']}`"
    )

st.markdown("---")
st.caption("Research / educational use only. Not a medical diagnosis.")
