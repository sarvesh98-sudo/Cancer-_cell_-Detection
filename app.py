"""CELLSIGHT — Streamlit UI for cancer cell detection."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Optional

import streamlit as st
from PIL import Image

from model_utils import (
    HAS_TENSORFLOW,
    MODEL_PATH,
    load_cancer_model,
    predict_cancer,
)

DEMO_DIR = Path(__file__).resolve().parent / "demo_images"

st.set_page_config(
    page_title="CELLSIGHT · Cancer Cell Detection",
    page_icon="⬡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Cloud-safe CSS injection (st.markdown often strips <style> / <h1> on deploy)
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Syne:wght@600;700;800&display=swap');

:root {
  --ink: #0a1c1f;
  --ink-soft: #3d5559;
  --paper: #f4f9f8;
  --teal: #0b6e6a;
  --teal-deep: #084e4b;
  --alert: #b42318;
  --alert-bg: #fef3f2;
  --clear: #067647;
  --clear-bg: #ecfdf3;
  --line: rgba(10, 28, 31, 0.12);
}

html, body, [class*="css"], .stApp, .stMarkdown, p, div, span, button, label {
  font-family: "Syne", "Segoe UI", sans-serif !important;
}

.stApp {
  background:
    radial-gradient(ellipse 90% 55% at 8% -8%, rgba(11, 110, 106, 0.18), transparent 55%),
    radial-gradient(ellipse 65% 45% at 100% 0%, rgba(180, 90, 60, 0.09), transparent 50%),
    linear-gradient(165deg, #e4f0ef 0%, #f4f9f8 45%, #eef5f3 100%) !important;
}

[data-testid="stHeader"] { background: transparent !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

.block-container {
  max-width: 720px !important;
  padding-top: 1.5rem !important;
  padding-bottom: 2.5rem !important;
}

[data-testid="stFileUploader"] {
  background: rgba(255,255,255,0.72) !important;
  border: 1.5px dashed rgba(11, 110, 106, 0.4) !important;
  border-radius: 18px !important;
  padding: 0.85rem 0.75rem 0.35rem !important;
}

[data-testid="stFileUploader"] section {
  border: none !important;
  background: transparent !important;
}

[data-testid="stFileUploader"] button {
  background: #0a1c1f !important;
  color: #f4f9f8 !important;
  border: none !important;
  border-radius: 10px !important;
  font-weight: 700 !important;
}

[data-testid="stFileUploader"] button:hover {
  background: #084e4b !important;
}

div[data-testid="stHorizontalBlock"] button {
  border-radius: 10px !important;
  border: 1px solid var(--line) !important;
  background: rgba(255,255,255,0.85) !important;
  color: #0a1c1f !important;
  font-weight: 700 !important;
  font-size: 0.78rem !important;
}

[data-testid="stImage"] img {
  border-radius: 14px !important;
  box-shadow: 0 8px 28px rgba(10, 28, 31, 0.08);
}

.stSpinner > div { border-top-color: #0b6e6a !important; }
"""


def inject_css(css: str) -> None:
    """Inject CSS in a Cloud-safe way (markdown often strips <style>)."""
    payload = f"<style>{css}</style>"
    if hasattr(st, "html"):
        st.html(payload)
    else:
        st.markdown(payload, unsafe_allow_html=True)


inject_css(CSS)

# Brand hero — use <div>/<p> (Streamlit Cloud often strips <h1>)
# Inline styles so brand stays visible even if CSS injection fails
st.markdown(
    """
    <div style="text-align:center;padding:1.75rem 0 0.5rem;animation:none;">
      <div style="
        font-family:Syne,Segoe UI,sans-serif;
        font-weight:800;
        font-size:clamp(2.6rem,9vw,4.2rem);
        letter-spacing:-0.045em;
        line-height:0.95;
        color:#0a1c1f;
        margin:0;
      ">CELL<span style="color:#0b6e6a;">SIGHT</span></div>
      <p style="
        font-family:Newsreader,Georgia,serif;
        font-size:clamp(1.15rem,2.8vw,1.4rem);
        font-weight:500;
        color:#3d5559;
        margin:1rem auto 0.4rem;
        max-width:28ch;
        line-height:1.35;
      ">See what the eye might miss.</p>
      <p style="
        font-family:Syne,Segoe UI,sans-serif;
        font-size:0.95rem;
        color:#3d5559;
        margin:0 auto;
        max-width:40ch;
        line-height:1.5;
        opacity:0.9;
      ">Upload a histopathology slide. Instant Yes / No from nuclear patterns and a trained CNN.</p>
      <div style="
        width:48px;height:3px;margin:1.25rem auto 0;border-radius:2px;
        background:linear-gradient(90deg,transparent,#0b6e6a,transparent);
      "></div>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model(_mtime: float):
    return load_cancer_model()


@st.cache_data
def load_demo(name: str) -> Image.Image:
    return Image.open(DEMO_DIR / name).convert("RGB")


def image_to_data_uri(image: Image.Image, max_side: int = 720) -> str:
    img = image.copy()
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def run_prediction(image: Image.Image) -> dict:
    model = None
    if HAS_TENSORFLOW and MODEL_PATH.exists():
        try:
            model = get_model(MODEL_PATH.stat().st_mtime)
        except Exception:
            model = None
    return predict_cancer(model, image)


uploaded = st.file_uploader(
    "Drop a slide image here",
    type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"],
    label_visibility="collapsed",
    help="PNG, JPG, TIFF, BMP, or WEBP",
)

if "demo_name" not in st.session_state:
    st.session_state.demo_name = None

demo_files = sorted(DEMO_DIR.glob("*.png")) if DEMO_DIR.exists() else []

if demo_files and uploaded is None:
    st.markdown(
        """
        <p style="
          font-size:0.72rem;font-weight:700;letter-spacing:0.14em;
          text-transform:uppercase;color:#3d5559;text-align:center;
          margin:1.4rem 0 0.55rem;font-family:Syne,sans-serif;
        ">Or try a sample slide</p>
        """,
        unsafe_allow_html=True,
    )
    # Two rows of 3 — cleaner on mobile / Cloud
    clear = [p for p in demo_files if p.name.startswith("no_")]
    positive = [p for p in demo_files if p.name.startswith("yes_")]
    for group, label in ((clear, "Clear"), (positive, "Positive")):
        if not group:
            continue
        cols = st.columns(len(group))
        for col, path in zip(cols, group):
            with col:
                st.image(load_demo(path.name), use_container_width=True)
                if st.button(label, key=f"demo_{path.name}", use_container_width=True):
                    st.session_state.demo_name = path.name

image: Optional[Image.Image] = None
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.session_state.demo_name = None
elif st.session_state.demo_name:
    image = load_demo(st.session_state.demo_name)

if image is None:
    if not HAS_TENSORFLOW:
        st.caption("Feature mode — densitometry scoring without TensorFlow.")
    elif not MODEL_PATH.exists():
        st.caption("Model missing. Train with `python3.11 train_bootstrap.py`.")
else:
    uri = image_to_data_uri(image)
    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.markdown(
            f"""
            <div style="
              position:relative;border-radius:18px;overflow:hidden;
              border:1px solid rgba(10,28,31,0.12);background:#0a1c1f;
              aspect-ratio:1;
            ">
              <img src="{uri}" alt="Slide preview" style="
                width:100%;height:100%;object-fit:cover;display:block;
                filter:saturate(1.08) contrast(1.04);
              "/>
              <div style="
                position:absolute;left:0;right:0;height:32%;top:0;
                background:linear-gradient(180deg,transparent,
                  rgba(120,230,220,0.22),transparent);
                animation:csScan 1.7s ease-in-out infinite;
                pointer-events:none;
              "></div>
            </div>
            <style>
              @keyframes csScan {{
                0% {{ transform: translateY(-30%); opacity: 0; }}
                15% {{ opacity: 0.9; }}
                100% {{ transform: translateY(130%); opacity: 0; }}
              }}
            </style>
            """,
            unsafe_allow_html=True,
        )

    with right:
        with st.spinner("Reading cellular architecture…"):
            result = run_prediction(image)

        kind = "yes" if result["has_cancer"] else "no"
        title = "Cancer signals found" if result["has_cancer"] else "No cancer signals"
        kicker = "Positive" if result["has_cancer"] else "Negative"
        pct = result["confidence"] * 100

        if kind == "yes":
            bg, border, color, bar = "#fef3f2", "rgba(180,35,24,0.22)", "#b42318", "#b42318"
        else:
            bg, border, color, bar = "#ecfdf3", "rgba(6,118,71,0.22)", "#067647", "#067647"

        st.markdown(
            f"""
            <div style="
              margin-top:0.2rem;padding:2rem 1.35rem;border-radius:22px;
              text-align:center;background:{bg};border:1px solid {border};
              color:{color};
            ">
              <div style="
                font-size:0.7rem;font-weight:700;letter-spacing:0.18em;
                text-transform:uppercase;opacity:0.75;margin-bottom:0.5rem;
                font-family:Syne,sans-serif;
              ">{kicker} reading</div>
              <p style="
                font-family:Syne,sans-serif;font-weight:800;
                font-size:clamp(1.55rem,3.5vw,2.2rem);letter-spacing:-0.03em;
                line-height:1.1;margin:0;color:{color};
              ">{title}</p>
              <div style="
                margin:1.25rem auto 0;max-width:240px;height:6px;
                border-radius:99px;background:rgba(10,28,31,0.08);overflow:hidden;
              "><div style="width:{pct:.1f}%;height:100%;background:{bar};border-radius:99px;"></div></div>
              <p style="
                font-family:Newsreader,Georgia,serif;font-size:1.05rem;
                margin-top:0.85rem;opacity:0.9;color:{color};
              ">{pct:.0f}% confidence</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <div style="
      text-align:center;margin-top:2.25rem;padding-top:1.1rem;
      border-top:1px solid rgba(10,28,31,0.12);font-size:0.78rem;
      color:#3d5559;font-family:Syne,sans-serif;
    ">
      CELLSIGHT · Research &amp; educational use only · Not a medical diagnosis
    </div>
    """,
    unsafe_allow_html=True,
)
