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
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Syne:wght@500;600;700;800&display=swap" rel="stylesheet">
    <style>
      :root {
        --ink: #0a1c1f;
        --ink-soft: #3d5559;
        --paper: #f7fbfa;
        --teal: #0b6e6a;
        --teal-deep: #084e4b;
        --alert: #b42318;
        --alert-bg: #fef3f2;
        --clear: #067647;
        --clear-bg: #ecfdf3;
        --line: rgba(10, 28, 31, 0.10);
        --glow-teal: rgba(11, 110, 106, 0.18);
      }

      html, body, [class*="css"] {
        font-family: "Syne", sans-serif;
      }

      .stApp {
        background:
          radial-gradient(ellipse 90% 60% at 10% -10%, rgba(11, 110, 106, 0.16), transparent 55%),
          radial-gradient(ellipse 70% 50% at 95% 5%, rgba(180, 90, 60, 0.08), transparent 50%),
          radial-gradient(circle at 50% 120%, rgba(11, 110, 106, 0.10), transparent 45%),
          linear-gradient(165deg, #e8f2f1 0%, var(--paper) 42%, #f3f7f5 100%);
        background-attachment: fixed;
      }

      .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        opacity: 0.35;
        background-image:
          radial-gradient(circle at 20% 30%, rgba(11,110,106,0.06) 0 1px, transparent 1.5px),
          radial-gradient(circle at 70% 60%, rgba(10,28,31,0.05) 0 1px, transparent 1.5px);
        background-size: 28px 28px, 42px 42px;
        z-index: 0;
        animation: drift 28s linear infinite;
      }

      @keyframes drift {
        from { background-position: 0 0, 0 0; }
        to { background-position: 112px 84px, -84px 126px; }
      }

      @keyframes rise {
        from { opacity: 0; transform: translateY(18px); }
        to { opacity: 1; transform: translateY(0); }
      }

      @keyframes scan {
        0% { transform: translateY(-30%); opacity: 0; }
        15% { opacity: 0.85; }
        100% { transform: translateY(130%); opacity: 0; }
      }

      @keyframes pulse-ring {
        0% { box-shadow: 0 0 0 0 var(--glow-teal); }
        70% { box-shadow: 0 0 0 16px transparent; }
        100% { box-shadow: 0 0 0 0 transparent; }
      }

      @keyframes result-in {
        from { opacity: 0; transform: scale(0.96) translateY(12px); filter: blur(4px); }
        to { opacity: 1; transform: scale(1) translateY(0); filter: blur(0); }
      }

      #MainMenu, footer, header { visibility: hidden; }
      [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }
      [data-testid="stHeader"] { background: transparent; }

      .block-container {
        max-width: 960px;
        padding: 1.1rem 1.4rem 2.75rem;
        position: relative;
        z-index: 1;
      }

      .cs-hero {
        text-align: center;
        padding: 2.25rem 0 1.5rem;
        animation: rise 0.85s ease-out both;
      }

      .cs-brand {
        font-family: "Syne", sans-serif;
        font-weight: 800;
        font-size: clamp(2.8rem, 8vw, 4.6rem);
        letter-spacing: -0.045em;
        line-height: 0.95;
        color: var(--ink);
        margin: 0;
      }

      .cs-brand span {
        background: linear-gradient(120deg, var(--teal-deep) 0%, var(--teal) 55%, #1a9b94 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
      }

      .cs-tagline {
        font-family: "Newsreader", Georgia, serif;
        font-size: clamp(1.15rem, 2.5vw, 1.45rem);
        font-weight: 500;
        color: var(--ink-soft);
        margin: 1rem auto 0.35rem;
        max-width: 28ch;
        line-height: 1.35;
        animation: rise 0.85s ease-out 0.12s both;
      }

      .cs-sub {
        font-size: 0.95rem;
        color: var(--ink-soft);
        opacity: 0.85;
        margin: 0 auto;
        max-width: 44ch;
        line-height: 1.5;
        animation: rise 0.85s ease-out 0.22s both;
      }

      .cs-rule {
        width: 48px;
        height: 3px;
        margin: 1.4rem auto 0;
        border-radius: 2px;
        background: linear-gradient(90deg, transparent, var(--teal), transparent);
        animation: rise 0.85s ease-out 0.3s both;
      }

      .cs-stage { margin-top: 1.75rem; animation: rise 0.9s ease-out 0.35s both; }

      [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.55);
        backdrop-filter: blur(12px);
        border: 1.5px dashed rgba(11, 110, 106, 0.35);
        border-radius: 20px;
        padding: 1.15rem 1rem 0.4rem;
        transition: border-color 0.25s ease, box-shadow 0.25s ease;
        animation: pulse-ring 2.8s ease-out infinite;
      }

      [data-testid="stFileUploader"]:hover {
        border-color: var(--teal);
        box-shadow: 0 12px 40px rgba(11, 110, 106, 0.12);
      }

      [data-testid="stFileUploader"] section {
        border: none !important;
        background: transparent !important;
      }

      [data-testid="stFileUploader"] label,
      [data-testid="stFileUploader"] small,
      [data-testid="stFileUploader"] span {
        color: var(--ink-soft) !important;
        font-family: "Syne", sans-serif !important;
      }

      [data-testid="stFileUploader"] button {
        background: var(--ink) !important;
        color: #f7fbfa !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: "Syne", sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em;
        padding: 0.45rem 1.1rem !important;
      }

      [data-testid="stFileUploader"] button:hover {
        background: var(--teal-deep) !important;
      }

      .cs-demos-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--ink-soft);
        text-align: center;
        margin: 1.6rem 0 0.65rem;
      }

      div[data-testid="stHorizontalBlock"] button {
        border-radius: 11px !important;
        border: 1px solid var(--line) !important;
        background: rgba(255,255,255,0.7) !important;
        color: var(--ink) !important;
        font-family: "Syne", sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.78rem !important;
        margin-top: 0.35rem !important;
      }

      .cs-frame {
        position: relative;
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid var(--line);
        background: #0a1c1f;
        animation: rise 0.6s ease-out both;
        aspect-ratio: 1;
      }

      .cs-frame img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
        filter: saturate(1.08) contrast(1.04);
      }

      .cs-scan {
        position: absolute;
        left: 0; right: 0;
        height: 32%;
        top: 0;
        background: linear-gradient(
          180deg,
          transparent 0%,
          rgba(120, 230, 220, 0.18) 45%,
          transparent 100%
        );
        animation: scan 1.7s ease-in-out infinite;
        pointer-events: none;
      }

      .cs-result {
        margin-top: 0.25rem;
        padding: 2rem 1.4rem;
        border-radius: 22px;
        text-align: center;
        animation: result-in 0.65s cubic-bezier(0.22, 1, 0.36, 1) both;
      }

      .cs-result.yes {
        background: var(--alert-bg);
        border: 1px solid rgba(180, 35, 24, 0.22);
        color: var(--alert);
      }

      .cs-result.no {
        background: var(--clear-bg);
        border: 1px solid rgba(6, 118, 71, 0.22);
        color: var(--clear);
      }

      .cs-result-kicker {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        opacity: 0.75;
        margin-bottom: 0.5rem;
      }

      .cs-result-title {
        font-family: "Syne", sans-serif;
        font-weight: 800;
        font-size: clamp(1.7rem, 3.5vw, 2.35rem);
        letter-spacing: -0.03em;
        line-height: 1.1;
        margin: 0;
      }

      .cs-meter {
        margin: 1.25rem auto 0;
        max-width: 260px;
        height: 6px;
        border-radius: 99px;
        background: rgba(10, 28, 31, 0.08);
        overflow: hidden;
      }

      .cs-meter > i {
        display: block;
        height: 100%;
        border-radius: 99px;
      }

      .cs-result.yes .cs-meter > i { background: var(--alert); }
      .cs-result.no .cs-meter > i { background: var(--clear); }

      .cs-conf {
        font-family: "Newsreader", Georgia, serif;
        font-size: 1.05rem;
        margin-top: 0.85rem;
        opacity: 0.9;
      }

      .cs-foot {
        text-align: center;
        margin-top: 2.5rem;
        padding-top: 1.15rem;
        border-top: 1px solid var(--line);
        font-size: 0.78rem;
        color: var(--ink-soft);
      }

      .cs-status {
        text-align: center;
        font-size: 0.8rem;
        color: var(--ink-soft);
        margin-top: 0.85rem;
      }

      [data-testid="stImage"] img {
        border-radius: 12px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="cs-hero">
      <h1 class="cs-brand">CELL<span>SIGHT</span></h1>
      <p class="cs-tagline">See what the eye might miss.</p>
      <p class="cs-sub">Upload a histopathology slide. Instant Yes / No from nuclear patterns and a trained CNN.</p>
      <div class="cs-rule"></div>
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


st.markdown('<div class="cs-stage">', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Drop a slide image here",
    type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"],
    label_visibility="collapsed",
    help="PNG, JPG, TIFF, BMP, or WEBP",
)
st.markdown("</div>", unsafe_allow_html=True)

if "demo_name" not in st.session_state:
    st.session_state.demo_name = None

demo_files = sorted(DEMO_DIR.glob("*.png")) if DEMO_DIR.exists() else []

if demo_files and uploaded is None:
    st.markdown(
        '<p class="cs-demos-label">Or try a sample slide</p>',
        unsafe_allow_html=True,
    )
    cols = st.columns(len(demo_files))
    for col, path in zip(cols, demo_files):
        with col:
            thumb = load_demo(path.name)
            st.image(thumb, use_container_width=True)
            tag = "Positive" if path.name.startswith("yes") else "Clear"
            if st.button(tag, key=f"demo_{path.name}", use_container_width=True):
                st.session_state.demo_name = path.name

image: Optional[Image.Image] = None
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.session_state.demo_name = None
elif st.session_state.demo_name:
    image = load_demo(st.session_state.demo_name)

if image is None:
    if not HAS_TENSORFLOW:
        st.markdown(
            '<p class="cs-status">Feature mode active — densitometry scoring without TensorFlow.</p>',
            unsafe_allow_html=True,
        )
    elif not MODEL_PATH.exists():
        st.markdown(
            '<p class="cs-status">Model file missing. Train with <code>python3.11 train_bootstrap.py</code>.</p>',
            unsafe_allow_html=True,
        )
else:
    uri = image_to_data_uri(image)
    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.markdown(
            f"""
            <div class="cs-frame">
              <img src="{uri}" alt="Slide preview" />
              <div class="cs-scan"></div>
            </div>
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

        st.markdown(
            f"""
            <div class="cs-result {kind}">
              <div class="cs-result-kicker">{kicker} reading</div>
              <p class="cs-result-title">{title}</p>
              <div class="cs-meter"><i style="width:{pct:.1f}%"></i></div>
              <p class="cs-conf">{pct:.0f}% confidence</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <div class="cs-foot">
      CELLSIGHT · Research &amp; educational use only · Not a medical diagnosis
    </div>
    """,
    unsafe_allow_html=True,
)
