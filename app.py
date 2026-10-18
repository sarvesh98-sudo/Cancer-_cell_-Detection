"""CELLSIGHT — dark Streamlit UI for cancer cell detection."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from model_utils import (
    HAS_TENSORFLOW,
    MODEL_PATH,
    load_cancer_model,
    predict_cancer,
)

st.set_page_config(
    page_title="CELLSIGHT · Cancer Cell Detection",
    page_icon="⬡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Light touch CSS only — do NOT restyle the uploader button (causes double text)
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, .stApp, [class*="css"] {
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif !important;
}

.stApp {
  background:
    radial-gradient(ellipse 80% 50% at 15% -10%, rgba(168, 85, 247, 0.22), transparent 55%),
    radial-gradient(ellipse 60% 40% at 95% 10%, rgba(99, 102, 241, 0.12), transparent 50%),
    radial-gradient(ellipse 50% 30% at 50% 100%, rgba(168, 85, 247, 0.08), transparent 50%),
    #0a0a0f !important;
}

[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer, header { visibility: hidden; }

.block-container {
  max-width: 760px !important;
  padding-top: 1.5rem !important;
  padding-bottom: 3rem !important;
}

/* Dropzone shell only — never touch the Browse button internals */
[data-testid="stFileUploader"] {
  background: rgba(20, 18, 28, 0.9) !important;
  border: 1px dashed rgba(168, 85, 247, 0.45) !important;
  border-radius: 16px !important;
  padding: 0.75rem !important;
}

[data-testid="stFileUploader"] section {
  border: none !important;
  background: transparent !important;
}

[data-testid="stImage"] img {
  border-radius: 14px !important;
  border: 1px solid rgba(168, 85, 247, 0.2);
}

div[data-testid="stHorizontalBlock"] {
  align-items: stretch !important;
  gap: 1rem !important;
}

.stSpinner > div { border-top-color: #a855f7 !important; }
"""


def inject_css(css: str) -> None:
    payload = f"<style>{css}</style>"
    if hasattr(st, "html"):
        st.html(payload)
    else:
        st.markdown(payload, unsafe_allow_html=True)


inject_css(CSS)

# —— Hero ——
st.markdown(
    """
    <div style="text-align:center;padding:1.25rem 0 0.75rem;">
      <div style="
        font-family:Outfit,sans-serif;font-weight:800;
        font-size:clamp(2.4rem,8vw,3.8rem);letter-spacing:-0.04em;
        line-height:1;color:#f5f3ff;margin:0;
      ">CELL<span style="color:#c084fc;">SIGHT</span></div>
      <p style="
        font-family:Outfit,sans-serif;font-weight:500;
        font-size:1.15rem;color:#c4b5fd;margin:0.85rem auto 0.35rem;
        max-width:28ch;
      ">See what the eye might miss.</p>
      <p style="
        font-size:0.92rem;color:#a1a1aa;margin:0 auto;max-width:42ch;line-height:1.5;
      ">Upload a histopathology image for an instant Yes / No reading with score breakdown.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model(_mtime: float):
    return load_cancer_model()


def run_prediction(image: Image.Image) -> dict:
    model = None
    if HAS_TENSORFLOW and MODEL_PATH.exists():
        try:
            model = get_model(MODEL_PATH.stat().st_mtime)
        except Exception:
            model = None
    return predict_cancer(model, image)


def score_bars(feature: float, cnn: float, blended: float) -> go.Figure:
    labels = ["Nuclear density", "CNN score", "Final blend"]
    values = [feature * 100, cnn * 100, blended * 100]
    colors = ["#7c3aed", "#a855f7", "#e879f9"]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.0f}%" for v in values],
            textposition="outside",
            textfont=dict(color="#e4e4e7", size=12),
            cliponaxis=False,
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=48, t=8, b=8),
        height=200,
        xaxis=dict(
            range=[0, 115],
            showgrid=True,
            gridcolor="rgba(168,85,247,0.12)",
            tickfont=dict(color="#71717a"),
            title=None,
        ),
        yaxis=dict(tickfont=dict(color="#d4d4d8", size=12), title=None),
        showlegend=False,
    )
    return fig


def gauge_chart(confidence: float, positive: bool) -> go.Figure:
    color = "#f472b6" if positive else "#34d399"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            number={"suffix": "%", "font": {"size": 36, "color": "#f5f3ff"}},
            title={"text": "Confidence", "font": {"size": 14, "color": "#a1a1aa"}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": "#52525b",
                    "tickfont": {"color": "#71717a"},
                },
                "bar": {"color": color},
                "bgcolor": "rgba(20,18,28,0.8)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "rgba(168,85,247,0.08)"},
                    {"range": [50, 100], "color": "rgba(168,85,247,0.16)"},
                ],
                "threshold": {
                    "line": {"color": "#c084fc", "width": 2},
                    "thickness": 0.8,
                    "value": 50,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=10),
        height=230,
    )
    return fig


uploaded = st.file_uploader(
    "Upload slide image",
    type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"],
    label_visibility="collapsed",
)

image: Optional[Image.Image] = None
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")

if image is None:
    # —— Empty state: details only ——
    st.markdown(
        """
        <div style="
          margin-top:1.5rem;padding:1.35rem 1.4rem;border-radius:16px;
          background:#14121c;border:1px solid rgba(168,85,247,0.22);
        ">
          <p style="
            font-family:Outfit,sans-serif;font-weight:700;font-size:0.95rem;
            color:#e9d5ff;margin:0 0 0.65rem;letter-spacing:0.02em;
          ">How CELLSIGHT works</p>
          <p style="color:#a1a1aa;font-size:0.88rem;line-height:1.65;margin:0;">
            The app blends two signals: a <b style="color:#d8b4fe;">nuclear-density</b> readout
            from H&amp;E color cues in the center patch, and a <b style="color:#d8b4fe;">CNN</b>
            trained on histopathology-style patches. The final Yes / No uses a 65% / 35% mix.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3, gap="medium")
    cards = [
        ("01", "Upload", "PNG, JPG, TIFF, BMP, or WEBP histology image."),
        ("02", "Analyze", "Density + CNN scores fuse into one confidence."),
        ("03", "Read", "Clear Yes / No with charts — research use only."),
    ]
    for col, (num, title, body) in zip((c1, c2, c3), cards):
        with col:
            st.markdown(
                f"""
                <div style="
                  padding:1.1rem 1rem;border-radius:14px;height:100%;
                  background:#121018;border:1px solid rgba(168,85,247,0.18);
                  text-align:left;
                ">
                  <div style="color:#7c3aed;font-weight:700;font-size:0.75rem;
                    letter-spacing:0.12em;margin-bottom:0.45rem;">{num}</div>
                  <div style="font-family:Outfit,sans-serif;font-weight:700;
                    color:#f5f3ff;font-size:1rem;margin-bottom:0.35rem;">{title}</div>
                  <div style="color:#a1a1aa;font-size:0.82rem;line-height:1.45;">{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Placeholder chart so the empty state still has a graph
    demo_fig = go.Figure()
    demo_fig.add_trace(
        go.Scatter(
            x=list(range(12)),
            y=np.clip(
                0.35 + 0.08 * np.sin(np.linspace(0, 5, 12)) + np.linspace(0, 0.15, 12),
                0,
                1,
            ),
            mode="lines+markers",
            line=dict(color="#a855f7", width=2.5),
            marker=dict(size=7, color="#c084fc"),
            name="Example score trend",
        )
    )
    demo_fig.add_hline(
        y=0.5,
        line_dash="dot",
        line_color="rgba(192,132,252,0.45)",
        annotation_text="Decision threshold",
        annotation_font_color="#a1a1aa",
    )
    demo_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10),
        height=240,
        title=dict(
            text="Example confidence curve (illustrative)",
            font=dict(size=13, color="#a1a1aa"),
            x=0,
            xanchor="left",
        ),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(color="#52525b"),
            title=None,
        ),
        yaxis=dict(
            range=[0, 1],
            showgrid=True,
            gridcolor="rgba(168,85,247,0.1)",
            tickfont=dict(color="#71717a"),
            title=None,
        ),
        showlegend=False,
    )
    st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)
    st.plotly_chart(demo_fig, use_container_width=True, config={"displayModeBar": False})

else:
    with st.spinner("Reading cellular architecture…"):
        result = run_prediction(image)

    positive = result["has_cancer"]
    title = "Cancer signals found" if positive else "No cancer signals"
    accent = "#f472b6" if positive else "#34d399"
    panel_bg = "rgba(244,114,182,0.08)" if positive else "rgba(52,211,153,0.08)"
    panel_border = "rgba(244,114,182,0.35)" if positive else "rgba(52,211,153,0.35)"

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.image(image, use_container_width=True, caption="Uploaded slide")
    with right:
        st.markdown(
            f"""
            <div style="
              min-height:260px;display:flex;flex-direction:column;
              align-items:center;justify-content:center;text-align:center;
              padding:1.75rem 1.25rem;border-radius:16px;
              background:{panel_bg};border:1px solid {panel_border};
            ">
              <div style="
                font-size:0.7rem;font-weight:700;letter-spacing:0.16em;
                text-transform:uppercase;color:{accent};opacity:0.9;margin-bottom:0.5rem;
              ">{"Positive" if positive else "Negative"} reading</div>
              <div style="
                font-family:Outfit,sans-serif;font-weight:800;
                font-size:clamp(1.4rem,3vw,1.85rem);color:#f5f3ff;
                letter-spacing:-0.02em;line-height:1.15;
              ">{title}</div>
              <div style="
                margin-top:0.85rem;font-size:1.05rem;color:{accent};
                font-family:Outfit,sans-serif;font-weight:600;
              ">{result['confidence'] * 100:.0f}% confidence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <p style="
          font-family:Outfit,sans-serif;font-weight:700;font-size:0.95rem;
          color:#e9d5ff;margin:1.75rem 0 0.35rem;
        ">Score breakdown</p>
        <p style="color:#71717a;font-size:0.82rem;margin:0 0 0.5rem;">
          Nuclear density (65%) + CNN (35%) → final decision at 50% threshold.
        </p>
        """,
        unsafe_allow_html=True,
    )

    g1, g2 = st.columns(2, gap="large")
    with g1:
        st.plotly_chart(
            gauge_chart(result["confidence"], positive),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with g2:
        st.plotly_chart(
            score_bars(
                float(result.get("feature_score", 0)),
                float(result.get("cnn_score", 0)),
                float(result.get("raw_score", result["confidence"])),
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # Detail metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Verdict", "Yes" if positive else "No")
    m2.metric("Final", f"{result['raw_score'] * 100:.1f}%")
    m3.metric("Density", f"{result['feature_score'] * 100:.1f}%")
    m4.metric("CNN", f"{result['cnn_score'] * 100:.1f}%")

    with st.expander("What these numbers mean", expanded=False):
        st.markdown(
            """
**Nuclear density** — estimates crowded / hematoxylin-stained nuclei in the center region  
(PatchCamelyon-style focus).

**CNN score** — output of the trained convolutional network when TensorFlow is available.

**Final blend** — `0.65 × density + 0.35 × CNN`. Values ≥ 50% → **Yes**.

This tool is for **research and education only** — not a clinical diagnosis.
            """
        )

st.markdown(
    """
    <div style="
      text-align:center;margin-top:2.5rem;padding-top:1.1rem;
      border-top:1px solid rgba(168,85,247,0.18);
      font-size:0.75rem;color:#71717a;
    ">
      CELLSIGHT · Research &amp; educational use only · Not a medical diagnosis
    </div>
    """,
    unsafe_allow_html=True,
)
