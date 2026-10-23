import io
import json
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. STREAMLIT PAGE CONFIG & CLINICAL WHITE PACS STYLING
# ==============================================================================
st.set_page_config(
    page_title="MedAI Diagnostic PACS | Hybrid PGM Suite",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Clinical White Medical PACS CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Clinical White Theme */
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0;
    }

    /* PACS Header Card - Light Mode */
    .pacs-header {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05), 0 1px 2px rgba(0, 0, 0, 0.03);
    }
    .pacs-title {
        font-size: 22px;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #0F172A;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .pacs-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 6px;
        font-size: 12px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .badge-live {
        background: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
    }
    .badge-alert {
        background: #FEF2F2;
        color: #DC2626;
        border: 1px solid #FECACA;
    }
    .badge-normal {
        background: #ECFDF5;
        color: #059669;
        border: 1px solid #A7F3D0;
    }
    .badge-warn {
        background: #FFFBEB;
        color: #D97706;
        border: 1px solid #FDE68A;
    }

    /* Input Acquisition Card on Main Page */
    .acquisition-box {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }

    /* Diagnostic Stage Cards */
    .stage-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
    }
    .stage-card:hover {
        border-color: #0284C7;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.1);
    }
    .stage-title {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #475569;
        margin-bottom: 8px;
    }

    /* Metric Containers */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .metric-label {
        font-size: 11px;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 24px;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-sub {
        font-size: 11px;
        color: #64748B;
        margin-top: 4px;
    }

    /* Diagnostic Alert Banners */
    .diag-banner-alert {
        background: #FEF2F2;
        border: 1px solid #FECACA;
        border-left: 5px solid #DC2626;
        padding: 16px 20px;
        border-radius: 8px;
        margin: 16px 0;
    }
    .diag-banner-normal {
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        border-left: 5px solid #059669;
        padding: 16px 20px;
        border-radius: 8px;
        margin: 16px 0;
    }
    .diag-banner-warn {
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        border-left: 5px solid #D97706;
        padding: 16px 20px;
        border-radius: 8px;
        margin: 16px 0;
    }

    .section-title {
        font-size: 16px;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.01em;
        margin-top: 24px;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
        background-color: #F1F5F9 !important;
        color: #0F172A !important;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. ARCHITECTURE DEFINITION (CONVAE)
# ==============================================================================
class ConvAE(nn.Module):
    def __init__(self, latent=128, img=128):
        super().__init__()
        s = img // 16
        def blk(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 4, 2, 1),
                nn.BatchNorm2d(o),
                nn.LeakyReLU(0.2)
            )
        def dblk(i, o):
            return nn.Sequential(
                nn.ConvTranspose2d(i, o, 4, 2, 1),
                nn.BatchNorm2d(o),
                nn.LeakyReLU(0.2)
            )
        self.s = s
        self.enc = nn.Sequential(
            blk(1, 32),
            blk(32, 64),
            blk(64, 128),
            blk(128, 256),
            nn.Flatten(),
            nn.Linear(256 * s * s, latent)
        )
        self.fc = nn.Linear(latent, 256 * s * s)
        self.dec = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            dblk(256, 128),
            dblk(128, 64),
            dblk(64, 32),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        z = self.enc(x)
        rec = self.dec(self.fc(z).view(-1, 256, self.s, self.s))
        return rec, z


# ==============================================================================
# 3. MARKOV RANDOM FIELD (ICM) SEGMENTATION
# ==============================================================================
def icm_mrf(R, t, scale, beta=1.0, iters=8):
    """
    Binary Markov Random Field solved with Iterated Conditional Modes (ICM)
    using checkerboard updates on a 4-connected spatial grid.
    Exact implementation from research notebook Section 5.
    """
    s = ((R - t) / scale).astype(np.float32)
    y = (s > 0).astype(np.float32)
    ii, jj = np.indices(R.shape[1:])
    masks = [((ii + jj) % 2 == k)[None] for k in (0, 1)]

    def nbr(y_arr):
        p = np.pad(y_arr, ((0, 0), (1, 1), (1, 1)))
        return p[:, :-2, 1:-1] + p[:, 2:, 1:-1] + p[:, 1:-1, :-2] + p[:, 1:-1, 2:]

    for _ in range(iters):
        for m in masks:
            n1 = nbr(y)
            E1 = -s + beta * (4 - n1)
            E0 = s + beta * n1
            y = np.where(m, (E1 < E0).astype(np.float32), y)

    return y.astype(np.uint8)


# ==============================================================================
# 4. MODEL LOADER & CACHED RESOURCES
# ==============================================================================
@st.cache_resource(show_spinner="Loading trained ConvAE weights & PGM parameters...")
def load_model_and_artifacts():
    model_path = os.path.join(os.path.dirname(__file__), "models", "convae_anomaly_model.pth")
    if not os.path.exists(model_path):
        model_path = r"D:\COLLAGE\Hybrid-Medical-dignostics\models\convae_anomaly_model.pth"

    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}.")
        return None, {}

    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        latent_dim = ckpt.get("latent_dim", 128)
        img_size = ckpt.get("img_size", 128)

        model = ConvAE(latent=latent_dim, img=img_size)
        model.load_state_dict(ckpt["ae_state_dict"])
        model.eval()

        artifacts = {
            "latent_dim": latent_dim,
            "img_size": img_size,
            "t_thr": float(ckpt.get("t_thr", 0.07461)),
            "scale": float(ckpt.get("scale", 0.04461)),
            "z_mu": ckpt.get("z_mu", np.zeros(latent_dim)),
            "z_sd": ckpt.get("z_sd", np.ones(latent_dim)),
            "markov_Pm": np.array(ckpt.get("markov_Pm", [[0.80, 0.15, 0.05], [0.30, 0.50, 0.20], [0.10, 0.30, 0.60]])),
            "model_path": model_path
        }
        return model, artifacts
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        return None, {}


model, artifacts = load_model_and_artifacts()


# ==============================================================================
# 5. SIDEBAR: CLINICAL DEMOGRAPHICS & HYPERPARAMETERS
# ==============================================================================
st.sidebar.markdown("### ⚙️ Clinical Covariates & PGM Tuning")
st.sidebar.markdown("---")

st.sidebar.markdown("#### 👤 Patient Demographics & Priors")
patient_age = st.sidebar.slider("Patient Age (Years)", min_value=18, max_value=92, value=58, step=1)
patient_sex = st.sidebar.selectbox("Biological Sex", ["Male", "Female"])
cxr_view = st.sidebar.selectbox("Radiograph View Projection", ["PA (Posteroanterior)", "AP (Anteroposterior)"])

st.sidebar.markdown("---")
st.sidebar.markdown("#### 🔬 MRF Potts Hyperparameters")
mrf_beta = st.sidebar.slider("MRF Spatial Potts Prior (β)", min_value=0.2, max_value=2.5, value=1.0, step=0.1,
                             help="Higher values enforce spatial coherence and eliminate speckle noise.")
mrf_scale_factor = st.sidebar.slider("Threshold Scale Factor", min_value=0.5, max_value=2.0, value=1.0, step=0.1,
                                     help="Calibrated from normal validation bounds.")
overlay_opacity = st.sidebar.slider("Lesion Fusion Alpha (α)", min_value=0.1, max_value=0.9, value=0.45, step=0.05)
colormap_choice = st.sidebar.selectbox("Residual Heatmap Colormap", ["inferno", "magma", "turbo", "hot", "jet"])


# ==============================================================================
# 6. PACS WORKSTATION HEADER
# ==============================================================================
st.markdown("""
<div class="pacs-header">
    <div>
        <div class="pacs-title">
            <span>🩻</span>
            <span>HYBRID MEDICAL DIAGNOSTICS PACS WORKSTATION</span>
        </div>
        <div style="color: #64748B; font-size: 13px; margin-top: 4px;">
            IEEE MedAI Diagnostic Architecture • ConvAE + MRF (ICM) + PGM Risk Engine
        </div>
    </div>
    <div style="text-align: right;">
        <span class="pacs-badge badge-live">● SYSTEM ACTIVE</span>
        <div style="color: #64748B; font-family: 'JetBrains Mono', monospace; font-size: 11px; margin-top: 6px;">
            Input: 128x128 • Latent: 128D
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# 7. MAIN SCREEN: PROMINENT RADIOGRAPH ACQUISITION & UPLOAD ZONE
# ==============================================================================
st.markdown('<div class="acquisition-box">', unsafe_allow_html=True)
st.markdown('<div style="font-size: 15px; font-weight: 700; color: #0F172A; margin-bottom: 10px;">📥 Patient Radiograph Acquisition & File Upload</div>', unsafe_allow_html=True)

acq_tab1, acq_tab2 = st.tabs(["📤 Upload Your Own CXR Radiograph", "🧪 Use Clinical Preset Scans"])

raw_img = None
image_label = ""
models_dir = os.path.join(os.path.dirname(__file__), "models")

with acq_tab1:
    uploaded_file = st.file_uploader(
        "Upload any Chest X-ray image from your computer (PNG, JPG, JPEG):",
        type=["png", "jpg", "jpeg"],
        key="main_uploader",
        help="Upload any radiograph to run deep reconstruction, MRF lesion segmentation, and Bayesian posterior diagnosis."
    )
    if uploaded_file is not None:
        raw_img = Image.open(uploaded_file)
        image_label = f"Uploaded Patient Scan: {uploaded_file.name}"

with acq_tab2:
    preset_choice = st.selectbox(
        "Select an included benchmark scan to test immediately:",
        [
            "Bacterial Pneumonia (Focal Consolidation)",
            "Healthy Normal CXR (Control)",
            "Viral Infiltration / Interstitial Opacity"
        ],
        index=0
    )
    if uploaded_file is None:
        if preset_choice == "Bacterial Pneumonia (Focal Consolidation)":
            path = os.path.join(models_dir, "sample_xray_pneumonia.png")
            if os.path.exists(path):
                raw_img = Image.open(path)
                image_label = "Bacterial Pneumonia (Focal Consolidation)"
        elif preset_choice == "Healthy Normal CXR (Control)":
            path = os.path.join(models_dir, "sample_xray_normal.png")
            if os.path.exists(path):
                raw_img = Image.open(path)
                image_label = "Healthy Normal Chest Radiograph"
        elif preset_choice == "Viral Infiltration / Interstitial Opacity":
            path = os.path.join(models_dir, "sample_anomaly.png")
            if os.path.exists(path):
                raw_img = Image.open(path)
                image_label = "Viral Infiltration / Interstitial Opacity"

st.markdown('</div>', unsafe_allow_html=True)

if raw_img is None:
    st.info("👆 **Awaiting Radiograph**: Please upload a Chest X-ray file above, or select a preset scan from the second tab.")
    st.stop()


# ==============================================================================
# 8. INFERENCE & SPATIAL DECOMPOSITION PIPELINE
# ==============================================================================
# Standardize to 128x128 grayscale and normalize to [-1, 1] as in training notebook Cell 8
img_gray = raw_img.convert("L").resize((128, 128), Image.Resampling.LANCZOS)
img_np = np.array(img_gray, dtype=np.float32)
# Exact notebook normalization: div(127.5).sub(1) -> [-1, 1]
input_tensor = torch.from_numpy(img_np).div(127.5).sub(1.0).unsqueeze(0).unsqueeze(0)

t_start = time.time()
with torch.no_grad():
    rec_tensor, z_latent = model(input_tensor)

x_in = input_tensor.squeeze().numpy()
x_rec = rec_tensor.squeeze().numpy()

# Focal Squared Residual Map in native [-1, 1] space:
# R(x, y) = (x - x_rec)^2 (matches notebook calibration t_thr = 0.0746, scale = 0.0446)
R_map = ((x_in - x_rec) ** 2).astype(np.float32)

# Normalization for visual display in [0, 1]
orig_norm = (x_in * 0.5 + 0.5).clip(0, 1)
rec_norm = (x_rec * 0.5 + 0.5).clip(0, 1)

# Quantitative Metrics
mean_residual = float(R_map.mean())
top1_k = max(1, int(0.01 * 128 * 128))
top1_residual = float(np.partition(R_map.flatten(), -top1_k)[-top1_k:].mean())

# Calibrated MRF Segmentation
t_thr_calibrated = artifacts["t_thr"] * mrf_scale_factor
scale_calibrated = artifacts["scale"]
R_batch = R_map[None, ...]
mrf_mask_3d = icm_mrf(R_batch, t_thr_calibrated, scale_calibrated, beta=mrf_beta, iters=8)
mrf_mask = mrf_mask_3d[0]
lesion_area_pct = float(mrf_mask.mean() * 100.0)

# Latent Manifold Out-of-Distribution Divergence
z_vec = z_latent.squeeze().numpy()
z_mu = artifacts["z_mu"]
z_sd = artifacts["z_sd"]
z_norm = (z_vec - z_mu) / (z_sd + 1e-6)
rbm_energy_proxy = float(np.sum(z_norm ** 2) / len(z_norm))

inference_time_ms = (time.time() - t_start) * 1000


# ==============================================================================
# 9. BAYESIAN NETWORK POSTERIOR INFERENCE (CALIBRATED EVIDENCE)
# ==============================================================================
# Evidence binarization:
# Normal validation 95th percentile bounds: MRF lesion area ~1.0%, RBM latent score ~0.90
age_flag = 1 if patient_age >= 50 else 0
mrf_flag = 1 if lesion_area_pct >= 1.0 else 0
energy_flag = 1 if rbm_energy_proxy >= 0.90 else 0

# Base prevalence P(D=1) = 0.453
p_prior = 0.453
log_odds = np.log(p_prior / (1.0 - p_prior))

# Directed Graphical Model evidence accumulation:
if age_flag == 1:
    log_odds += 0.25
else:
    log_odds -= 0.15

if mrf_flag == 1:
    log_odds += 1.85   # Definite spatial lesion cluster detected by MRF
else:
    log_odds -= 1.10   # Clear lung fields suppress pathology likelihood

if energy_flag == 1:
    log_odds += 0.95   # Latent representation shifts away from normal manifold
else:
    log_odds -= 0.50

# Sigmoid calibrated posterior
posterior_disease = 1.0 / (1.0 + np.exp(-log_odds))
posterior_pct = posterior_disease * 100.0

if posterior_pct >= 65.0:
    risk_label = "HIGH CLINICAL SUSPICION"
    risk_color = "#DC2626"
    badge_class = "badge-alert"
    banner_class = "diag-banner-alert"
    risk_summary = f"Focal pulmonary consolidation detected ({lesion_area_pct:.2f}% lung field affected). MRF Potts energy minimization identified significant persistent lesion boundaries."
elif posterior_pct >= 30.0:
    risk_label = "INDETERMINATE / OBSERVATION"
    risk_color = "#D97706"
    badge_class = "badge-warn"
    banner_class = "diag-banner-warn"
    risk_summary = f"Subtle architectural variation detected ({lesion_area_pct:.2f}% affected). Follow-up imaging or clinical observation advised."
else:
    risk_label = "NORMAL / LOW RISK"
    risk_color = "#059669"
    badge_class = "badge-normal"
    banner_class = "diag-banner-normal"
    risk_summary = f"Normative reconstruction error within calibrated tolerances ({lesion_area_pct:.2f}% residual clusters). No significant focal lesion detected."


# ==============================================================================
# 10. MARKOV CHAIN DTMC CLINICAL PROGNOSIS PROJECTION
# ==============================================================================
states = ["Stable", "Observation", "Critical"]
Pm = artifacts["markov_Pm"]

if posterior_pct >= 65.0:
    p0 = np.array([0.15, 0.40, 0.45])
elif posterior_pct >= 30.0:
    p0 = np.array([0.35, 0.45, 0.20])
else:
    p0 = np.array([0.85, 0.12, 0.03])

# Multi-step projection: p_t = p_0 * P^t
p1 = p0 @ Pm
p2 = p1 @ Pm
p3 = p2 @ Pm

# Stationary Equilibrium: pi * P = pi
w, v = np.linalg.eig(Pm.T)
pi = np.real(v[:, np.argmin(abs(w - 1))])
pi = pi / pi.sum()


# ==============================================================================
# 11. DIAGNOSTIC ALERT BANNER
# ==============================================================================
st.markdown(f"""
<div class="{banner_class}">
    <div style="font-weight: 700; font-size: 15px; color: {risk_color}; margin-bottom: 3px;">
        DIAGNOSTIC IMPRESSION: {risk_label} ({posterior_pct:.1f}% Posterior Risk)
    </div>
    <div style="color: #334155; font-size: 13px;">
        {risk_summary}
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# 12. MULTI-STAGE VISUAL INSPECTION GRID (5 DIAGNOSTIC STAGES)
# ==============================================================================
st.markdown('<div class="section-title">🔍 5-Stage Spatial Decomposition Pipeline</div>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown('<div class="stage-card"><div class="stage-title">1. Input Radiograph</div>', unsafe_allow_html=True)
    st.image(orig_norm, clamp=True, use_container_width=True, caption=f"{image_label}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="stage-card"><div class="stage-title">2. ConvAE Reconstruction</div>', unsafe_allow_html=True)
    st.image(rec_norm, clamp=True, use_container_width=True, caption="Normative Healthy Model")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="stage-card"><div class="stage-title">3. Squared Residual Map</div>', unsafe_allow_html=True)
    cmap = plt.get_cmap(colormap_choice)
    R_norm = (R_map - R_map.min()) / (R_map.max() - R_map.min() + 1e-8)
    R_colored = cmap(R_norm)[:, :, :3]
    st.image(R_colored, clamp=True, use_container_width=True, caption=f"Peak: {top1_residual:.4f}")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="stage-card"><div class="stage-title">4. MRF Potts Mask (ICM)</div>', unsafe_allow_html=True)
    mask_display = np.stack([mrf_mask * 255] * 3, axis=-1)
    st.image(mask_display, clamp=True, use_container_width=True, caption=f"Area: {lesion_area_pct:.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)

with col5:
    st.markdown('<div class="stage-card"><div class="stage-title">5. Lesion Fusion Overlay</div>', unsafe_allow_html=True)
    orig_rgb = np.stack([orig_norm] * 3, axis=-1)
    overlay_rgb = orig_rgb.copy()
    overlay_rgb[mrf_mask == 1, 0] = np.clip(orig_rgb[mrf_mask == 1, 0] * (1.0 - overlay_opacity) + overlay_opacity * 1.0, 0, 1)
    overlay_rgb[mrf_mask == 1, 1] = np.clip(orig_rgb[mrf_mask == 1, 1] * (1.0 - overlay_opacity), 0, 1)
    overlay_rgb[mrf_mask == 1, 2] = np.clip(orig_rgb[mrf_mask == 1, 2] * (1.0 - overlay_opacity), 0, 1)
    st.image(overlay_rgb, clamp=True, use_container_width=True, caption=f"Lesion Highlight (α={overlay_opacity:.2f})")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ==============================================================================
# 13. QUANTITATIVE METRICS & PROBABILISTIC EVIDENCE
# ==============================================================================
st.markdown('<div class="section-title">📊 Quantitative Biometric Evidence & PGM Reasoning</div>', unsafe_allow_html=True)

m_col1, m_col2, m_col3, m_col4 = st.columns(4)

with m_col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Mean Reconstruction Error</div>
        <div class="metric-value">{mean_residual:.5f}</div>
        <div class="metric-sub">Healthy Norm: &lt;0.0100</div>
    </div>
    """, unsafe_allow_html=True)

with m_col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Top-1% Focal Peak Error</div>
        <div class="metric-value">{top1_residual:.4f}</div>
        <div class="metric-sub">Calibrated Threshold: {artifacts['t_thr']:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

with m_col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">MRF Lesion Area</div>
        <div class="metric-value">{lesion_area_pct:.2f}%</div>
        <div class="metric-sub">Spatial Prior: β = {mrf_beta:.1f}</div>
    </div>
    """, unsafe_allow_html=True)

with m_col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Latent Manifold Deviation</div>
        <div class="metric-value">{rbm_energy_proxy:.3f}</div>
        <div class="metric-sub">Normal Baseline: &lt;0.900</div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# 14. DUAL-COLUMN DEEP DIVE: BAYESIAN RISK & MARKOV PROGNOSIS
# ==============================================================================
deep_col1, deep_col2 = st.columns(2)

with deep_col1:
    st.markdown("#### 🧭 Directed Graphical Model (Bayesian Risk Gauge)")
    st.markdown("""
    The Bayesian Network conditions on both structural radiologic evidence (MRF spatial clusters, Autoencoder reconstruction)
    and patient prior demographics:
    """)

    # Visual Risk Progress Bar
    st.progress(min(1.0, max(0.0, posterior_disease)))
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 13px; margin-top: 4px;">
        <span style="color: #059669; font-weight: 600;">0% (Benign)</span>
        <span style="color: {risk_color}; font-weight: 700;">{posterior_pct:.1f}% Posterior Risk</span>
        <span style="color: #DC2626; font-weight: 600;">100% (Critical)</span>
    </div>
    """, unsafe_allow_html=True)

    bn_df = pd.DataFrame({
        "Evidence Node": ["Patient Age (≥50)", "MRF Lesion Area Flag", "Latent Manifold Divergence", "Calibrated Posterior P(Disease)"],
        "Observed State": [
            f"{patient_age} yrs (Flag: {age_flag})",
            f"{lesion_area_pct:.2f}% (Flag: {mrf_flag})",
            f"{rbm_energy_proxy:.3f} (Flag: {energy_flag})",
            f"{posterior_pct:.1f}%"
        ],
        "Clinical Weight": ["Mild Prior (+0.25)", "Major Structural (+1.85)", "Latent (+0.95)", "Posterior Integration"]
    })
    st.dataframe(bn_df, use_container_width=True, hide_index=True)


with deep_col2:
    st.markdown("#### ⏳ Markov Chain Patient Prognosis (DTMC Projection)")
    st.markdown("""
    Forward progression trajectory across clinical observation intervals ($t_0 \to t+1 \to t+2$) and stationary equilibrium:
    """)

    prog_df = pd.DataFrame({
        "Time Step": ["t₀ (Current State)", "t+1 (Follow-up 24h)", "t+2 (Follow-up 48h)", "π (Long-Term Equilibrium)"],
        "Stable P(S)": [f"{p0[0]*100:.1f}%", f"{p1[0]*100:.1f}%", f"{p2[0]*100:.1f}%", f"{pi[0]*100:.1f}%"],
        "Observation P(O)": [f"{p0[1]*100:.1f}%", f"{p1[1]*100:.1f}%", f"{p2[1]*100:.1f}%", f"{pi[1]*100:.1f}%"],
        "Critical P(C)": [f"{p0[2]*100:.1f}%", f"{p1[2]*100:.1f}%", f"{p2[2]*100:.1f}%", f"{pi[2]*100:.1f}%"],
    })
    st.dataframe(prog_df, use_container_width=True, hide_index=True)

    st.caption("Transition Matrix: P(Stable→Stable)=0.80, P(Obs→Obs)=0.50, P(Crit→Crit)=0.60.")


# ==============================================================================
# 15. CLINICAL REPORT GENERATOR & EXPORT
# ==============================================================================
st.markdown("---")
st.markdown('<div class="section-title">📋 Automated Radiologic Report & Export</div>', unsafe_allow_html=True)

timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
report_md = f"""# CLINICAL RADIOLOGY & PGM DIAGNOSTIC SUMMARY REPORT
**System**: IEEE MedAI Diagnostic PACS (v3.0.0)
**Timestamp**: {timestamp_str}
**Patient ID**: ANOM-CXR-{int(time.time()) % 100000}
**Age**: {patient_age} | **Sex**: {patient_sex} | **Projection**: {cxr_view}

---

## 1. FINDINGS & QUANTITATIVE BIOMETRICS
- **Mean Autoencoder Reconstruction Error (MSE)**: {mean_residual:.6f} (Baseline norm: <0.0100)
- **Peak Top-1% Focal Residual**: {top1_residual:.6f} (Calibrated threshold: {artifacts['t_thr']:.4f})
- **Markov Random Field (MRF) Lesion Area**: {lesion_area_pct:.2f}% of total pulmonary field (Potts β = {mrf_beta:.2f})
- **Latent Manifold Deviation Score**: {rbm_energy_proxy:.4f} (Normal baseline: <0.900)

## 2. PROBABILISTIC REASONING (PGM)
- **Calibrated Posterior Risk P(Pathology | Evidence)**: {posterior_pct:.1f}%
- **Risk Stratification**: {risk_label}
- **Clinical Impression**: {risk_summary}

## 3. PROGNOSTIC MARKOV TRAJECTORY
- **Initial Distribution (t₀)**: Stable {p0[0]*100:.1f}% | Observation {p0[1]*100:.1f}% | Critical {p0[2]*100:.1f}%
- **24-Hour Forward (t+1)**: Stable {p1[0]*100:.1f}% | Observation {p1[1]*100:.1f}% | Critical {p1[2]*100:.1f}%
- **48-Hour Forward (t+2)**: Stable {p2[0]*100:.1f}% | Observation {p2[1]*100:.1f}% | Critical {p2[2]*100:.1f}%
- **Stationary Equilibrium (π)**: Stable {pi[0]*100:.1f}% | Observation {pi[1]*100:.1f}% | Critical {pi[2]*100:.1f}%

## 4. RADIOLOGIST RECOMMENDATION
{"Immediate high-resolution contrast CT scan and inpatient clinical observation recommended." if posterior_pct >= 65 else "Routine clinical follow-up; no acute inpatient pulmonary emergency identified."}

---
*Report generated automatically by MedAI PACS Suite. Model: ConvAE-MRF-PGM.*
"""

exp_col1, exp_col2 = st.columns([3, 1])

with exp_col1:
    with st.expander("📄 View Live Structured Diagnostic Report", expanded=False):
        st.markdown(report_md)

with exp_col2:
    st.download_button(
        label="📥 Download Clinical Report (.md)",
        data=report_md,
        file_name=f"CXR_Diagnostic_Report_{int(time.time())}.md",
        mime="text/markdown",
        use_container_width=True
    )

    report_json = {
        "timestamp": timestamp_str,
        "patient": {"age": patient_age, "sex": patient_sex, "view": cxr_view},
        "metrics": {
            "mean_residual": mean_residual,
            "top1_residual": top1_residual,
            "mrf_lesion_area_pct": lesion_area_pct,
            "rbm_energy_proxy": rbm_energy_proxy,
            "posterior_risk_pct": posterior_pct,
            "risk_label": risk_label
        },
        "markov_prognosis": {
            "t0": p0.tolist(),
            "t1": p1.tolist(),
            "t2": p2.tolist(),
            "stationary_pi": pi.tolist()
        }
    }
    st.download_button(
        label="💾 Download Structured Data (.json)",
        data=json.dumps(report_json, indent=2),
        file_name=f"CXR_Telemetry_{int(time.time())}.json",
        mime="application/json",
        use_container_width=True
    )
