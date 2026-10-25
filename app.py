"""
MedAI Diagnostic PACS Workstation | Enterprise Clinical Suite.
Entrypoint orchestrator launching modular clinical components from src/hybrid_medical_diagnostics.
"""

import sys
from pathlib import Path
import streamlit as st

# Ensure src/ is on Python search path
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hybrid_medical_diagnostics import (
    inject_pacs_theme,
    load_model_and_artifacts,
    get_default_models_dir,
    render_pacs_sidebar,
    render_pacs_header,
    render_acquisition_selector,
    render_diagnostic_banner,
    render_spatial_decomposition_grid,
    render_enterprise_tabs,
    preprocess_radiograph,
    run_convae_inference,
    run_vision_backbone,
    run_mlp_head,
    compute_fusion_and_risk,
    compute_markov_prognosis
)


# ==============================================================================
# 1. PAGE CONFIGURATION & ENTERPRISE THEME
# ==============================================================================
st.set_page_config(
    page_title="MedAI Diagnostic PACS | Enterprise Clinical Suite",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_pacs_theme()


# ==============================================================================
# 2. CACHED MODEL & ARTIFACT LOADING
# ==============================================================================
models_dir = get_default_models_dir()
model, artifacts = load_model_and_artifacts(models_dir)


# ==============================================================================
# 3. SIDEBAR VIEWPORT & CONTROLS
# ==============================================================================
sidebar_params = render_pacs_sidebar()


# ==============================================================================
# 4. MAIN WORKSTATION HEADER & RADIOGRAPH ACQUISITION
# ==============================================================================
render_pacs_header()

raw_img, image_label = render_acquisition_selector(models_dir)
if raw_img is None:
    st.info("👆 **Awaiting Radiograph**: Please upload a Chest X-ray file above, or select a preset scan from the second tab.")
    st.stop()


# ==============================================================================
# 5. DIAGNOSTIC INFERENCE PIPELINE
# ==============================================================================
# Stage A: Preprocessing & DICOM WW/WL Normalization
input_tensor, ww_img, norm_source = preprocess_radiograph(
    raw_img=raw_img,
    window_level=sidebar_params["window_level"],
    window_width=sidebar_params["window_width"],
    invert_polarity=sidebar_params["invert_polarity"],
    target_size=(artifacts["img_size"], artifacts["img_size"])
)

# Stage B: ConvAE Reconstruction & Edge-Preserving MRF
recon_data = run_convae_inference(
    model=model,
    input_tensor=input_tensor,
    artifacts=artifacts,
    mrf_scale_factor=sidebar_params["mrf_scale_factor"],
    mrf_beta=sidebar_params["mrf_beta"]
)

# Stage C: Deep Vision Backbone (Fine-Tuned DenseNet-121)
s_vision = run_vision_backbone(
    dn_model=artifacts.get("dn_model"),
    raw_img=raw_img
)

# Stage D: Tuned Deep MLP Representation Head
s_mlp = run_mlp_head(
    mlp_model=artifacts.get("mlp_model"),
    feat_ext=artifacts.get("feat_extractor"),
    raw_img=raw_img,
    artifacts=artifacts
)

# Stage E: Multi-Modal Late-Fusion & Clinical Risk Stratification
risk_data = compute_fusion_and_risk(
    s_vision=s_vision,
    s_mlp=s_mlp,
    patient_age=sidebar_params["patient_age"],
    patient_sex=sidebar_params["patient_sex"],
    cxr_view=sidebar_params["cxr_view"],
    w_alpha_norm=sidebar_params["w_alpha_norm"],
    w_beta_norm=sidebar_params["w_beta_norm"],
    w_gamma_norm=sidebar_params["w_gamma_norm"],
    lesion_area_pct=recon_data["lesion_area_pct"],
    detected_lesions_count=len(recon_data["detected_lesions"])
)

# Stage F: Discrete-Time Markov Chain (DTMC) Prognosis Rollout
prognosis_data = compute_markov_prognosis(
    posterior_pct=risk_data["posterior_pct"],
    Pm=artifacts["markov_Pm"]
)


# ==============================================================================
# 6. PRESENTATION & INTERACTIVE EXPLORATION
# ==============================================================================
# Clinical Impression Alert Banner
render_diagnostic_banner(risk_data)

# 5-Stage Spatial Decomposition Pipeline Visual Cards
render_spatial_decomposition_grid(
    recon_data=recon_data,
    colormap_choice=sidebar_params["colormap_choice"],
    overlay_opacity=sidebar_params["overlay_opacity"],
    show_bboxes=sidebar_params["show_bboxes"]
)

# 5 Enterprise Diagnostic Intelligence Tabs
render_enterprise_tabs(
    recon_data=recon_data,
    risk_data=risk_data,
    prognosis_data=prognosis_data,
    artifacts=artifacts,
    sidebar_params=sidebar_params
)
