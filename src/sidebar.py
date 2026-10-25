"""
Sidebar Clinical PACS Controls & Hyperparameter Viewport.
"""

from typing import Dict, Any
import streamlit as st


def render_pacs_sidebar() -> Dict[str, Any]:
    """
    Renders the clinical PACS sidebar with the prioritized hierarchy:
    1. Patient Demographics & Priors
    2. Windowing & Contrast (WW / WL)
    3. Multi-Modal Late-Fusion Weights
    4. Threshold & Spatial MRF Tuning
    """
    st.sidebar.markdown("### 🩻 Clinical PACS Viewport")
    st.sidebar.markdown("---")

    # 1. TOP: Patient Demographics & Priors
    st.sidebar.markdown("#### 👤 Patient Demographics & Priors")
    patient_age = st.sidebar.slider("Patient Age (Years)", min_value=18, max_value=92, value=58, step=1)
    patient_sex = st.sidebar.selectbox("Biological Sex", ["Male", "Female"])
    cxr_view = st.sidebar.selectbox("Radiograph View Projection", ["PA (Posteroanterior)", "AP (Anteroposterior)"])

    st.sidebar.markdown("---")

    # 2. MIDDLE: Windowing & Contrast (WW / WL)
    st.sidebar.markdown("#### 🔍 Windowing & Contrast (WW / WL)")
    window_level = st.sidebar.slider(
        "Window Level (Brightness)", min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="Adjusts center intensity, simulating DICOM Window Leveling."
    )
    window_width = st.sidebar.slider(
        "Window Width (Contrast)", min_value=0.2, max_value=2.0, value=1.0, step=0.1,
        help="Narrow window increases contrast in soft lung parenchyma."
    )
    invert_polarity = st.sidebar.checkbox(
        "Invert Polarity (Bones Dark / Infiltrate White)", value=False,
        help="Radiologists routinely invert polarity to spot retrocardiac opacities."
    )

    st.sidebar.markdown("---")

    # 3. MULTI-MODAL LATE-FUSION WEIGHTS (Learned Simplex)
    st.sidebar.markdown("#### ⚖️ Multi-Modal Late-Fusion Weights")
    st.sidebar.caption("Learned on validation partition to maximize AUROC ($S_{\\text{Hybrid}} = \\alpha S_v + \\beta S_m + \\gamma P_d$):")
    w_alpha = st.sidebar.slider(
        "Vision Model Weight (α)", min_value=0.0, max_value=1.0, value=0.634, step=0.005,
        help="Weight assigned to fine-tuned DenseNet-121 visual features."
    )
    w_beta = st.sidebar.slider(
        "Tuned MLP Head Weight (β)", min_value=0.0, max_value=1.0, value=0.324, step=0.005,
        help="Weight assigned to deep bottleneck representation MLP."
    )
    w_gamma = st.sidebar.slider(
        "Demographic PGM Prior (γ)", min_value=0.0, max_value=0.5, value=0.043, step=0.005,
        help="Weight assigned to Bayesian causal demographic risk prior."
    )

    w_total = w_alpha + w_beta + w_gamma
    if w_total > 0:
        w_alpha_norm = w_alpha / w_total
        w_beta_norm = w_beta / w_total
        w_gamma_norm = w_gamma / w_total
    else:
        w_alpha_norm, w_beta_norm, w_gamma_norm = 0.634, 0.324, 0.043

    st.sidebar.markdown(f"""
    <div style="font-size: 11px; color: #475569; background: #F8FAFC; border: 1px solid #E2E8F0; padding: 6px 10px; border-radius: 6px; margin-bottom: 8px; font-family: 'JetBrains Mono', monospace;">
        <b>Normalized Simplex</b>:<br>α = {w_alpha_norm:.3f} | β = {w_beta_norm:.3f} | γ = {w_gamma_norm:.3f} (Σ = 1.000)
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")

    # 4. LAST: Threshold & Spatial MRF Tuning
    st.sidebar.markdown("#### 🔬 Threshold & Spatial MRF Tuning")
    mrf_scale_factor = st.sidebar.slider(
        "Threshold Scale Factor", min_value=0.5, max_value=2.0, value=1.0, step=0.1,
        help="Scales calibrated focal anomaly cutoff threshold."
    )
    mrf_beta = st.sidebar.slider(
        "MRF Spatial Potts Prior (λ)", min_value=0.2, max_value=2.5, value=1.0, step=0.1,
        help="Smoothness regularizer penalizing isolated noise pixels."
    )
    overlay_opacity = st.sidebar.slider(
        "Lesion Heatmap Opacity", min_value=0.1, max_value=0.9, value=0.45, step=0.05,
        help="Alpha blending transparency of the diagnostic heatmap overlay."
    )
    colormap_choice = st.sidebar.selectbox("Residual Heatmap Colormap", ["inferno", "magma", "turbo", "hot", "jet"])
    show_bboxes = st.sidebar.checkbox("Render Lesion Bounding Boxes", value=True)

    return {
        "patient_age": patient_age,
        "patient_sex": patient_sex,
        "cxr_view": cxr_view,
        "window_level": window_level,
        "window_width": window_width,
        "invert_polarity": invert_polarity,
        "w_alpha": w_alpha,
        "w_beta": w_beta,
        "w_gamma": w_gamma,
        "w_alpha_norm": w_alpha_norm,
        "w_beta_norm": w_beta_norm,
        "w_gamma_norm": w_gamma_norm,
        "mrf_scale_factor": mrf_scale_factor,
        "mrf_beta": mrf_beta,
        "overlay_opacity": overlay_opacity,
        "colormap_choice": colormap_choice,
        "show_bboxes": show_bboxes
    }
