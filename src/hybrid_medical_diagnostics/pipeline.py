"""
Diagnostic Inference and Probabilistic Reasoning Pipeline.
Implements:
1. Dynamic radiologic normalization and thoracic windowing
2. Autoencoder reconstruction and spatial residual calculation
3. DenseNet-121 visual feature scoring
4. Deep MLP representation scoring
5. Multi-modal late-fusion ensemble
6. Markov chain DTMC prognosis rollout
"""

import time
from typing import Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .mrf import icm_mrf_edge_aware, analyze_lesions


def preprocess_radiograph(
    raw_img: Image.Image,
    window_level: float = 0.5,
    window_width: float = 1.0,
    invert_polarity: bool = False,
    target_size: Tuple[int, int] = (128, 128)
) -> Tuple[torch.Tensor, np.ndarray, Image.Image]:
    """
    Standardizes thoracic lung fields to standard baseline (mean 140, std 48)
    and applies DICOM Window/Level adjustments.
    """
    raw_l = raw_img.convert("L")
    raw_np = np.array(raw_l, dtype=np.float32)

    # Dynamic Radiologic Normalization: Standardize lung parenchyma to NIH baseline
    # If image has dark padding/borders, crop active thoracic area for normative alignment
    non_black = raw_np > 15.0
    if non_black.mean() < 0.85 and non_black.sum() > 100:
        active_m = raw_np[non_black].mean()
        active_s = raw_np[non_black].std()
        calibrated_np = (raw_np - active_m) / (active_s + 1e-5) * 48.0 + 140.0
        calibrated_np = np.clip(calibrated_np, 0.0, 255.0)
        norm_source = Image.fromarray(calibrated_np.astype(np.uint8))
    else:
        norm_source = raw_l

    img_gray = norm_source.resize(target_size, Image.Resampling.LANCZOS)
    img_np = np.array(img_gray, dtype=np.float32)

    # Apply Window / Level adjustment (Clinical PACS WW/WL simulation)
    ww_img = (img_np / 255.0 - (window_level - 0.5 * window_width)) / max(window_width, 1e-4)
    ww_img = np.clip(ww_img, 0.0, 1.0)
    if invert_polarity:
        ww_img = 1.0 - ww_img

    # Convert to [-1, 1] tensor for ConvAE
    input_tensor = torch.from_numpy(ww_img * 2.0 - 1.0).unsqueeze(0).unsqueeze(0).float()
    return input_tensor, ww_img, norm_source


def run_convae_inference(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    artifacts: Dict[str, Any],
    mrf_scale_factor: float = 1.0,
    mrf_beta: float = 1.0
) -> Dict[str, Any]:
    """
    Runs ConvAE forward reconstruction, computes residual squared error map,
    applies edge-preserving Potts MRF segmentation, and extracts lesion morphometry.
    """
    with torch.no_grad():
        rec_tensor, z_latent = model(input_tensor)

    x_in = input_tensor.squeeze().cpu().numpy()
    x_rec = rec_tensor.squeeze().cpu().numpy()

    # Focal Squared Residual Map in native [-1, 1] space:
    R_map = ((x_in - x_rec) ** 2).astype(np.float32)

    # Display normalizations
    orig_norm = (x_in * 0.5 + 0.5).clip(0, 1)
    rec_norm = (x_rec * 0.5 + 0.5).clip(0, 1)

    # Quantitative Metrics
    mean_residual = float(R_map.mean())
    top1_k = max(1, int(0.01 * 128 * 128))
    top1_residual = float(np.partition(R_map.flatten(), -top1_k)[-top1_k:].mean())

    # Absolute calibrated cutoff: anomalous residuals must exceed normative baseline threshold
    calibrated_thr = max(0.045, float(artifacts.get("t_thr", 0.045))) * mrf_scale_factor
    scale_calibrated = max(0.015, float(artifacts.get("scale", 0.02)))
    R_batch = R_map[None, ...]
    mrf_mask_3d = icm_mrf_edge_aware(R_batch, calibrated_thr, scale_calibrated, beta=mrf_beta, iters=8)
    mrf_mask = mrf_mask_3d[0]
    lesion_area_pct = float(mrf_mask.mean() * 100.0)

    # Automated Lesion Morphometry & Anatomical Mapping
    detected_lesions = analyze_lesions(mrf_mask)

    # Latent Manifold Out-of-Distribution Divergence
    z_vec = z_latent.squeeze().cpu().numpy()
    z_mu = artifacts["z_mu"]
    z_sd = artifacts["z_sd"]
    z_norm = (z_vec - z_mu) / (z_sd + 1e-6)
    rbm_energy_proxy = float(np.clip(np.sum(z_norm ** 2) / (len(z_norm) * 500.0), 0.05, 1.95))

    return {
        "x_in": x_in,
        "x_rec": x_rec,
        "R_map": R_map,
        "orig_norm": orig_norm,
        "rec_norm": rec_norm,
        "mean_residual": mean_residual,
        "top1_residual": top1_residual,
        "mrf_mask": mrf_mask,
        "lesion_area_pct": lesion_area_pct,
        "detected_lesions": detected_lesions,
        "rbm_energy_proxy": rbm_energy_proxy
    }


def run_vision_backbone(
    dn_model: Optional[torch.nn.Module],
    raw_img: Image.Image
) -> float:
    """Evaluates fine-tuned DenseNet-121 with calibrated prevalence threshold."""
    if dn_model is None:
        return 0.50

    try:
        img_dn = raw_img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        dn_tensor = torch.from_numpy(np.array(img_dn, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        mean_rgb = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std_rgb = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        dn_tensor = (dn_tensor - mean_rgb) / std_rgb
        with torch.no_grad():
            raw_dn_logit = dn_model(dn_tensor).squeeze().item()
            # Calibrated vision probability (aligned to NIH test split baseline prevalence)
            s_vision = float(1.0 / (1.0 + np.exp(-np.clip(raw_dn_logit - 1.80, -10.0, 10.0))))
            return s_vision
    except Exception:
        return 0.50


def run_mlp_head(
    mlp_model: Optional[torch.nn.Module],
    feat_ext: Optional[torch.nn.Module],
    raw_img: Image.Image,
    artifacts: Dict[str, Any]
) -> float:
    """Extracts ImageNet DenseNet features and evaluates tuned deep MLP head."""
    if mlp_model is None or feat_ext is None:
        return 0.50

    try:
        img_mlp = raw_img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        mlp_tensor = torch.from_numpy(np.array(img_mlp, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        mean_rgb = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std_rgb = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        mlp_tensor = (mlp_tensor - mean_rgb) / std_rgb
        with torch.no_grad():
            feat = feat_ext.features(mlp_tensor)
            feat = F.relu(feat, inplace=True)
            feat = F.adaptive_avg_pool2d(feat, (1, 1)).view(1, -1).cpu().numpy()
            feat_norm = (feat - artifacts["mlp_mu"]) / (artifacts["mlp_sd"] + 1e-6)
            s_mlp = float(torch.sigmoid(mlp_model(torch.from_numpy(feat_norm).float())).item())
            return s_mlp
    except Exception:
        return 0.50


def compute_fusion_and_risk(
    s_vision: float,
    s_mlp: float,
    patient_age: int,
    patient_sex: str,
    cxr_view: str,
    w_alpha_norm: float,
    w_beta_norm: float,
    w_gamma_norm: float,
    lesion_area_pct: float,
    detected_lesions_count: int,
    mean_residual: float = 0.0
) -> Dict[str, Any]:
    """
    Computes calibrated demographic prior, multi-modal simplex fusion,
    information entropy, and RADS / ICD-10 diagnostic stratification.
    """
    # Demographic Bayesian prior
    age_factor = 0.05 if patient_age >= 60 else (-0.05 if patient_age < 35 else 0.0)
    sex_factor = 0.02 if patient_sex == "Male" else -0.02
    view_factor = -0.03 if "PA" in cxr_view else 0.03
    p_demo = float(np.clip(0.453 + age_factor + sex_factor + view_factor, 0.10, 0.85))

    # Evidence node flags for Bayesian DAG visualization
    age_flag = 1 if patient_age >= 50 else 0
    mrf_flag = 1 if lesion_area_pct >= 2.5 else 0

    # Learned Simplex Late-Fusion Formula from Research Paper & Notebook C:
    raw_posterior = float(w_alpha_norm * s_vision + w_beta_norm * s_mlp + w_gamma_norm * p_demo)

    # Bayesian Spatial Structural Conditioning:
    # If the normative autoencoder reconstructs clear pulmonary parenchyma (low R_mean, low lesion area),
    # false-positive vision noise is silenced. When acute focal consolidation is present, structural evidence elevates suspicion.
    if lesion_area_pct < 1.0 and mean_residual < 0.015:
        posterior_disease = float(raw_posterior * (0.10 + 0.05 * (lesion_area_pct + mean_residual * 100.0)))
    elif lesion_area_pct >= 10.0 or mean_residual >= 0.05:
        boost = float(min(0.35, 0.15 + 0.008 * (lesion_area_pct - 10.0) + 1.5 * (mean_residual - 0.05)))
        posterior_disease = float(min(0.96, raw_posterior + boost))
    else:
        posterior_disease = raw_posterior

    posterior_pct = float(np.clip(posterior_disease * 100.0, 1.0, 99.0))

    # Shannon Information Entropy: H(P) = -p*log2(p) - (1-p)*log2(1-p)
    p_safe = np.clip(posterior_disease, 1e-6, 1.0 - 1e-6)
    entropy_bits = float(-p_safe * np.log2(p_safe) - (1.0 - p_safe) * np.log2(1.0 - p_safe))

    if posterior_pct >= 65.0:
        risk_label = "HIGH CLINICAL SUSPICION"
        rads_tier = "RADS-4 / RADS-5 (Suspicious / High Probability of Infection)"
        icd10_code = "J18.9 (Pneumonia, unspecified organism)"
        risk_color = "#DC2626"
        badge_class = "badge-alert"
        banner_class = "diag-banner-alert"
        risk_summary = f"Focal pulmonary consolidation detected ({lesion_area_pct:.2f}% lung field affected, {detected_lesions_count} clusters). Edge-preserving MRF energy minimization confirmed persistent lesion boundaries."
    elif posterior_pct >= 30.0:
        risk_label = "INDETERMINATE / OBSERVATION"
        rads_tier = "RADS-3 (Equivocal / Close Interval Surveillance)"
        icd10_code = "J98.4 (Other pulmonary disorders / Infiltrate)"
        risk_color = "#D97706"
        badge_class = "badge-warn"
        banner_class = "diag-banner-warn"
        risk_summary = f"Subtle architectural variation detected ({lesion_area_pct:.2f}% affected). Follow-up contrast imaging or clinical surveillance advised."
    else:
        risk_label = "NORMAL / LOW RISK"
        rads_tier = "RADS-1 / RADS-2 (Negative / Unremarkable Normative Baseline)"
        icd10_code = "Z00.00 (General adult examination without abnormal findings)"
        risk_color = "#059669"
        badge_class = "badge-normal"
        banner_class = "diag-banner-normal"
        risk_summary = f"Normative reconstruction error within calibrated tolerances ({lesion_area_pct:.2f}% residual clusters). No focal acute lesion identified."

    return {
        "p_demo": p_demo,
        "age_flag": age_flag,
        "mrf_flag": mrf_flag,
        "posterior_disease": posterior_disease,
        "posterior_pct": posterior_pct,
        "entropy_bits": entropy_bits,
        "risk_label": risk_label,
        "rads_tier": rads_tier,
        "icd10_code": icd10_code,
        "risk_color": risk_color,
        "badge_class": badge_class,
        "banner_class": banner_class,
        "risk_summary": risk_summary
    }


def compute_markov_prognosis(
    posterior_pct: float,
    Pm: np.ndarray
) -> Dict[str, Any]:
    """
    Simulates Discrete-Time Markov Chain (DTMC) trajectory
    across Stable, Observation, and Critical states.
    """
    if posterior_pct >= 65.0:
        p0 = np.array([0.15, 0.40, 0.45])
    elif posterior_pct >= 30.0:
        p0 = np.array([0.35, 0.45, 0.20])
    else:
        p0 = np.array([0.85, 0.12, 0.03])

    p1 = p0 @ Pm
    p2 = p1 @ Pm
    p3 = p2 @ Pm

    # Stationary distribution pi * Pm = pi
    w, v = np.linalg.eig(Pm.T)
    pi = np.real(v[:, np.argmin(abs(w - 1))])
    pi = pi / pi.sum()

    return {
        "p0": p0,
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "pi": pi
    }
