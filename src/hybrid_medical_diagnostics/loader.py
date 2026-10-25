"""
Model, weights, and benchmark artifact loading routines.
"""

import os
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torchvision

from .models import ConvAE, build_tuned_mlp_head


def get_default_models_dir() -> str:
    """Resolves models directory path with fallbacks."""
    # Priority 1: Relative to project root
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidate = os.path.join(base_dir, "models")
    if os.path.exists(candidate):
        return candidate
    
    # Priority 2: Absolute known workspace path
    known_path = r"D:\COLLAGE\Hybrid-Medical-dignostics\models"
    if os.path.exists(known_path):
        return known_path

    return candidate


@st.cache_resource(show_spinner="Loading trained ConvAE & Fine-Tuned DenseNet-121...")
def load_model_and_artifacts(models_dir: Optional[str] = None) -> Tuple[Optional[ConvAE], Dict[str, Any]]:
    """
    Loads all trained PyTorch models, weights, normalization constants, and clinical benchmarks.
    """
    if models_dir is None:
        models_dir = get_default_models_dir()

    model_path = os.path.join(models_dir, "convae_anomaly_model.pth")
    dn_path = os.path.join(models_dir, "best_finetuned_densenet.pt")
    mlp_path = os.path.join(models_dir, "best_tuned_mlp_head.pth")
    csv_path = os.path.join(models_dir, "final_080_leaderboard.csv")
    roc_path = os.path.join(models_dir, "final_roc_curves.png")

    model = None
    artifacts: Dict[str, Any] = {
        "latent_dim": 128,
        "img_size": 128,
        "t_thr": 0.07461,
        "scale": 0.04461,
        "z_mu": np.zeros(128),
        "z_sd": np.ones(128),
        "markov_Pm": np.array([[0.80, 0.15, 0.05], [0.30, 0.50, 0.20], [0.10, 0.30, 0.60]]),
        "model_path": model_path,
        "dn_model": None,
        "mlp_model": None,
        "feat_extractor": None,
        "mlp_mu": np.zeros(1024),
        "mlp_sd": np.ones(1024),
        "leaderboard_df": None,
        "roc_path": roc_path if os.path.exists(roc_path) else None
    }

    # 1. Load ConvAE Anomaly Detection Model
    if os.path.exists(model_path):
        try:
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            latent_dim = ckpt.get("latent_dim", 128)
            img_size = ckpt.get("img_size", 128)
            model = ConvAE(latent=latent_dim, img=img_size)
            model.load_state_dict(ckpt["ae_state_dict"])
            model.eval()
            artifacts["t_thr"] = float(ckpt.get("t_thr", 0.07461))
            artifacts["scale"] = float(ckpt.get("scale", 0.04461))
            artifacts["z_mu"] = ckpt.get("z_mu", np.zeros(latent_dim))
            artifacts["z_sd"] = ckpt.get("z_sd", np.ones(latent_dim))
        except Exception as e:
            st.warning(f"Note on ConvAE: {e}")

    # 2. Load Fine-Tuned DenseNet-121 Vision Backbone
    if os.path.exists(dn_path):
        try:
            dn = torchvision.models.densenet121(weights=None)
            dn.classifier = nn.Linear(1024, 1)
            dn_state = torch.load(dn_path, map_location="cpu", weights_only=False)
            if "model" in dn_state:
                dn.load_state_dict(dn_state["model"])
            else:
                dn.load_state_dict(dn_state)
            dn.eval()
            artifacts["dn_model"] = dn
            artifacts["dn_path"] = dn_path
        except Exception as e:
            st.warning(f"Note on DenseNet-121: {e}")

    # 2b. Load Tuned Deep MLP Head & ImageNet Feature Extractor
    if os.path.exists(mlp_path):
        try:
            mlp_ckpt = torch.load(mlp_path, map_location="cpu", weights_only=False)
            mlp = build_tuned_mlp_head()
            mlp.load_state_dict(mlp_ckpt["model_state_dict"])
            mlp.eval()
            artifacts["mlp_model"] = mlp
            artifacts["mlp_mu"] = mlp_ckpt.get("mu", np.zeros(1024))
            artifacts["mlp_sd"] = mlp_ckpt.get("sd", np.ones(1024))

            # Load pretrained DenseNet feature extractor
            feat_ext = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
            feat_ext.eval()
            artifacts["feat_extractor"] = feat_ext
        except Exception as e:
            st.warning(f"Note on MLP Head: {e}")

    # 3. Load Clinical Leaderboard
    if os.path.exists(csv_path):
        try:
            artifacts["leaderboard_df"] = pd.read_csv(csv_path)
        except Exception:
            pass

    return model, artifacts
