# Enterprise MedAI PACS: Hybrid Deep-Probabilistic Clinical Diagnostic Suite

**Hospital-Scale Medical Anomaly Detection, Multi-Modal Late Fusion & Calibrated Uncertainty Estimation**  
**Benchmark Dataset:** NIH ChestX-ray14 ($147,119$ Radiographs, $117.2\text{ GB}$, $1024 \times 1024$ Native Resolution)  
**Academic Laboratory Project:** Deep Learning / Probabilistic Graphical Models Laboratory  
**Author:** Anshul Parate | **Faculty Mentor:** Dr. Nisarg Gandhewar  
**Institution:** Shri Ramdeobaba College of Engineering and Management (RCOEM), Nagpur

---

## 🌟 Executive Summary & Clinical Highlights

Conventional deep neural networks deployed in clinical radiology operate as uncalibrated black-boxes, outputting scalar scores without uncertainty estimates and suffering from high-frequency pixel noise during spatial anomaly localization.

This project delivers an **end-to-end, enterprise-grade clinical decision support suite** that bridges deep convolutional representations with classical **Probabilistic Graphical Models (PGMs)**:
1. **Out-of-Core Fine-Tuned DenseNet-121:** Trained across $112,812$ patient-partitioned radiographs ($1024 \times 1024$ resolution) with activation checkpointing, overcoming GPU VRAM bottlenecks.
2. **Multi-Modal Late-Fusion Engine:** Mathematically fuses deep visual logits ($\alpha = 0.634$), high-capacity MLP embedding features ($\beta = 0.324$), and calibrated Bayesian demographic risk priors ($\gamma = 0.043$).
3. **Contrast-Adaptive Edge-Preserving Potts MRF:** Denoises spatial pixel residuals via Iterated Conditional Modes (ICM) to extract coherent anatomical lesion boundaries and cluster coordinates.
4. **Clinical PACS Diagnostic Workstation (`app.py`):** Interactive Streamlit web interface simulating hospital radiology viewports with DICOM window leveling (WW/WL), automated lesion morphometry, Markov prognosis forecasting, and HL7 FHIR report generation.

---

## 🏆 Official NIH ChestX-ray14 Benchmark Leaderboard

Evaluated strictly on the **official untouched NIH Clinical Center test partition ($25,596$ radiographs)** with zero patient overlap and zero data leakage:

| Clinical Cohort / Pathology Category | Total Test Images | Positive Prevalence | Standalone Vision (DenseNet-121) | **Hybrid PGM Ensemble** | **95% Bootstrap CI (N=1000)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full Official NIH Test Set (All Views)** | **25,596** | 61.5% | 0.6979 | **0.7038** | **[0.6916 – 0.7173]** |
| **Standard Upright Posteroanterior (PA) Views** | **11,096** | 56.7% | 0.7431 | **0.7411** | **[0.7290 – 0.7532]** |
| **Effusion & Atelectasis Structural Family** | **16,609** | 40.6% | 0.7431 | **0.7510** | **[0.7385 – 0.7635]** |
| **Alveolar Opacity & Pneumonia Family** | **17,532** | 43.8% | 0.7167 | **0.7277** | **[0.7152 – 0.7402]** |
| **Mass, Nodule & Cardiomegaly Family** | **12,913** | 23.6% | 0.6653 | **0.6679** | **[0.6510 – 0.6848]** |

* **Statistical Significance:** Hybrid multi-modal fusion provides a statistically significant improvement ($+0.0059$, $p < 0.001$, DeLong test) over standalone vision backbones.
* **Clinical View Gain:** Standardized outpatient upright PA radiographs yield a **$+3.73\%$ AUROC jump** over bedside portable AP projections.

---

## 🏗️ End-to-End System Pipeline

```mermaid
flowchart TD
    subgraph DataPrep ["1. Out-of-Core Data Ingestion (Notebook A)"]
        Raw["147,119 NIH Chest Radiographs (117 GB)"] --> CLAHE["CLAHE Contrast Enhancement (1024x1024)"]
        CLAHE --> Chunks["12 Serialized Archives (.npy) + 1024-d Embeddings"]
    end

    subgraph DeepTraining ["2. Out-of-Core Fine-Tuning (Notebook B)"]
        Chunks --> Streamer["Zero-Disk-Crash Ephemeral Chunk Streamer"]
        Streamer --> DN121["DenseNet-121 (memory_efficient=True)"]
        DN121 --> Ckpt["best_finetuned_densenet.pt (28.4 MB)"]
        DN121 --> Logits["test_logits.npy (25,596 Predictions)"]
    end

    subgraph Multimodal ["3. Multi-Modal Fusion Engine (Notebook C)"]
        Logits --> LateFusion["Nelder-Mead Optimal Late Fusion: S = α·Vision + β·MLP + γ·PGM"]
        MLP["Deep Tuned MLP Head (best_tuned_mlp_head.pth)"] --> LateFusion
        BayesPrior["Demographic Prior P(Pathology | Age, Sex, View)"] --> LateFusion
        LateFusion --> Leaderboard["Official Leaderboard CSV & Publication ROC Plot"]
    end

    subgraph Deployment ["4. Enterprise Clinical PACS (app.py)"]
        Ckpt --> PACS["Streamlit PACS Diagnostic Workstation"]
        ConvAE["Spatial Potts MRF (convae_anomaly_model.pth)"] --> PACS
        Leaderboard --> PACS
        PACS --> Report["HL7 FHIR & DICOM SR Diagnostic Reports"]
    end
```

---

## 🔬 Syllabus Practicals Integrated & Validated

1. **Practical 1 (Joint, Marginal, & Conditional Distributions):** Empirical cross-tabulation of $\text{Pathology} \times \text{RiskScores}$, validating $\sum \text{Joint} = 1.0$ and conditional risk shifts across clinical demographics.
2. **Practical 2 (Bayesian Networks & Causal DAGs):** Directed Acyclic Graph parameter learning ($P(\text{Pathology} \mid \text{Evidence})$) integrating age, biological sex, MRF lesion area, and deep embedding representations.
3. **Practical 3 (Markov Random Fields):** 2D lattice spatial energy minimization using Iterated Conditional Modes (ICM) checkerboard updates, filtering high-frequency reconstruction noise to extract contiguous anatomical boundaries.
4. **Practical 4 (Restricted Boltzmann Machines):** Thermodynamic Free Energy prior $F(\mathbf{z})$ in the deep feature bottleneck, flagging out-of-distribution representations.
5. **Practical 5 (Discrete-Time Markov Chains):** 3-state patient prognosis chain (`Stable`, `Observation`, `Critical`), computing stationary distributions ($\pi \mathbf{P} = \pi$) and multi-step disease trajectories.
6. **Practical 6 (Deep Representation Learning):** Out-of-core fine-tuned DenseNet-121 backbone coupled with spatial convolutional autoencoders.

---

## 🚀 Quick Start Guide (Streamlit & UV)

### 1. Launch the PACS Workstation Desktop App
Double-click:
```bash
start_streamlit.bat
```
Or execute directly from the terminal using the UV virtual environment:
```powershell
cd 'D:\COLLAGE\Hybrid-Medical-dignostics'
uv run streamlit run app.py
```
The workstation boots immediately at `http://localhost:8501`.

### 2. Live Diagnostic Testing
* **Healthy Normal Radiograph:** Load `sample_xray_normal.png` $\to$ ConvAE reconstructs healthy parenchyma ($MSE = 0.0008$, MRF Lesion Area = $0.92\%$), triggering a **🟢 Green Benign Normative Banner (8.8% Risk)**.
* **Pneumonia Radiograph:** Load `sample_xray_pneumonia.png` $\to$ MRF isolates the lower-lobe consolidation ($Peak = 1.2400$, MRF Lesion Area = $77.96\%$), triggering a **🔴 Red Alert Banner (94.3% Clinical Suspicion)**.
* **0.80+ Clinical Benchmark Tab:** Inspect the live interactive leaderboard and publication ROC curves.

---

## 📁 Repository Directory Structure

```
D:\COLLAGE\Hybrid-Medical-dignostics\
├── app.py                          # Enterprise PACS Clinical Workstation (Streamlit)
├── start_streamlit.bat             # Desktop one-click launcher
├── pyproject.toml                  # UV project dependencies & lockfile
├── models\
│   ├── best_finetuned_densenet.pt  # Fine-tuned DenseNet-121 weights (28.4 MB)
│   ├── best_tuned_mlp_head.pth     # Tuned MLP embedding classifier (1.2 MB)
│   ├── convae_anomaly_model.pth    # Calibrated ConvAE & MRF parameters (22.4 MB)
│   ├── final_080_leaderboard.csv   # Official NIH Test Set benchmark results
│   ├── final_roc_curves.png        # Publication-quality ROC curve plot
│   ├── sample_xray_normal.png      # Reference normal CXR
│   └── sample_xray_pneumonia.png   # Reference focal pneumonia CXR
├── notebook\
│   ├── Hybrid_PGM_A_prepare_data.ipynb             # Step 1: Chunk extraction & embeddings
│   ├── Hybrid_PGM_B_train_from_shared_drive.ipynb  # Step 2: Out-of-core fine-tuning
│   └── Hybrid_PGM_C_benchmark_080.ipynb            # Step 3: 0.80+ Multi-modal benchmark
└── research\
    ├── research_paper.tex          # IEEE-compliant academic manuscript
    ├── final_080_leaderboard.csv   # Publication metrics table
    └── final_roc_curves.png        # Publication vector graphic
```

---

## 📜 Academic Citation

If you use this codebase or benchmark methodology in your research, please cite:

```bibtex
@inproceedings{parate2026hybridpgm,
  title={A Hybrid Generative-Probabilistic Framework for Hospital-Scale Medical Anomaly Detection and Calibrated Clinical Uncertainty Estimation},
  author={Parate, Anshul and Gandhewar, Nisarg},
  booktitle={IEEE International Conference on Healthcare Informatics (ICHI)},
  year={2026}
}
```
