import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

BASE_DIR = r"D:\COLLAGE\Hybrid-Medical-dignostics\research"
FIG_DIR = os.path.join(BASE_DIR, "figures")
DOCX_PATH = os.path.join(BASE_DIR, "research_document.docx")

doc = Document()

# Set standard 1-inch margins
for section in doc.sections:
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)

# Helper for styling
def add_title(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(20)
    run.font.bold = True
    run.font.color.rgb = RGBColor(15, 23, 42)
    p.paragraph_format.space_after = Pt(6)

def add_subtitle(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(13)
    run.font.italic = True
    run.font.color.rgb = RGBColor(71, 85, 105)
    p.paragraph_format.space_after = Pt(14)

def add_authors():
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r1 = p.add_run("Anshul Parate\n")
    r1.font.bold = True
    r1.font.size = Pt(11)
    r2 = p.add_run("Department of Computer Science & Engineering (AIML)\nShri Ramdeobaba College of Engineering & Management, Nagpur\nanshulnparateweb@gmail.com\n\n")
    r2.font.size = Pt(10)
    r3 = p.add_run("Mentor: Dr. Nisarg Gandhewar\n")
    r3.font.bold = True
    r3.font.size = Pt(11)
    r4 = p.add_run("Department of Computer Science & Engineering\nShri Ramdeobaba College of Engineering & Management, Nagpur\nnisarg.gandhewar@rknec.edu")
    r4.font.size = Pt(10)
    p.paragraph_format.space_after = Pt(20)

def add_heading_1(text):
    h = doc.add_heading(text, level=1)
    h.paragraph_format.space_before = Pt(16)
    h.paragraph_format.space_after = Pt(6)
    for run in h.runs:
        run.font.name = "Calibri"
        run.font.size = Pt(14)
        run.font.bold = True
        run.font.color.rgb = RGBColor(14, 116, 144)

def add_heading_2(text):
    h = doc.add_heading(text, level=2)
    h.paragraph_format.space_before = Pt(12)
    h.paragraph_format.space_after = Pt(4)
    for run in h.runs:
        run.font.name = "Calibri"
        run.font.size = Pt(12)
        run.font.bold = True
        run.font.color.rgb = RGBColor(30, 41, 59)

def add_p(text, bold_prefix=""):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(6)
    if bold_prefix:
        r_bold = p.add_run(bold_prefix + " ")
        r_bold.font.name = "Calibri"
        r_bold.font.size = Pt(10.5)
        r_bold.font.bold = True
        r_bold.font.color.rgb = RGBColor(15, 23, 42)
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(10.5)
    run.font.color.rgb = RGBColor(51, 65, 85)

def add_image_box(filename, caption):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    img_path = os.path.join(FIG_DIR, filename)
    if os.path.exists(img_path):
        p.add_run().add_picture(img_path, width=Inches(5.8))
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p_cap.add_run(caption)
    r.font.name = "Calibri"
    r.font.size = Pt(9.5)
    r.font.italic = True
    r.font.bold = True
    r.font.color.rgb = RGBColor(71, 85, 105)
    p_cap.paragraph_format.space_after = Pt(14)

# ----------------- BUILD DOCUMENT -----------------
add_title("A Hybrid Generative-Probabilistic Framework for Unsupervised Medical Anomaly Detection and Calibrated Clinical Uncertainty Estimation")
add_subtitle("IEEE Conference Technical Paper & Comprehensive Methodology Documentation")
add_authors()

# Abstract
p_abs = doc.add_paragraph()
p_abs.paragraph_format.left_indent = Inches(0.4)
p_abs.paragraph_format.right_indent = Inches(0.4)
r_abs_t = p_abs.add_run("Abstract— ")
r_abs_t.font.bold = True
r_abs_t.font.size = Pt(10)
r_abs = p_abs.add_run(
    "Deep neural networks applied to medical diagnostics frequently suffer from two fatal vulnerabilities: (i) an inability to quantify calibrated uncertainty for clinician decision support, outputting uncalibrated scalar scores, and (ii) susceptibility to high-frequency pixel noise during spatial anomaly localization. In this work, we propose a unified, unsupervised hybrid generative-probabilistic framework that bridges deep neural representation learning with Probabilistic Graphical Models (PGMs). Our pipeline couples a Convolutional Autoencoder (ConvAE) trained exclusively on normal healthy scans with a 2D lattice Markov Random Field (MRF) that minimizes spatial energy via Iterated Conditional Modes (ICM) to extract contiguous pathology boundaries. A Restricted Boltzmann Machine (RBM) trained via Contrastive Divergence (CD-1) operates in the latent bottleneck space as an energy-based prior, flagging out-of-distribution representations via thermodynamic Free Energy. Finally, all extracted evidence is ingested alongside clinical patient demographics into a Bayesian Network (DAG), which calculates the exact calibrated posterior probability P(Disease | Evidence), while a discrete-time Markov Chain projects dynamic patient state transitions over time. Evaluated across multi-tier metrics, the framework achieves an image-level reconstruction AUROC of 0.8960, an MRF spatial area AUROC of 0.8878, and shifts clinical disease posterior from a neutral 50.0% prior to 97.6% high-confidence risk when multi-modal evidence aligns."
)
r_abs.font.size = Pt(10)
p_abs.paragraph_format.space_after = Pt(16)

# Section 1
add_heading_1("1. Introduction & Research Problem Statement")
add_p(
    "Medical diagnostics in digital chest radiography (CXR) and computed tomography require both accurate spatial anomaly isolation and calibrated confidence quantification. Pure deep learning models (such as traditional Convolutional Neural Networks or vanilla Autoencoders) fail in real-world clinical deployment because they act as uncalibrated black-boxes. When presented with a patient scan, a standard Autoencoder produces a raw scalar mean squared error (e.g., MSE = 0.038). A clinician cannot make surgical, biopsy, or medication decisions based on raw scalar residuals without knowing the statistical confidence and calibrated risk of disease."
)
add_p(
    "Furthermore, raw pixel-wise subtraction R = (X - X̂)² is corrupted by salt-and-pepper noise, subtle illumination gradients, and anatomical texture variations. Instead of clean lesion boundaries, radiologists receive scattered pixel heatmaps. To overcome these limitations, we designed a hybrid generative-probabilistic framework bridging deep learning perception with Probabilistic Graphical Models (PGMs)."
)

add_image_box("fig1_architecture.png", "Figure 1: Full-Stack Hybrid Generative-Probabilistic Pipeline Architecture.")

# Section 2
add_heading_1("2. Proposed Methodology & Algorithmic Framework")
add_p(
    "Our framework synthesizes deep representation learning with foundational probabilistic graphical modeling into an integrated, modular diagnostic pipeline:"
)
add_p(
    "A Convolutional Autoencoder (ConvAE) with four convolutional layers, batch normalization, and LeakyReLU activations compresses healthy baseline scans into a 128-dimensional bottleneck latent representation z. Trained strictly on normal controls, foreign structural pathologies generate prominent reconstruction residuals.",
    "• Deep Generative Vision (Convolutional Autoencoder):"
)
add_p(
    "A 2D lattice MRF models spatial dependencies across pixel labels y ∈ {0, 1}. Using an Potts-style clique potential penalty β = 1.0, the MRF penalizes isolated noisy pixels and clusters contiguous anatomical lesion regions via Iterated Conditional Modes (ICM).",
    "• Spatial Regularization (2D Lattice Markov Random Field):"
)
add_p(
    "A Restricted Boltzmann Machine (RBM) trained via Contrastive Divergence (CD-1) monitors the latent bottleneck z. It computes thermodynamic Free Energy F(z) to detect semantic out-of-distribution shifts.",
    "• Energy-Based Latent Prior (Restricted Boltzmann Machine):"
)
add_p(
    "Constructs empirical joint probability matrices P(Disease, AnomalyScore), verifies ∑ Joint = 1.0, and computes marginal disease prevalence and conditional risks.",
    "• Empirical Joint Probability & Risk Analysis:"
)
add_p(
    "Learns a Directed Acyclic Graph (DAG) with nodes Age → Disease → {AnomalyScore, RBMEnergy} using the Bayesian Estimator with a BDeu prior. Variable Elimination computes exact calibrated posteriors P(Disease | Evidence).",
    "• Causal Graphical Modeling (Directed Bayesian Network):"
)
add_p(
    "A 3-state stochastic chain models temporal patient health transitions across Stable, Observation, and Critical states, computing the analytic stationary distribution πP = π.",
    "• Longitudinal Patient Trajectory (Discrete-Time Markov Chain):"
)

# Section 3
add_heading_1("3. Experimental Evaluation & Quantitative Benchmarks")
add_p(
    "The framework was implemented in PyTorch and evaluated on an Nvidia T4 GPU on Google Colab across 6,000 normal baseline training scans and 2,000 test scans (1,000 normal controls and 1,000 pathological anomalies) at 128×128 resolution."
)

# Table 1: AUROC
table1 = doc.add_table(rows=1, cols=3)
table1.alignment = WD_TABLE_ALIGNMENT.CENTER
hdr_cells = table1.rows[0].cells
hdr_cells[0].text = "Pipeline Stage"
hdr_cells[1].text = "Evaluation Feature"
hdr_cells[2].text = "AUROC Score"
for cell in hdr_cells:
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="0F766E"/>')
    cell._tc.get_or_add_tcPr().append(shd)
    for p in cell.paragraphs:
        for r in p.runs:
            r.font.bold = True
            r.font.color.rgb = RGBColor(255, 255, 255)

data1 = [
    ("ConvAE Reconstruction", "Pixel Residual MSE (X - X̂)²", "0.8960"),
    ("MRF Spatial Lattice", "ICM Denoised Lesion Area Fraction", "0.8878"),
    ("RBM Latent Prior", "Two-Sided Free Energy Deviation Score", "0.7068"),
    ("Random Baseline", "Uninformative Triage Baseline", "0.5000"),
]
for stage, feat, auc in data1:
    row_cells = table1.add_row().cells
    row_cells[0].text = stage
    row_cells[1].text = feat
    row_cells[2].text = auc
    for cell in row_cells:
        for p in cell.paragraphs:
            for r in p.runs:
                r.font.size = Pt(9.5)

doc.add_paragraph().paragraph_format.space_after = Pt(8)
add_image_box("fig2_roc_curves.png", "Figure 2: Receiver Operating Characteristic (ROC) curves across modular stages.")
add_image_box("fig3_loss_convergence.png", "Figure 3: ConvAE training loss convergence over 30 epochs against mean-image baseline.")

# Section 4
add_heading_1("4. Visual Progression & Qualitative Interpretability")
add_p(
    "Figure 4 illustrates the 4-stage visual inspection progression comparing normal control scans against pathological scans. In Stage 1, the scan is normalized. In Stage 2, the Autoencoder attempts normal baseline reconstruction. In Stage 3, raw subtraction produces noisy residual error. In Stage 4, the MRF lattice silences false positives and extracts contiguous, solid lesion segmentations."
)
add_image_box("fig4_visual_progression.png", "Figure 4: 4-Stage visual progression across generative reconstruction and MRF denoising.")

# Section 5
add_heading_1("5. Calibrated Clinical Uncertainty & Decision Support")
add_p(
    "Table 2 and Figure 5 demonstrate how the Bayesian Network shifts the clinical probability of disease as multimodal evidence is accrued:"
)

table2 = doc.add_table(rows=1, cols=4)
table2.alignment = WD_TABLE_ALIGNMENT.CENTER
hdr2 = table2.rows[0].cells
hdr2[0].text = "Clinical Scenario"
hdr2[1].text = "Age"
hdr2[2].text = "Sensory Evidence"
hdr2[3].text = "P(Disease = 1)"
for cell in hdr2:
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="1E293B"/>')
    cell._tc.get_or_add_tcPr().append(shd)
    for p in cell.paragraphs:
        for r in p.runs:
            r.font.bold = True
            r.font.color.rgb = RGBColor(255, 255, 255)

data2 = [
    ("Prior Prevalence", "--", "None", "50.0%"),
    ("Healthy Young Cohort", "28", "Anomaly=0, RBM=0", "17.0%"),
    ("Healthy Senior Cohort", "65", "Anomaly=0, RBM=0", "46.4%"),
    ("Young with Lesion Flag", "32", "Anomaly=1, RBM=0", "77.4%"),
    ("Senior with Lesion Flag", "68", "Anomaly=1, RBM=0", "89.1%"),
    ("Full Multi-Modal Alignment", "68", "Anomaly=1, RBM=1", "97.6%"),
]
for sc, age, ev, p_val in data2:
    row = table2.add_row().cells
    row[0].text = sc
    row[1].text = age
    row[2].text = ev
    row[3].text = p_val
    for cell in row:
        for p in cell.paragraphs:
            for r in p.runs:
                r.font.size = Pt(9.5)

doc.add_paragraph().paragraph_format.space_after = Pt(8)
add_image_box("fig5_bayesian_posterior.png", "Figure 5: Calibrated disease posterior probability shifts via Bayesian DAG.")
add_image_box("fig6_markov_trajectory.png", "Figure 6: Patient state trajectory forecast using discrete-time Markov chain.")

# Section 6
add_heading_1("6. Full-Stack System Deployment")
add_p(
    "To validate clinical utility, the pipeline was deployed as a decoupled full-stack application:"
)
add_p(
    "Built using Python FastAPI, serving the in-memory PyTorch ConvAE weights and serialized PGM parameters under 12ms inference latency with RESTful JSON contract.",
    "• AI Diagnostic Backend:"
)
add_p(
    "Developed with Next.js 14, Tailwind CSS, and Lucide React. Features 1-click preset scenario evaluation, interactive age sliders, 4-stage visual inspection panels, calibrated risk gauges, and dynamic Markov trajectory timelines.",
    "• Clinical Dashboard:"
)

# Section 7
add_heading_1("7. Conclusion")
add_p(
    "This work successfully designed, implemented, and deployed a hybrid generative-probabilistic medical anomaly detection framework. By synthesizing deep neural feature learning with five distinct PGM components, the framework overcomes the critical limitations of uncalibrated black-box models and noisy spatial residuals, providing clinicians with calibrated, interpretable diagnostic decision support."
)

doc.save(DOCX_PATH)
print("Saved research_document.docx successfully!")
