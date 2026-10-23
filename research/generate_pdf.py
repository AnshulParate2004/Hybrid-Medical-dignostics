import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

BASE_DIR = r"D:\COLLAGE\Hybrid-Medical-dignostics\research"
FIG_DIR = os.path.join(BASE_DIR, "figures")
PDF_PATH = os.path.join(BASE_DIR, "research_paper.pdf")

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, letter[1] - 36, "IEEE Transactions / Conference on Medical Image Computing & Uncertainty")
            self.drawRightString(letter[0] - 54, letter[1] - 36, "Parate & Gandhewar: Hybrid Anomaly Detection Framework")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(54, letter[1] - 42, letter[0] - 54, letter[1] - 42)
        
        # Footer
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 54, 36, page_text)
        self.drawString(54, 36, "Department of Computer Science & Engineering, RCOEM, Nagpur")
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 46, letter[0] - 54, 46)
        self.restoreState()

doc = SimpleDocTemplate(
    PDF_PATH,
    pagesize=letter,
    leftMargin=54,
    rightMargin=54,
    topMargin=54,
    bottomMargin=54
)

styles = getSampleStyleSheet()

# Custom styles
title_style = ParagraphStyle(
    "PaperTitle",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=18,
    leading=22,
    alignment=1,
    textColor=colors.HexColor("#0f172a"),
    spaceAfter=8
)

subtitle_style = ParagraphStyle(
    "PaperSubtitle",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=11,
    leading=14,
    alignment=1,
    textColor=colors.HexColor("#475569"),
    spaceAfter=12
)

author_style = ParagraphStyle(
    "AuthorBlock",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=9.5,
    leading=13,
    alignment=1,
    textColor=colors.HexColor("#1e293b"),
    spaceAfter=14
)

abs_style = ParagraphStyle(
    "Abstract",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=9,
    leading=12.5,
    alignment=4,
    textColor=colors.HexColor("#1e293b")
)

h1_style = ParagraphStyle(
    "Heading1",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=12,
    leading=15,
    textColor=colors.HexColor("#0f766e"),
    spaceBefore=14,
    spaceAfter=6,
    keepWithNext=True
)

h2_style = ParagraphStyle(
    "Heading2",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=10,
    leading=13,
    textColor=colors.HexColor("#1e293b"),
    spaceBefore=10,
    spaceAfter=4,
    keepWithNext=True
)

body_style = ParagraphStyle(
    "Body",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=9.2,
    leading=12.8,
    alignment=4,
    textColor=colors.HexColor("#334155"),
    spaceAfter=6
)

bullet_style = ParagraphStyle(
    "Bullet",
    parent=body_style,
    leftIndent=15,
    firstLineIndent=-10,
    spaceAfter=4
)

cap_style = ParagraphStyle(
    "Caption",
    parent=styles["Normal"],
    fontName="Helvetica-BoldOblique",
    fontSize=8.5,
    leading=11,
    alignment=1,
    textColor=colors.HexColor("#475569"),
    spaceBefore=4,
    spaceAfter=12
)

story = []

# Title & Metadata
story.append(Paragraph("A Hybrid Generative-Probabilistic Framework for Unsupervised Medical Anomaly Detection and Calibrated Clinical Uncertainty Estimation", title_style))
story.append(Paragraph("IEEE Conference Technical Paper • Academic Laboratory Project", subtitle_style))

authors_text = "<b>Anshul Parate</b><sup>1</sup>, <b>Dr. Nisarg Gandhewar</b><sup>2</sup><br/>" \
               "<sup>1,2</sup>Department of Computer Science & Engineering (AIML), Shri Ramdeobaba College of Engineering and Management, Nagpur<br/>" \
               "<i>Emails: anshulnparateweb@gmail.com, nisarg.gandhewar@rknec.edu</i>"
story.append(Paragraph(authors_text, author_style))

# Abstract Box
abs_html = "<b><i>Abstract</i>— Deep neural networks applied to medical diagnostics frequently suffer from two fatal vulnerabilities: (i) an inability to quantify calibrated uncertainty for clinician decision support, outputting uncalibrated scalar scores, and (ii) susceptibility to high-frequency pixel noise during spatial anomaly localization. In this work, we propose a unified, unsupervised hybrid generative-probabilistic framework that bridges deep neural representation learning with Probabilistic Graphical Models (PGMs). Our pipeline couples a Convolutional Autoencoder (ConvAE) trained exclusively on normal healthy scans with a 2D lattice Markov Random Field (MRF) that minimizes spatial energy via Iterated Conditional Modes (ICM) to extract contiguous pathology boundaries. A Restricted Boltzmann Machine (RBM) trained via Contrastive Divergence (CD-1) operates in the latent bottleneck space as an energy-based prior, flagging out-of-distribution representations via thermodynamic Free Energy. Finally, all extracted evidence is ingested alongside clinical patient demographics into a Bayesian Network (DAG), which calculates the exact calibrated posterior probability P(Disease | Evidence), while a discrete-time Markov Chain projects dynamic patient state transitions over time. Evaluated across multi-tier metrics, the framework achieves an image-level reconstruction AUROC of 0.8960, an MRF spatial area AUROC of 0.8878, and shifts clinical disease posterior from a neutral 50.0% prior to 97.6% high-confidence risk when multi-modal evidence aligns.</b>"

abs_table = Table([[Paragraph(abs_html, abs_style)]], colWidths=[504])
abs_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
    ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#cbd5e1")),
    ("TOPPADDING", (0, 0), (-1, -1), 8),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
]))
story.append(abs_table)
story.append(Spacer(1, 10))

# Keywords
story.append(Paragraph("<b>Keywords:</b> Medical Anomaly Detection, Convolutional Autoencoder, Markov Random Field, Restricted Boltzmann Machine, Bayesian Networks, Uncertainty Estimation, Markov Chains.", ParagraphStyle("KW", parent=body_style, fontSize=8.5, textColor=colors.HexColor("#0f172a"))))
story.append(Spacer(1, 10))

# 1. Introduction
story.append(Paragraph("1. Introduction & Research Problem", h1_style))
story.append(Paragraph(
    "Automated medical image triage requires both high spatial fidelity in anomaly segmentation and calibrated confidence quantification for healthcare practitioners. However, purely deep neural networks (such as traditional convolutional architectures or vanilla autoencoders) present critical failure modes when applied in high-stakes clinical workflows. Standard autoencoders act as uncalibrated black-boxes: when evaluated on a patient's chest radiograph, they output a scalar reconstruction loss (e.g., MSE = 0.038) without statistical risk calibration. Clinicians cannot initiate invasive biopsies or therapeutic interventions based solely on raw residual metrics without knowing the calibrated probability P(Disease | Evidence).",
    body_style
))
story.append(Paragraph(
    "Moreover, raw pixel-wise subtraction R = (X - X̂)² is severely corrupted by high-frequency spatial noise, subtle illumination gradients, and anatomical texture variations. Instead of clean, continuous anatomical lesion boundaries, radiologists receive jagged, noisy heatmaps. To resolve these challenges, we introduce an unsupervised hybrid generative-probabilistic framework uniting deep representation learning with five classical Probabilistic Graphical Models (PGMs).",
    body_style
))

# Figure 1
fig1_path = os.path.join(FIG_DIR, "fig1_architecture.png")
if os.path.exists(fig1_path):
    story.append(Image(fig1_path, width=6.8*inch, height=3.1*inch))
    story.append(Paragraph("Figure 1: Full-Stack Hybrid Generative-Probabilistic Pipeline Architecture.", cap_style))

# 2. Methodology & Syllabus Practical Mapping
story.append(Paragraph("2. Modular Methodology & Lab Practical Integration", h1_style))
story.append(Paragraph(
    "Our framework synthesizes the core practical topics of the Deep Learning & Probabilistic Graphical Models syllabus into an end-to-end clinical decision-support pipeline:",
    body_style
))
story.append(Paragraph("• <b>Practical 6 (Convolutional Autoencoder):</b> A 4-layer convolutional encoder-decoder with batch normalization and LeakyReLU activations compresses healthy baseline scans into a 32-dimensional bottleneck latent representation z. Trained strictly on normal controls, foreign structural lesions generate elevated reconstruction residuals.", bullet_style))
story.append(Paragraph("• <b>Practical 3 (Markov Random Field Spatial Lattice):</b> A 2D lattice MRF models spatial pixel dependencies via an Ising-style clique potential penalty (β=1.0). Optimized via Iterated Conditional Modes (ICM) checkerboard updates, it eliminates false-positive pixel noise to produce contiguous lesion masks.", bullet_style))
story.append(Paragraph("• <b>Practical 4 (Restricted Boltzmann Machine Latent Prior):</b> A 32-visible, 64-hidden unit RBM trained via Contrastive Divergence (CD-1) monitors the latent bottleneck z, measuring thermodynamic Free Energy F(z) to catch out-of-distribution representations.", bullet_style))
story.append(Paragraph("• <b>Practical 1 (Joint & Conditional Probability Distributions):</b> Constructs empirical cross-tabulation matrices of Disease × AnomalyScore, proves ∑ Joint = 1.0, and computes marginal prevalence and conditional risks.", bullet_style))
story.append(Paragraph("• <b>Practical 2 (Bayesian Networks & Variable Elimination):</b> Learns a DAG: Age → Disease → {AnomalyScore, RBMEnergy} using Bayesian Estimator with BDeu prior. Exact variable elimination calculates calibrated posteriors P(Disease | Evidence).", bullet_style))
story.append(Paragraph("• <b>Practical 5 (Discrete-Time Markov Chains):</b> Formulates a 3-state patient prognosis chain across Stable, Observation, and Critical states, verifying the analytic stationary distribution πP = π.", bullet_style))

# Figure 4 (Visual Progression)
fig4_path = os.path.join(FIG_DIR, "fig4_visual_progression.png")
if os.path.exists(fig4_path):
    story.append(KeepTogether([
        Image(fig4_path, width=6.8*inch, height=3.4*inch),
        Paragraph("Figure 4: 4-Stage visual progression across generative reconstruction and MRF spatial denoising.", cap_style)
    ]))

# 3. Experimental Evaluation
story.append(Paragraph("3. Quantitative Benchmarks & Experimental Results", h1_style))
story.append(Paragraph(
    "The framework was implemented in PyTorch and evaluated on an Nvidia T4 GPU (Google Colab environment). The training cohort comprised 6,000 normal baseline scans, and the independent test cohort evaluated 2,000 scans (1,000 normal controls and 1,000 pathological anomalies) at 32×32 resolution.",
    body_style
))

# Table 1: AUROC
table1_data = [
    [Paragraph("<b>Pipeline Stage</b>", ParagraphStyle("TH", parent=body_style, textColor=colors.white, fontName="Helvetica-Bold")),
     Paragraph("<b>Evaluation Feature</b>", ParagraphStyle("TH", parent=body_style, textColor=colors.white, fontName="Helvetica-Bold")),
     Paragraph("<b>AUROC Score</b>", ParagraphStyle("TH", parent=body_style, textColor=colors.white, fontName="Helvetica-Bold"))],
    [Paragraph("ConvAE Reconstruction", body_style), Paragraph("Pixel Residual MSE (X - X̂)²", body_style), Paragraph("<b>0.8960</b>", body_style)],
    [Paragraph("MRF Spatial Lattice", body_style), Paragraph("ICM Denoised Lesion Area Fraction", body_style), Paragraph("<b>0.8878</b>", body_style)],
    [Paragraph("RBM Latent Prior", body_style), Paragraph("Two-Sided Free Energy Deviation", body_style), Paragraph("<b>0.7068</b>", body_style)],
    [Paragraph("Random Baseline", body_style), Paragraph("Uninformative Triage Baseline", body_style), Paragraph("0.5000", body_style)]
]
t1 = Table(table1_data, colWidths=[160, 240, 104])
t1.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
]))
story.append(t1)
story.append(Spacer(1, 10))

# Figures 2 & 3 side by side or sequentially
fig2_path = os.path.join(FIG_DIR, "fig2_roc_curves.png")
fig3_path = os.path.join(FIG_DIR, "fig3_loss_convergence.png")
if os.path.exists(fig2_path) and os.path.exists(fig3_path):
    t_figs = Table([[Image(fig2_path, width=3.35*inch, height=2.5*inch), Image(fig3_path, width=3.35*inch, height=2.5*inch)]], colWidths=[252, 252])
    t_figs.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))
    story.append(t_figs)
    story.append(Paragraph("Figures 2 & 3: Multi-tier ROC performance curves (left) and ConvAE training loss convergence (right).", cap_style))

# 4. Bayesian Evidence Shift
story.append(Paragraph("4. Calibrated Clinical Decision Support via Bayesian DAG", h1_style))
story.append(Paragraph(
    "Table II and Figure 5 demonstrate how the Bayesian Network factorizes sensory evidence and clinical covariates to provide calibrated probability updates:",
    body_style
))

table2_data = [
    [Paragraph("<b>Clinical Scenario</b>", ParagraphStyle("TH2", parent=body_style, textColor=colors.white, fontName="Helvetica-Bold")),
     Paragraph("<b>Age</b>", ParagraphStyle("TH2", parent=body_style, textColor=colors.white, fontName="Helvetica-Bold")),
     Paragraph("<b>Sensory Evidence</b>", ParagraphStyle("TH2", parent=body_style, textColor=colors.white, fontName="Helvetica-Bold")),
     Paragraph("<b>P(Disease = 1)</b>", ParagraphStyle("TH2", parent=body_style, textColor=colors.white, fontName="Helvetica-Bold"))],
    [Paragraph("Prior Prevalence", body_style), Paragraph("--", body_style), Paragraph("None", body_style), Paragraph("50.0%", body_style)],
    [Paragraph("Healthy Young Cohort", body_style), Paragraph("28", body_style), Paragraph("Anomaly=0, RBM=0", body_style), Paragraph("<b>17.0%</b>", body_style)],
    [Paragraph("Healthy Senior Cohort", body_style), Paragraph("65", body_style), Paragraph("Anomaly=0, RBM=0", body_style), Paragraph("<b>46.4%</b>", body_style)],
    [Paragraph("Young with Lesion Flag", body_style), Paragraph("32", body_style), Paragraph("Anomaly=1, RBM=0", body_style), Paragraph("<b>77.4%</b>", body_style)],
    [Paragraph("Senior with Lesion Flag", body_style), Paragraph("68", body_style), Paragraph("Anomaly=1, RBM=0", body_style), Paragraph("<b>89.1%</b>", body_style)],
    [Paragraph("Full Multi-Modal Alignment", body_style), Paragraph("68", body_style), Paragraph("Anomaly=1, RBM=1", body_style), Paragraph("<b>97.6%</b>", body_style)],
]
t2 = Table(table2_data, colWidths=[150, 54, 200, 100])
t2.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
]))
story.append(t2)
story.append(Spacer(1, 10))

# Figures 5 & 6
fig5_path = os.path.join(FIG_DIR, "fig5_bayesian_posterior.png")
fig6_path = os.path.join(FIG_DIR, "fig6_markov_trajectory.png")
if os.path.exists(fig5_path) and os.path.exists(fig6_path):
    t_figs2 = Table([[Image(fig5_path, width=3.35*inch, height=2.3*inch), Image(fig6_path, width=3.35*inch, height=2.3*inch)]], colWidths=[252, 252])
    t_figs2.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))
    story.append(t_figs2)
    story.append(Paragraph("Figures 5 & 6: Bayesian posterior probability shifts (left) and Markov chain patient state trajectory forecast (right).", cap_style))

# 5. Conclusion
story.append(Paragraph("5. Conclusion & Deployment", h1_style))
story.append(Paragraph(
    "We presented a hybrid generative-probabilistic medical anomaly detection framework that overcomes the fundamental black-box and spatial noise vulnerabilities of deep learning. By coupling deep autoencoders with Markov random fields, RBM latent priors, Bayesian networks, and Markov chains, the framework achieves an AUROC of 0.8960 while outputting calibrated clinical probabilities (e.g., 97.6% under multi-modal alignment). The system is fully deployed with a high-performance Python FastAPI backend and an interactive Next.js 14 clinical dashboard.",
    body_style
))

# References
story.append(Paragraph("References", h1_style))
refs = [
    "[1] T. Schlegl et al., 'f-AnoGAN: Fast unsupervised anomaly detection with generative adversarial networks in medical imaging,' <i>IEEE TMI</i>, vol. 38, no. 1, pp. 30–44, 2019.",
    "[2] C. Baur et al., 'Autoencoders for unsupervised anomaly segmentation in brain MR images: A comparative study,' <i>Med. Image Anal.</i>, vol. 69, p. 101952, 2021.",
    "[3] E. Begoli et al., 'The need for uncertainty quantification in machine-assisted medical decision making,' <i>Nature Machine Intelligence</i>, vol. 1, pp. 20–23, 2019.",
    "[4] G. E. Hinton, 'Training products of experts by minimizing contrastive divergence,' <i>Neural Comput.</i>, vol. 14, no. 8, pp. 1771–1800, 2002.",
    "[5] J. Besag, 'On the statistical analysis of dirty pictures,' <i>J. R. Stat. Soc. Ser. B</i>, vol. 48, no. 3, pp. 259–279, 1986.",
    "[6] D. Koller and N. Friedman, <i>Probabilistic Graphical Models: Principles and Techniques</i>. MIT Press, 2009."
]
for r in refs:
    story.append(Paragraph(r, ParagraphStyle("Ref", parent=body_style, fontSize=8, leading=10.5, leftIndent=15, firstLineIndent=-15, spaceAfter=3)))

doc.build(story, canvasmaker=NumberedCanvas)
print("Saved research_paper.pdf successfully!")
