"""
Enterprise PACS Workstation Styling & Theme Injection.
"""

import streamlit as st


def inject_pacs_theme() -> None:
    """Injects high-end enterprise PACS workstation CSS styling."""
    st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0;
    }

    .pacs-header {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
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

    .acquisition-box {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }

    .stage-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        transition: transform 0.15s ease, border-color 0.15s ease;
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
