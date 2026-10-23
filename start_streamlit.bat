@echo off
title Hybrid Medical Diagnostics PACS (Streamlit)
cd /d "%~dp0"
echo =====================================================================
echo  Starting MedAI Diagnostic PACS Workstation (ConvAE + MRF + PGM)
echo  Environment: UV virtualenv
echo =====================================================================
echo.
uv run streamlit run app.py
pause
