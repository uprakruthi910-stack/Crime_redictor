"""
Urban Crime Heatmap Predictor
Main Streamlit Application Entry Point
"""
 
import streamlit as st
 
st.set_page_config(
    page_title="CrimeScope — Urban Crime Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/yourusername/urban-crime-heatmap",
        "Report a bug": "https://github.com/yourusername/urban-crime-heatmap/issues",
        "About": "# CrimeScope\nUrban Crime Heatmap Predictor — Powered by ML & Open Data",
    },
)
 
from pages.dashboard import render_dashboard
 
render_dashboard()
