"""Entrypoint. No model training, clustering or notebook import at startup."""
from pathlib import Path
import streamlit as st

st.set_page_config(page_title='Cacao Data Mining',layout='wide')
st.sidebar.title('Cacao Data Mining')
dataset=st.session_state.get('dataset')
st.sidebar.caption(('Dataset: '+dataset['name']) if dataset else 'Upload or load a dataset in Module 1.')
st.sidebar.caption(f"Trained model configurations: {len(st.session_state.get('trained_models',{}))}")
st.sidebar.caption('Each module runs separately. Expensive tasks require a Run button.')

pages=[
    st.Page('ui/01_preprocessing.py',title='1. Upload and preprocess',default=True),
    st.Page('ui/02_statistics.py',title='2. Statistical analysis'),
    st.Page('ui/03_eda.py',title='3. Exploratory analysis'),
    st.Page('ui/04_clustering.py',title='4. Clustering'),
    st.Page('ui/05_models.py',title='5. Prediction models'),
    st.Page('ui/06_prediction_analysis.py',title='6. Prediction analysis'),
    st.Page('ui/07_dashboard.py',title='7. Decision dashboard'),
    st.Page('ui/08_forecasting.py',title='8. Forecasting'),
    st.Page('ui/09_exports.py',title='9. Export results'),
]
st.navigation(pages).run()
