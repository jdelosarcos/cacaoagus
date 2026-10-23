"""Shared UI guards and small export helpers; no analysis runs here."""
import streamlit as st

def require_dataset():
    dataset=st.session_state.get('dataset')
    if dataset is None:
        st.info('Open Module 1, upload CacaoY.csv or load the sample, and run preprocessing.')
        st.stop()
    return dataset

def require_models():
    dataset=require_dataset()
    trained=st.session_state.get('trained_models',{})
    if not trained:
        st.info('Train at least one model in Module 5 first. Opening this page does not start training.')
        st.stop()
    return dataset,trained

def activate_dataset(dataset,name):
    if st.session_state.get('dataset',{}).get('id')!=dataset['id']:
        for key in ['trained_models','clusters','cluster_stability','association_result','prediction_explanations',
                    'new_predictions','feature_importance','export_bytes']:
            st.session_state.pop(key,None)
    st.session_state['dataset']={**dataset,'name':name}
    st.session_state.setdefault('trained_models',{})

def csv_download(frame,filename,label='Download CSV',key=None):
    st.download_button(label,frame.to_csv(index=False).encode('utf-8-sig'),
                       filename,'text/csv',key=key)

def configuration_label(bundle):
    return f"{bundle['name']} · {bundle['folds']}-fold CV"

def same_fold_models(trained,folds):
    return {b['name']:b for b in trained.values() if b['folds']==folds}
