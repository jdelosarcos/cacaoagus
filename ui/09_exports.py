import streamlit as st

st.title('9. Export completed results')
st.write('Create a ZIP only when needed. It includes completed tasks from this session and never starts missing analyses.')
available={
    'Prepared survey':bool(st.session_state.get('dataset')),
    'Statistical associations':bool(st.session_state.get('association_result')),
    'Clusters':bool(st.session_state.get('clusters')),
    'Trained configurations':len(st.session_state.get('trained_models',{})),
    'Held-out prediction explanations':bool(st.session_state.get('prediction_explanations')),
    'New-farmer predictions':bool(st.session_state.get('new_predictions')),
    'Feature importance':bool(st.session_state.get('feature_importance')),
    'Dated-series forecasts':bool(st.session_state.get('forecast_result'))}
st.json(available)
if st.button('Prepare results ZIP',type='primary'):
    if not any(available.values()):st.warning('Complete at least one analysis task first.')
    else:
        from modules.exporting import build_results_zip
        with st.spinner('Packaging completed tables...'):
            st.session_state['export_bytes']=build_results_zip(st.session_state)
        st.success('The results package is ready.')
if st.session_state.get('export_bytes'):
    st.download_button('Download last prepared results ZIP',st.session_state['export_bytes'],'Cacao_Analysis_Results.zip','application/zip')
    st.caption('Prepare the ZIP again after completing additional tasks. Only results are included; application source is supplied separately.')
