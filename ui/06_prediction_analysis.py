import json
import pandas as pd
import streamlit as st
from modules.config import FEATURES,YIELD_CLASSES
from ui.common import require_models,csv_download,configuration_label,same_fold_models

dataset,trained=require_models()
st.title('6. Prediction analysis and interpretations')
key=st.selectbox('Trained model',list(trained),format_func=lambda k:configuration_label(trained[k]))
bundle=trained[key]
mode=st.radio('Analyze',['Held-out records','New farmers CSV'],horizontal=True)
st.caption('Only the requested records are explained. Raw SVM scores are not probabilities; other model probabilities are uncalibrated outputs.')

if mode=='Held-out records':
    available=list(bundle['predictions'].source_csv_row.astype(int))
    with st.form('explain_holdout'):
        ids=st.multiselect('Source CSV rows',available,default=available[:3],max_selections=100)
        sensitivity=st.checkbox('Include local input-replacement sensitivity',value=True)
        run=st.form_submit_button('Analyze selected predictions',type='primary')
    if run:
        if not ids:st.warning('Choose at least one record.')
        else:
            from modules.explanations import explain
            records=dataset['clean'].set_index('source_csv_row').loc[ids]
            with st.spinner(f'Explaining {len(ids)} predictions only...'):
                details,checks=explain(bundle,bundle['dataset_split']['X_train'],records[FEATURES],ids,
                    actual=records.yield_class,comparisons=same_fold_models(trained,bundle['folds']),sensitivity=sensitivity)
            st.session_state['prediction_explanations']={'key':key,'source':'held-out records','details':details,'checks':checks}
            st.session_state.pop('export_bytes',None)
    saved=st.session_state.get('prediction_explanations')
else:
    template=dataset['clean'].iloc[bundle['dataset_split']['test']].sort_values('source_csv_row')[FEATURES].head(3)
    csv_download(template,'new_farmer_input_example.csv','Download input example (three real holdout profiles)')
    st.caption('The example shows the schema using existing validation records; it is not newly collected farmer data.')
    with st.form('new_farmer_prediction'):
        upload=st.file_uploader('Upload new farmers with the cleaned predictor column names',type=['csv'])
        sensitivity=st.checkbox('Include local sensitivity (up to 100 rows)',value=False)
        run=st.form_submit_button('Predict and interpret new farmers',type='primary')
    if run:
        if upload is None:st.warning('Upload the new-farmer CSV first.')
        else:
            try:
                from modules.data import prepare_new_farmers
                from modules.explanations import explain
                records=prepare_new_farmers(upload.getvalue(),FEATURES)
                if len(records)>100:raise ValueError('Use batches of at most 100 rows for interpreted predictions.')
                with st.spinner('Predicting the uploaded records...'):
                    details,checks=explain(bundle,bundle['dataset_split']['X_train'],records,list(range(1,len(records)+1)),
                        comparisons=same_fold_models(trained,bundle['folds']),sensitivity=sensitivity)
                details=details.rename(columns={'source_csv_row':'input_row'})
                st.session_state['new_predictions']={'key':key,'source':upload.name+' (actual outcomes not supplied)','details':details,'checks':checks}
                st.session_state.pop('export_bytes',None)
            except ValueError as error:st.error(str(error))
    saved=st.session_state.get('new_predictions')

if saved:
    detail=saved['details']
    st.caption(f"Last completed analysis: {saved['key']} · {saved['source']}. Change inputs and press Analyze/Predict to replace it.")
    st.dataframe(detail.drop(columns=['scores_by_class','local_sensitivities','decision_tree_path','interpretation','suggested_record_review']),hide_index=True,width='stretch')
    record_position=st.selectbox('Inspect an analyzed record',list(range(len(detail))),format_func=lambda i:f"Row {detail.iloc[i].get('source_csv_row',detail.iloc[i].get('input_row'))}")
    row=detail.iloc[record_position]
    st.subheader('Prediction interpretation')
    st.write(row.interpretation)
    st.write('Record follow-up: '+row.suggested_record_review)
    st.dataframe(pd.DataFrame(list(json.loads(row.scores_by_class).items()),columns=['yield_category','model_score']),hide_index=True)
    local=json.loads(row.local_sensitivities)
    if local:st.dataframe(pd.DataFrame(local),hide_index=True,width='stretch')
    if row.decision_tree_path:st.code(row.decision_tree_path.replace(' AND ','\nAND '),language=None)
    csv_download(detail,'prediction_analysis.csv','Download prediction analysis')
    if len(saved['checks']):csv_download(saved['checks'],'prediction_sensitivity_checks.csv','Download sensitivity checks')
    st.caption('Sensitivity replaces one field with common training values while keeping the fitted model fixed. These substitutions can form unrealistic profiles and do not establish causal or agronomic benefits.')

st.subheader('Optional global feature importance')
if st.button('Calculate importance for the selected model'):
    from modules.explanations import importance
    with st.spinner('Permuting original fields (10 repeats)...'):
        result=importance(bundle,10)
    st.session_state['feature_importance']={'key':key,'table':result}
    st.session_state.pop('export_bytes',None)
saved_importance=st.session_state.get('feature_importance')
if saved_importance:
    st.caption('Last computed for '+saved_importance['key'])
    st.dataframe(saved_importance['table'],hide_index=True)
    csv_download(saved_importance['table'],'feature_importance.csv')
