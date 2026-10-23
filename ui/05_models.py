import pandas as pd
import streamlit as st
from modules.services import train_model
from ui.common import require_dataset,csv_download,configuration_label

dataset=require_dataset()
st.title('5. Prediction models')
st.write('Train one model per request. Completed fits are cached, so switching pages and changing display filters do not retrain them.')
names=['Majority baseline','Decision tree','Cluster + decision tree','Cluster-only decision tree','Logistic regression','Random forest','Support vector machine']
with st.form('train_form'):
    name=st.selectbox('Model',names,index=2)
    folds=st.selectbox('Training CV folds',[5,3],format_func=lambda n:f'{n} folds'+(' — notebook comparison' if n==5 else ' — faster preview'))
    run=st.form_submit_button('Train selected model',type='primary')
if run:
    try:
        with st.spinner(f'Training {name}; no other model is running...'):
            result=train_model(dataset['id'],dataset['clean'],name,folds)
        key=f'{name}|{folds}'
        st.session_state.setdefault('trained_models',{})[key]=result
        st.session_state.pop('export_bytes',None)
        st.success(configuration_label(result)+' is ready.')
    except ValueError as error:st.error(str(error))
trained=st.session_state.get('trained_models',{})
if trained:
    comparison=pd.DataFrame([b['metrics'] for b in trained.values()]).sort_values(['cv_folds','cv_macro_f1'],ascending=[False,False])
    st.subheader('Completed fits')
    st.dataframe(comparison,hide_index=True,width='stretch')
    csv_download(comparison,'model_comparison.csv')
    key=st.selectbox('Inspect a completed fit',list(trained),format_func=lambda k:configuration_label(trained[k]))
    bundle=trained[key];metric=bundle['metrics']
    a,b,c=st.columns(3)
    a.metric('Holdout accuracy',f"{metric['holdout_accuracy']:.1%}")
    b.metric('Holdout macro F1',f"{metric['holdout_macro_f1']:.3f}")
    c.metric('Training CV macro F1',f"{metric['cv_macro_f1']:.3f}")
    st.caption(f"{metric['train_records']} training records; {metric['holdout_records']} holdout records. Best settings: {metric['best_parameters']}")
    st.write('First three held-out predictions, chosen in original source order')
    st.dataframe(bundle['predictions'].head(3),hide_index=True,width='stretch')
    st.dataframe(bundle['classification_report'],hide_index=True)
    with st.expander('Confusion matrix: reported rows, predicted columns'):st.dataframe(bundle['confusion_matrix'])
    csv_download(bundle['predictions'],'holdout_predictions.csv','Download this model’s predictions')
    st.caption('Use training CV macro F1 to compare candidates with the same fold count. CV selection scores are not independent performance estimates. Holdout scores describe this fixed validation sample.')
else:st.info('No model is trained yet. Begin with the baseline or cluster-assisted decision tree; then run logistic regression, random forest and SVM separately.')
