import pandas as pd
import streamlit as st
from modules.config import CORE_MODELS,YIELD_CLASSES
from ui.common import require_models,csv_download,configuration_label,same_fold_models

dataset,trained=require_models()
st.title('7. Decision-support dashboard')
st.caption('This module reads completed results. It never trains a model or generates local explanations automatically.')
key=st.selectbox('Completed model configuration',list(trained),format_func=lambda k:configuration_label(trained[k]))
bundle=trained[key];metrics=bundle['metrics']
group=same_fold_models(trained,bundle['folds'])
candidates=[b for n,b in group.items() if n not in ['Majority baseline','Cluster-only decision tree']]
if candidates:
    winner=max(candidates,key=lambda b:b['metrics']['cv_macro_f1'])
    st.info(f"Best training CV macro F1 among {len(candidates)} trained candidate(s) with {bundle['folds']} folds: {winner['name']}. This is not a comparison with untrained algorithms.")
a,b,c,d=st.columns(4)
a.metric('Survey records',len(dataset['clean']));b.metric('Full-holdout accuracy',f"{metrics['holdout_accuracy']:.1%}")
c.metric('Full-holdout macro F1',f"{metrics['holdout_macro_f1']:.3f}");d.metric('Training CV macro F1',f"{metrics['cv_macro_f1']:.3f}")
st.write('Model performance for this CV configuration')
comparison=pd.DataFrame([b['metrics'] for b in group.values()]).sort_values('cv_macro_f1',ascending=False)
st.dataframe(comparison[['model','cv_folds','cv_macro_f1','holdout_accuracy','holdout_macro_f1','holdout_balanced_accuracy']],hide_index=True,width='stretch')
baseline=group.get('Majority baseline')
if baseline:
    base=baseline['metrics']
    st.write(f"Compared with the majority baseline: accuracy difference {metrics['holdout_accuracy']-base['holdout_accuracy']:+.1%}; macro F1 difference {metrics['holdout_macro_f1']-base['holdout_macro_f1']:+.3f}.")
else:st.caption('Train the majority baseline in Module 5 for a direct comparison under this CV configuration.')
rows=bundle['predictions'].copy()
profiles=dataset['clean'][['source_csv_row','training_attendance','farm_support']]
rows=rows.merge(profiles,on='source_csv_row',how='left')
voters=[b for n,b in group.items() if n in CORE_MODELS]
if len(voters)>1:
    vote_frame=pd.DataFrame({b['name']:b['predictions'].set_index('source_csv_row').predicted_yield_class for b in voters})
    rows['models_disagree']=rows.source_csv_row.map(vote_frame.nunique(axis=1).gt(1))
else:rows['models_disagree']=False
st.caption(f"Agreement checks use {len(voters)} of four primary comparison algorithms trained with the same fold count. Missing models are not counted as agreement.")
left,right,last=st.columns(3)
location=left.selectbox('Municipality',['All']+sorted(rows.location.unique()))
actual=right.selectbox('Reported yield',['All']+YIELD_CLASSES)
outcome=last.selectbox('Prediction result',['All','Correct','Incorrect'])
if location!='All':rows=rows.loc[rows.location.eq(location)]
if actual!='All':rows=rows.loc[rows.yield_class.eq(actual)]
if outcome!='All':rows=rows.loc[rows.correct.eq(outcome=='Correct')]
a,b,c,d=st.columns(4)
a.metric('Filtered records',len(rows));b.metric('Filtered accuracy',f'{rows.correct.mean():.1%}' if len(rows) else '—')
c.metric('Observed low yield',int(rows.yield_class.eq(YIELD_CLASSES[0]).sum()));d.metric('Known errors',int((~rows.correct).sum()))
if len(rows):
    chart=pd.DataFrame({'Reported':rows.yield_class.value_counts(),'Predicted':rows.predicted_yield_class.value_counts()}).reindex(YIELD_CLASSES).fillna(0)
    st.bar_chart(chart)
    st.dataframe(rows,hide_index=True,width='stretch')
    st.subheader('Record follow-up')
    a,b,c=st.columns(3)
    a.metric('Reported no training',int(rows.training_attendance.eq('no').sum()))
    b.metric('Reported no support',int(rows.farm_support.eq('no').sum()))
    c.metric('Model disagreement',int(rows.models_disagree.sum()) if len(voters)>1 else 'Not assessed')
    def review(row):
        reasons=[]
        if row.yield_class==YIELD_CLASSES[0]:reasons.append('Confirm low-yield range, denominator and season.')
        if not row.correct:reasons.append('Review the model error against the source response.')
        if row.models_disagree:reasons.append('Inspect model disagreement.')
        if row.training_attendance=='no':reasons.append('Ask about training access and needs.')
        if row.farm_support=='no':reasons.append('Clarify reported support access and stated needs.')
        return ' '.join(reasons) or 'Verify the record and retain it for routine monitoring.'
    rows['suggested_record_review']=rows.apply(review,axis=1)
    st.dataframe(rows[['source_csv_row','yield_class','predicted_yield_class','suggested_record_review']],hide_index=True,width='stretch')
    csv_download(rows,'filtered_decision_dashboard.csv','Download filtered records and review questions')
else:st.info('No records match these filters. Change municipality, yield or prediction result.')
st.subheader('Interpretation')
st.write(f"Training versus validation macro F1 gap: {metrics['train_macro_f1']-metrics['cv_macro_f1']:.3f}. Large gaps indicate that training fit is not reproduced during cross-validation.")
st.write('Per-class precision, recall and F1 describe which categories the model recognizes.')
st.dataframe(bundle['classification_report'].head(3),hide_index=True)
if bundle['name']=='Cluster + decision tree' and 'Decision tree' in group:
    plain=group['Decision tree']['predictions']
    same=bundle['predictions'].predicted_yield_class.tolist()==plain.predicted_yield_class.tolist()
    if same:st.warning('The cluster-assisted and plain decision trees make identical holdout predictions. No added predictive benefit from clustering is observed here.')
st.caption('Review reasons are separate, nonexclusive observations. They are not a validated urgency score, funding rule or recommended agronomic treatment. Temporal and geographic generalization remain untested.')
