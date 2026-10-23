import pandas as pd
import streamlit as st
from modules.config import COLUMNS,YIELD_CLASSES
from ui.common import require_dataset,csv_download

dataset=require_dataset();clean=dataset['clean']
st.title('3. Exploratory analysis')
view=st.radio('Chart',['Category distribution','Yield by municipality','Practice and yield'],horizontal=True)
if view=='Category distribution':
    variable=st.selectbox('Variable',COLUMNS,index=COLUMNS.index('yield_original'))
    counts=clean[variable].value_counts().rename('records').to_frame()
    st.bar_chart(counts)
    st.dataframe(counts.rename_axis(variable).reset_index(),hide_index=True)
elif view=='Yield by municipality':
    counts=pd.crosstab(clean.location,clean.yield_class).reindex(columns=YIELD_CLASSES,fill_value=0)
    percentages=st.checkbox('Show within-municipality percentages',value=True)
    chart=counts.div(counts.sum(axis=1),axis=0)*100 if percentages else counts
    st.bar_chart(chart)
    st.dataframe(counts)
    csv_download(counts.reset_index(),'municipality_yield_counts.csv')
else:
    practice=st.selectbox('Practice',['fertilizer','soil_management','irrigation','pruning','training_attendance','soil_fertility'])
    counts=pd.crosstab(clean[practice],clean.yield_class).reindex(columns=YIELD_CLASSES,fill_value=0)
    st.bar_chart(counts)
    st.dataframe(counts)
st.caption('Only the selected chart is generated. These survey patterns describe responses and do not establish treatment effects.')
