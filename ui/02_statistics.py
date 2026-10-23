import streamlit as st
from modules.config import COLUMNS
from modules.services import describe,association_analysis
from ui.common import require_dataset,csv_download

dataset=require_dataset();clean=dataset['clean']
st.title('2. Statistical analysis')
freq=describe(clean)
variable=st.selectbox('Frequency table',COLUMNS+['yield_class'],index=COLUMNS.index('yield_original'))
st.dataframe(freq.loc[freq.variable.eq(variable)],hide_index=True,width='stretch')
st.caption('Age, farm area and yield are ranges; their midpoints are not treated as measured numeric values.')
st.write('Reported training attendance counts')
st.dataframe(clean.training_count.describe().to_frame('attendance_count'))
csv_download(freq,'categorical_frequencies.csv','Download all frequency tables')
st.subheader('Exploratory yield associations')
choices=[c for c in COLUMNS if c!='yield_original' and clean[c].nunique()>1]
with st.form('association_form'):
    variables=st.multiselect('Variables to test',choices,default=choices)
    permutations=st.selectbox('Label permutations',[199,999,1999,4999],index=2)
    submitted=st.form_submit_button('Run association analysis')
if submitted:
    if not variables:st.warning('Select at least one variable.')
    else:
        try:
            with st.spinner('Calculating permutation tests...'):
                result=association_analysis(clean,tuple(variables),permutations)
            st.session_state['association_result']={'table':result,'variables':variables,'permutations':permutations}
            st.session_state.pop('export_bytes',None)
        except ValueError as error:st.error(str(error))
saved=st.session_state.get('association_result')
if saved:
    st.caption(f"Saved result: {len(saved['variables'])} variables, {saved['permutations']} permutations. Changing the form does not recompute it.")
    st.dataframe(saved['table'],hide_index=True,width='stretch')
    significant=saved['table'].loc[saved['table'].q_bh<.05]
    st.write('Exploratory q < 0.05: '+(', '.join(significant.variable) if len(significant) else 'none of the tested variables'))
    csv_download(saved['table'],'yield_associations.csv')
st.caption('Permutation p-values and BH-adjusted q-values apply to the selected test family. Associations are unadjusted and do not establish causality; independence and sampling limitations remain.')
