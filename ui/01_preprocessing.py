from pathlib import Path
import streamlit as st
from modules.services import preprocess
from ui.common import activate_dataset,csv_download

st.title('1. Upload and preprocess')
st.write('Load the survey once. Other modules use the prepared dataset without rerunning all analyses.')
upload=st.file_uploader('Upload CacaoY.csv or its canonical preprocessed CSV',type=['csv'],key='survey_upload')
col1,col2=st.columns(2)
run=col1.button('Preprocess uploaded CSV',disabled=upload is None,type='primary')
sample=col2.button('Load included CacaoY sample')
if run or sample:
    try:
        content=upload.getvalue() if run else (Path(__file__).resolve().parents[1]/'sample_data/CacaoY.csv').read_bytes()
        with st.spinner('Normalizing the survey...'):
            prepared=preprocess(content)
        activate_dataset(prepared,upload.name if run else 'CacaoY.csv (included sample)')
        st.success('Preprocessing complete. Continue to any available module from the sidebar.')
    except (ValueError,KeyError,UnicodeError,FileNotFoundError) as error:
        st.error(str(error))
dataset=st.session_state.get('dataset')
if dataset:
    clean=dataset['clean'];audit=dataset['audit']
    a,b,c=st.columns(3)
    a.metric('Retained records',len(clean));b.metric('Empty rows removed',audit['empty_rows_removed']);c.metric('Standardized cells',audit['changed_cells'])
    st.caption('Literal “None” answers are retained. Identical response profiles are flagged rather than assumed to be duplicate farmers.')
    st.dataframe(clean.head(20),width='stretch',hide_index=True)
    with st.expander('Data-quality audit'):st.json(audit)
    with st.expander('Cleaning log preview'):st.dataframe(dataset['changes'].head(100),hide_index=True)
    csv_download(clean,'CacaoY_preprocessed.csv','Download preprocessed data')
    csv_download(dataset['changes'],'cleaning_log.csv','Download complete cleaning log')
    st.info('Yield is categorical. The two upper observed ranges are pooled as 251 kg and above for modeling; original ranges remain in the data.')
