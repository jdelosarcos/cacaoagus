import streamlit as st
from modules.services import clustering,seed_stability
from ui.common import require_dataset,csv_download

dataset=require_dataset()
st.title('4. Categorical clustering')
st.write('K-modes compares matching categorical fields. Candidate k values are evaluated only on the training subset; held-out farmers are assigned afterward.')
if st.button('Run clustering',type='primary'):
    try:
        with st.spinner('Selecting categorical clusters on training records...'):
            result=clustering(dataset['id'],dataset['clean'])
        st.session_state['clusters']=result
        st.session_state.pop('export_bytes',None)
    except ValueError as error:st.error(str(error))
result=st.session_state.get('clusters')
if result:
    a,b,c=st.columns(3)
    a.metric('Selected k',result['k']);b.metric('Training Hamming silhouette',f"{result['silhouette']:.3f}");c.metric('Training records',result['train_records'])
    st.dataframe(result['selection'],hide_index=True)
    st.bar_chart(result['counts'])
    with st.expander('Cluster profiles'):st.dataframe(result['profiles'],hide_index=True,width='stretch')
    csv_download(result['clustered'],'CacaoY_clustered.csv','Download assigned clusters')
    if st.button('Run optional initialization-stability check'):
        with st.spinner('Comparing ten clustering seeds...'):
            st.session_state['cluster_stability']=seed_stability(dataset['id'],dataset['clean'],result['k'])
        st.session_state.pop('export_bytes',None)
    stability=st.session_state.get('cluster_stability')
    if stability is not None:
        st.metric('Mean seed agreement (adjusted Rand index)',f'{stability.adjusted_rand_index.mean():.3f}')
        st.dataframe(stability,hide_index=True)
    if result['silhouette']<.1:st.warning('The silhouette is close to zero: these groups overlap substantially. Treat the segmentation as exploratory.')
    st.caption('The cluster-assisted tree refits its own encoding, cluster selection and modes inside every training fold. It does not reuse whole-dataset clusters during validation.')
else:st.info('Clustering waits until you press Run. Other pages can be used without loading its results.')
