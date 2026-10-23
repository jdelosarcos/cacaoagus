"""Bounded caches; compute only when a selected page explicitly calls a task.

Cached model resources are treated as read-only after fit. Dataset bytes/hash,
model name and CV settings isolate results; no global cache clearing is needed.
"""
import streamlit as st

@st.cache_data(show_spinner=False,max_entries=4)
def preprocess(content):
    from .data import prepare_dataset
    return prepare_dataset(content)

@st.cache_data(show_spinner=False,max_entries=8)
def describe(clean):
    from .statistics import frequencies
    return frequencies(clean)

@st.cache_data(show_spinner=False,max_entries=4)
def association_analysis(clean,variables,permutations):
    from .statistics import associations
    return associations(clean,variables,permutations)

@st.cache_resource(show_spinner=False,max_entries=4)
def clustering(dataset_id,clean):
    from .modeling import fit_clusters
    return fit_clusters(clean)

@st.cache_resource(show_spinner=False,max_entries=16)
def train_model(dataset_id,clean,name,folds):
    from .modeling import train_one
    return train_one(clean,name,folds)

@st.cache_data(show_spinner=False,max_entries=4)
def forecasting(content,horizon,seasonal_period):
    from .forecast import forecast_bytes
    return forecast_bytes(content,horizon,seasonal_period)

@st.cache_data(show_spinner=False,max_entries=4)
def seed_stability(dataset_id,clean,k):
    from .modeling import create_split
    from .categorical import CategoricalDesign,KModesSimple
    from sklearn.metrics import adjusted_rand_score
    import pandas as pd
    split=create_split(clean)
    design=CategoricalDesign(with_cluster=True).fit(split['X_train'])
    encoded=design.ordinal_.transform(split['X_train']).astype(int)
    rows=[]
    for seed in range(10):
        alternate=KModesSimple(k,random_state=seed).fit(encoded)
        rows.append({'seed':seed,'adjusted_rand_index':adjusted_rand_score(design.km_.labels_,alternate.labels_)})
    return pd.DataFrame(rows)
