"""Training is explicit, one model at a time; preprocessing remains inside CV."""
import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,classification_report,confusion_matrix
from .config import FEATURES,YIELD_CLASSES,SEED
from .categorical import CategoricalDesign

MODEL_NAMES=['Majority baseline','Decision tree','Cluster + decision tree',
    'Cluster-only decision tree','Logistic regression','Random forest','Support vector machine']

def create_split(clean):
    X=clean[FEATURES].copy();y=clean.yield_class.copy()
    if len(clean)<25 or y.nunique()!=3:
        raise ValueError('Training requires all three yield classes and at least 25 labeled records.')
    groups=pd.util.hash_pandas_object(X,index=False).astype(str)
    if y.value_counts().min()<5:raise ValueError('At least five labeled records per pooled yield category are required for the grouped holdout split.')
    train,test=next(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED).split(X,y,groups))
    if set(y.iloc[train])!=set(YIELD_CLASSES) or set(y.iloc[test])!=set(YIELD_CLASSES):
        raise ValueError('The grouped split does not represent every yield class in both subsets. Collect more independent labeled records.')
    assert set(groups.iloc[train]).isdisjoint(set(groups.iloc[test]))
    return {'train':train,'test':test,'X_train':X.iloc[train],'X_test':X.iloc[test],
        'y_train':y.iloc[train],'y_test':y.iloc[test],'groups_train':groups.iloc[train]}

def get_spec(name):
    tree_grid={'classifier__max_depth':[3,5,None],'classifier__min_samples_leaf':[5,10]}
    specs={
        'Majority baseline':(False,False,DummyClassifier(strategy='most_frequent'),{}),
        'Decision tree':(False,False,DecisionTreeClassifier(class_weight='balanced',random_state=SEED),tree_grid),
        'Cluster + decision tree':(True,False,DecisionTreeClassifier(class_weight='balanced',random_state=SEED),tree_grid),
        'Cluster-only decision tree':(True,True,DecisionTreeClassifier(class_weight='balanced',random_state=SEED),{'classifier__max_depth':[2,3],'classifier__min_samples_leaf':[5,10]}),
        'Logistic regression':(False,False,LogisticRegression(class_weight='balanced',max_iter=3000,random_state=SEED),{'classifier__C':[0.1,1,10]}),
        'Random forest':(False,False,RandomForestClassifier(n_estimators=250,class_weight='balanced',n_jobs=1,random_state=SEED),{'classifier__max_depth':[5,None],'classifier__min_samples_leaf':[2,5]}),
        'Support vector machine':(False,False,SVC(class_weight='balanced',kernel='rbf'),{'classifier__C':[0.1,1,10]})}
    return specs[name]

def train_one(clean,name,folds=5):
    started=time.perf_counter();split=create_split(clean)
    if folds not in [3,5]:raise ValueError('Choose 3 or 5 CV folds.')
    cv=list(StratifiedGroupKFold(n_splits=folds,shuffle=True,random_state=43).split(
        split['X_train'],split['y_train'],split['groups_train']))
    for tr,va in cv:
        if set(split['y_train'].iloc[tr])!=set(YIELD_CLASSES):raise ValueError('A training fold is missing a yield class; more independent records are needed.')
        assert set(split['groups_train'].iloc[tr]).isdisjoint(set(split['groups_train'].iloc[va]))
    cluster,only,classifier,grid=get_spec(name)
    model=Pipeline([('design',CategoricalDesign(cluster,only)),('classifier',classifier)])
    search=GridSearchCV(model,grid,scoring='f1_macro',cv=cv,n_jobs=1,refit=True,
                        return_train_score=True,error_score='raise')
    search.fit(split['X_train'],split['y_train'])
    prediction=search.predict(split['X_test'])
    metrics={'model':name,'cv_folds':folds,'cv_macro_f1':float(search.best_score_),
        'cv_sd':float(search.cv_results_['std_test_score'][search.best_index_]),
        'train_macro_f1':float(search.cv_results_['mean_train_score'][search.best_index_]),
        'holdout_accuracy':accuracy_score(split['y_test'],prediction),
        'holdout_macro_f1':f1_score(split['y_test'],prediction,average='macro',zero_division=0),
        'holdout_balanced_accuracy':balanced_accuracy_score(split['y_test'],prediction),
        'train_records':len(split['train']),'holdout_records':len(split['test']),
        'best_parameters':json.dumps(search.best_params_),'training_seconds':time.perf_counter()-started}
    rows=clean.iloc[split['test']][['source_csv_row','location','yield_original','yield_class']].copy()
    rows['predicted_yield_class']=prediction
    rows['correct']=rows.yield_class.eq(rows.predicted_yield_class)
    rows=rows.sort_values('source_csv_row')
    report=pd.DataFrame(classification_report(split['y_test'],prediction,labels=YIELD_CLASSES,
        output_dict=True,zero_division=0)).transpose().rename_axis('yield_class').reset_index()
    return {'name':name,'folds':folds,'estimator':search.best_estimator_,'metrics':metrics,
        'predictions':rows,'classification_report':report,
        'confusion_matrix':pd.DataFrame(confusion_matrix(split['y_test'],prediction,labels=YIELD_CLASSES),index=YIELD_CLASSES,columns=YIELD_CLASSES),
        'cv_results':pd.DataFrame(search.cv_results_).drop(columns=['params']),
        'dataset_split':split}

def fit_clusters(clean):
    split=create_split(clean)
    design=CategoricalDesign(with_cluster=True).fit(split['X_train'])
    labels=design.cluster_labels(clean[FEATURES])
    clustered=clean.copy();clustered['cluster']=labels
    counts=pd.crosstab(clustered.cluster,clustered.yield_class).reindex(columns=YIELD_CLASSES,fill_value=0)
    profiles=[]
    for k,part in clustered.groupby('cluster'):
        profiles.append({'cluster':int(k),'records':len(part),**{c+'_mode':part[c].mode().iloc[0] for c in FEATURES}})
    return {'design':design,'clustered':clustered,'selection':design.cluster_scores_,
        'counts':counts,'profiles':pd.DataFrame(profiles),'k':design.best_k_,
        'silhouette':float(design.silhouette_),'train_records':len(split['train'])}
