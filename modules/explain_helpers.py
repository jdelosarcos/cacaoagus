"""Per-record model scores and local sensitivity; no estimator is refitted."""
import json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from .config import FEATURES,YIELD_CLASSES
CLASS_INDEX={label:i for i,label in enumerate(YIELD_CLASSES)}
def class_scores(model,records):
    """Return scores in yield order; never manufacture SVM probabilities."""
    classifier=model.named_steps['classifier']
    if hasattr(model,'predict_proba'):
        scores=model.predict_proba(records)
        kind='uncalibrated_probability'
    else:
        scores=model.decision_function(records)
        kind='raw_decision_score'
    order=[list(classifier.classes_).index(label) for label in YIELD_CLASSES]
    return np.asarray(scores)[:,order],kind

def score_replacement_checks(model,records,predicted,base_scores,reference_train):
    """One-field sensitivity, using only training frequencies as references."""
    edits=[]; descriptions=[]
    records=records.reset_index(drop=True)
    for feature in FEATURES:
        alternatives=reference_train[feature].value_counts(normalize=True).head(3)
        for replacement,weight in alternatives.items():
            positions=np.flatnonzero(records[feature].ne(replacement).to_numpy())
            if not len(positions):continue
            changed=records.iloc[positions].copy()
            changed[feature]=replacement
            edits.append(changed)
            descriptions.extend((int(pos),feature,str(replacement),float(weight)) for pos in positions)
    if not edits:return pd.DataFrame()
    changed_records=pd.concat(edits,ignore_index=True)
    changed_scores,_=class_scores(model,changed_records)
    predicted_columns=np.array([CLASS_INDEX[predicted[item[0]]] for item in descriptions])
    assigned_scores=changed_scores[np.arange(len(changed_scores)),predicted_columns]
    temporary=pd.DataFrame(descriptions,columns=['record_position','feature','alternative_value','training_frequency'])
    temporary['replacement_score']=assigned_scores
    temporary['weighted_score']=temporary.replacement_score*temporary.training_frequency
    rows=[]
    for (position,feature),group in temporary.groupby(['record_position','feature'],sort=False):
        original_score=base_scores[position,CLASS_INDEX[predicted[position]]]
        replaced=float(group.weighted_score.sum()/group.training_frequency.sum())
        rows.append({'record_position':int(position),'feature':feature,
            'observed_value':str(records.iloc[position][feature]),
            'reference_alternatives':' | '.join(group.alternative_value),
            'assigned_class_score':float(original_score),'weighted_replacement_score':replaced,
            'score_difference':float(original_score-replaced)})
    return pd.DataFrame(rows)

def decision_paths_for_records(model,records):
    classifier=model.named_steps['classifier']
    if not isinstance(classifier,DecisionTreeClassifier):return ['']*len(records)
    design=model.named_steps['design']
    encoded=design.transform(records)
    names=design.get_feature_names_out()
    paths=classifier.decision_path(encoded)
    result=[]
    for i in range(len(records)):
        conditions=[]
        for node in paths.indices[paths.indptr[i]:paths.indptr[i+1]]:
            f=classifier.tree_.feature[node]
            if f<0:continue
            threshold=classifier.tree_.threshold[node]
            relation='<=' if encoded[i,f]<=threshold else '>'
            conditions.append(f'{names[f]} {relation} {threshold:.3f}')
        result.append(' AND '.join(conditions))
    return result
