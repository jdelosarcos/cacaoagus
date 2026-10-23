"""Explain requested records only; do not compute all models at page startup."""
import json
from collections import Counter
import numpy as np
import pandas as pd
from .config import FEATURES,YIELD_CLASSES,CORE_MODELS
from .explain_helpers import class_scores,score_replacement_checks,decision_paths_for_records

def explain(bundle,reference_train,records,source_rows,actual=None,comparisons=None,sensitivity=True):
    records=records[FEATURES].reset_index(drop=True).copy()
    if len(records)>100:raise ValueError('Explain at most 100 records per batch to keep the interface responsive.')
    actuals=[None]*len(records) if actual is None else list(actual)
    ids=list(source_rows)
    if len(ids)!=len(records) or len(actuals)!=len(records):raise ValueError('Identifiers/outcomes must match the input row count.')
    model=bundle['estimator'];name=bundle['name'];predicted=model.predict(records)
    scores,kind=class_scores(model,records)
    local=score_replacement_checks(model,records,predicted,scores,reference_train) if sensitivity else pd.DataFrame()
    if len(local):local['source_csv_row']=[int(ids[i]) for i in local.record_position]
    paths=decision_paths_for_records(model,records)
    comparison_models={n:b for n,b in (comparisons or {}).items() if n in CORE_MODELS}
    votes={n:b['estimator'].predict(records) for n,b in comparison_models.items()}
    ordinal={label:i for i,label in enumerate(YIELD_CLASSES)};rows=[]
    for i,profile in records.iterrows():
        label=str(predicted[i]);truth=actuals[i]
        correct=None if truth is None else bool(label==truth)
        distance=None if truth is None else ordinal[label]-ordinal[truth]
        result='Outcome not supplied' if truth is None else ('Correct category' if correct else
            ('Predicted above reported category' if distance>0 else 'Predicted below reported category'))
        column=ordinal[label];alternative=max((j for j in range(3) if j!=column),key=lambda j:scores[i,j])
        vote=Counter(prediction[i] for prediction in votes.values())
        available=len(votes);agreement=int(vote[label]);maximum=max(vote.values()) if vote else 0
        top=[]
        if len(local):
            part=local.loc[local.record_position.eq(i)].copy();part['abs']=part.score_difference.abs()
            for entry in part.sort_values('abs',ascending=False).head(3).to_dict('records'):
                if entry['abs']>1e-10:top.append({k:entry[k] for k in ['feature','observed_value','reference_alternatives','score_difference']})
        outcome=('Actual yield was not supplied; correctness is unknown.' if truth is None else
            ('This matches the reported category.' if correct else f'The reported category is {truth}; the prediction is {abs(distance)} category band(s) '+('higher.' if distance>0 else 'lower.')))
        score=(f'Assigned-class probability {scores[i,column]:.1%}, uncalibrated and not a chance of correctness.' if kind=='uncalibrated_probability'
               else f'Raw SVM decision score {scores[i,column]:.3f}; not a probability.')
        agreement_text=(f'{agreement} of {available} trained comparison algorithms predict this category.' if available else 'No primary comparison algorithms have been trained for this CV setting.')
        sensitivity_text=('Largest model sensitivities: '+', '.join(f"{e['feature']} ({e['score_difference']:+.3f})" for e in top)+'.' if top else
            ('Tested replacements did not change the assigned-class score.' if sensitivity else 'Local replacement sensitivity was not requested.'))
        actions=[]
        if truth==YIELD_CLASSES[0]:actions.append('Confirm the reported low yield, hectare denominator and season definition.')
        if truth is None and label==YIELD_CLASSES[0]:actions.append('Measure actual yield before treating the predicted low-yield category as an observed need.')
        if correct is False:actions.append('Review this error against the source response and retain it in validation reporting.')
        if available>1 and maximum<available:actions.append('Inspect disagreement among the trained comparison algorithms.')
        if profile.training_attendance=='no':actions.append('Ask about reported training access and needs.')
        if profile.farm_support=='no':actions.append('Clarify the reported lack of support-service access and stated needs.')
        if not actions:actions.append('Verify the source record and retain it for routine model monitoring.')
        rows.append({'source_csv_row':int(ids[i]),'municipality':profile.location,'model':name,
            'cv_folds':bundle['folds'],'reported_yield_class':truth,'predicted_yield_class':label,
            'correct':correct,'error_type':result,'signed_error_category_bands':distance,
            'score_kind':kind,'assigned_class_score':float(scores[i,column]),
            'alternative_class':YIELD_CLASSES[alternative],'alternative_score':float(scores[i,alternative]),
            'score_margin':float(scores[i,column]-scores[i,alternative]),
            'agreement_count':agreement,'comparison_models_available':available,
            'comparison_models_expected':4,'scores_by_class':json.dumps(dict(zip(YIELD_CLASSES,map(float,scores[i])))),
            'local_sensitivities':json.dumps(top),'decision_tree_path':paths[i],
            'interpretation':f'{name} predicts {label}. {outcome} {score} {agreement_text} {sensitivity_text} Replacement checks describe model behavior, not causal effects on yield.',
            'suggested_record_review':' '.join(actions)})
    return pd.DataFrame(rows),local

def importance(bundle,repeats=10):
    from sklearn.inspection import permutation_importance
    split=bundle['dataset_split']
    result=permutation_importance(bundle['estimator'],split['X_test'],split['y_test'],
                                  scoring='f1_macro',n_repeats=repeats,random_state=42,n_jobs=1)
    return pd.DataFrame({'feature':FEATURES,'macro_f1_decrease_mean':result.importances_mean,
        'macro_f1_decrease_sd':result.importances_std}).sort_values('macro_f1_decrease_mean',ascending=False)
