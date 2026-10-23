"""Build requested exports in memory only, with no model serialization."""
import io
import re
import json
import zipfile
import pandas as pd

def build_results_zip(state):
    stream=io.BytesIO()
    def csv(archive,name,frame):archive.writestr(name,frame.to_csv(index=False).encode('utf-8-sig'))
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as archive:
        dataset=state.get('dataset')
        if dataset:
            csv(archive,'CacaoY_preprocessed.csv',dataset['clean'])
            csv(archive,'cleaning_log.csv',dataset['changes'])
            archive.writestr('data_quality_audit.json',json.dumps(dataset['audit'],indent=2))
        if state.get('association_result'):csv(archive,'yield_associations.csv',state['association_result']['table'])
        if state.get('clusters'):
            cluster=state['clusters'];csv(archive,'CacaoY_clustered.csv',cluster['clustered'])
            csv(archive,'cluster_selection.csv',cluster['selection']);csv(archive,'cluster_profiles.csv',cluster['profiles'])
        if state.get('cluster_stability') is not None:csv(archive,'cluster_seed_stability.csv',state['cluster_stability'])
        trained=state.get('trained_models',{})
        if trained:
            csv(archive,'model_comparison.csv',pd.DataFrame([b['metrics'] for b in trained.values()]))
            for bundle in trained.values():
                prefix=re.sub(r'[^a-z0-9]+','_',bundle['name'].lower()).strip('_')+f"_{bundle['folds']}fold"
                csv(archive,prefix+'/holdout_predictions.csv',bundle['predictions'])
                csv(archive,prefix+'/classification_report.csv',bundle['classification_report'])
                csv(archive,prefix+'/cv_grid_results.csv',bundle['cv_results'])
        for key in ['prediction_explanations','new_predictions']:
            if state.get(key):
                csv(archive,key+'.csv',state[key]['details'])
                if len(state[key]['checks']):csv(archive,key+'_sensitivity.csv',state[key]['checks'])
        if state.get('feature_importance'):csv(archive,'feature_importance.csv',state['feature_importance']['table'])
        if state.get('forecast_result'):
            result=state['forecast_result'];csv(archive,'forecast_values.csv',result['predictions'])
            csv(archive,'forecast_validation.csv',result['metrics'])
            archive.writestr('forecast_source.json',json.dumps({'file':result['name'],'sha256':result['digest'],'horizon':result['horizon'],'seasonal_period':result['season']},indent=2))
        archive.writestr('RESULTS_README.txt','This export contains only completed tasks from the current session.\nModel comparisons retain CV-fold settings. Predictions are yield categories, not exact kg.\nProbabilities are uncalibrated outputs; local sensitivity is not a causal effect.\nForecasts, if present, use a separate dated numeric series named in forecast_source.json.\nNo trained pickle/joblib file is exported or accepted.\n')
    return stream.getvalue()
