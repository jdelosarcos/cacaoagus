"""Read the supplied two-header survey or its canonical preprocessed CSV."""
import hashlib
import io
import re
import pandas as pd
from .config import COLUMNS,YIELD_ORDER_ORIGINAL,YIELD_CLASSES
from .normalization import normalize

def prepare_dataset(content):
    if not content:raise ValueError('The uploaded file is empty.')
    first=pd.read_csv(io.BytesIO(content),dtype=str,keep_default_na=False,
                      encoding='utf-8',encoding_errors='replace')
    canonical=set(COLUMNS).issubset(first.columns)
    if canonical:
        original=first[COLUMNS].copy();header_index=0
    else:
        header_index=1
        original=pd.read_csv(io.BytesIO(content),header=header_index,dtype=str,
                            keep_default_na=False,encoding='utf-8',encoding_errors='replace')
        empty_trailing=[c for c in original if str(c).startswith('Unnamed:') and original[c].str.strip().eq('').all()]
        original=original.drop(columns=empty_trailing)
        if original.shape[1]!=36 or str(original.columns[0]).strip().lower()!='location':
            raise ValueError('Expected the original CacaoY.csv with its two header rows, or the canonical preprocessed CSV. The survey needs 36 fields beginning with Location.')
        original.columns=COLUMNS
    parsed_rows=len(original)
    keep=original.apply(lambda s:s.str.strip().ne('')).any(axis=1)
    raw=original.loc[keep].copy()
    if not len(raw):raise ValueError('No survey records were found after empty rows were removed.')
    if len(raw)>10000:raise ValueError('This survey app supports up to 10,000 records. Use a separate batch workflow for larger files.')
    if canonical and 'source_csv_row' in first:
        ids=pd.to_numeric(first.loc[keep,'source_csv_row'],errors='raise').astype(int)
        if ids.duplicated().any():raise ValueError('source_csv_row must identify each retained row uniquely.')
    else:ids=raw.index+header_index+2
    raw.insert(0,'source_csv_row',ids)
    raw.reset_index(drop=True,inplace=True)
    clean=raw.copy();changes=[]
    for col in COLUMNS:
        clean[col]=raw[col].map(lambda value:normalize(value,col))
        for index in clean.index[clean[col].ne(raw[col])]:
            changes.append({'source_csv_row':int(clean.at[index,'source_csv_row']),
                'column':col,'original':raw.at[index,col],'normalized':clean.at[index,col]})
    unexpected=sorted(set(clean.yield_original)-set(YIELD_ORDER_ORIGINAL))
    if unexpected:raise ValueError('Unrecognized or missing yield categories: '+', '.join(unexpected)+'. Supply the labeled survey for training; new unlabeled farmers belong in Module 6.')
    clean['yield_class']=clean.yield_original.map({
        'less than 100 kg':YIELD_CLASSES[0],'100-250 kg':YIELD_CLASSES[1],
        '251-500 kg':YIELD_CLASSES[2],'more than 750 kg':YIELD_CLASSES[2]})
    def attendance(value):
        if value=='no':return 0
        match=re.search(r'\d+',value)
        return int(match.group()) if match else pd.NA
    clean['training_count']=clean.training_attendance.map(attendance).astype('Int64')
    clean['identical_response_flag']=clean.duplicated(COLUMNS,keep=False)
    audit={'csv_rows_after_headers':parsed_rows,'empty_rows_removed':int((~keep).sum()),
        'retained_survey_records':len(clean),'survey_fields':36,
        'true_empty_cells_before_normalization':int(raw[COLUMNS].eq('').sum().sum()),
        'raw_exact_duplicate_excess':int(raw.duplicated(COLUMNS).sum()),
        'normalized_duplicate_excess':int(clean.duplicated(COLUMNS).sum()),
        'invalid_buyer_responses':int(clean.buyer.str.startswith('unknown').sum()),
        'literal_none_responses_preserved':int(raw[COLUMNS].apply(lambda s:s.str.strip().str.lower().eq('none')).sum().sum()),
        'changed_cells':len(changes)}
    return {'id':hashlib.sha256(content).hexdigest(),'raw':raw,'clean':clean,
        'audit':audit,'changes':pd.DataFrame(changes,columns=['source_csv_row','column','original','normalized'])}

def prepare_new_farmers(content,features):
    frame=pd.read_csv(io.BytesIO(content),dtype=str,keep_default_na=False,
                      encoding='utf-8',encoding_errors='replace')
    missing=set(features)-set(frame.columns)
    if missing:raise ValueError('Missing predictor columns: '+', '.join(sorted(missing)))
    if not len(frame):raise ValueError('The new-farmer CSV has no rows.')
    if len(frame)>1000:raise ValueError('Predict at most 1,000 new farmers at a time in this interactive app.')
    return pd.DataFrame({col:frame[col].map(lambda value:normalize(value,col)) for col in features})
