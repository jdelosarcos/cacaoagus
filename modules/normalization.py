"""Characteristic-based survey normalization; no learned transformations."""
import re
from .config import MULTI

def basic_text(value):
    s = str(value).strip().lower().replace('\ufffd', '-')
    s = s.replace('\u2013','-').replace('\u2014','-')
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'\s*-\s*', '-', s)
    s = re.sub(r'\s*/\s*', ' / ', s)
    return s if s else 'unknown'

def normalize(value, col):
    s = basic_text(value)
    fixes = {'pesticdes':'pesticides', 'livestack':'livestock',
             'coorperatives':'cooperatives', 'coorperative':'cooperative',
             'knowlwdge':'knowledge', 'drougth':'drought',
             'membes':'members', 'tacechniques':'techniques'}
    for a,b in fixes.items():
        s = s.replace(a,b)
    s = re.sub(r'^y es', 'yes', s)
    s = s.replace('!0 times','10 times')
    if col == 'age':
        if s.startswith('30 years'): s = '30 years old and below'
        if s.startswith('61 years'): s = '61 years old and above'
    if col == 'variety':
        s = re.sub(r'uf\s*-?\s*18', 'uf-18', s)
        s = re.sub(r'br\s*-?\s*25', 'br-25', s)
    if col == 'spacing': s = s.replace('2.6-3-5','2.6-3.5')
    if col == 'fertilizer' and s == 'both chemical and organic':
        s = 'both organic and chemical'
    if col == 'harvests_per_year' and s == 'more than 3': s = 'more than 3 times'
    if col == 'intercrop' and s == 'rubber': s = 'yes-rubber'
    if col == 'main_challenge':
        s = s.replace('pes and disease','pest and disease')
        s = s.replace('lak of knowledge','lack of knowledge')
        s = s.replace('adverse weather and climate','adverse weather/climate')
        s = re.sub(r'adverse weather\s*/\s*climate','adverse weather/climate',s)
        s = re.sub(r'lack of knowledge(?:\s*/\s*training)?', 'lack of knowledge/training',s)
    if col == 'products_sold':
        s = s.replace('dried caca beans','dried cacao beans')
        s = re.sub(r'dried cacao bean(?=\s*(?:/|$))','dried cacao beans',s)
    if col == 'buyer':
        if 'cacao' in s: return 'unknown_invalid_product_response'
        s = re.sub(r'direct consumers?', 'direct consumers', s)
    if col == 'agency_influence':
        if 'knowledge' in s: s = 'provides technical knowledge'
        elif 'productivity' in s: s = 'increases productivity'
        elif 'farming' in s: s = 'improves farming techniques'
    if col in MULTI and ' / ' in s:
        prefix = 'yes-' if s.startswith('yes-') else ''
        if prefix: s = s[4:]
        s = prefix + ' / '.join(sorted(set(s.split(' / '))))
    return s
