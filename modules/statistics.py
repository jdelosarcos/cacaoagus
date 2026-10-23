"""Survey statistics are independent of model training."""
import numpy as np
import pandas as pd
from .config import COLUMNS,YIELD_CLASSES,SEED

def frequencies(clean):
    return pd.DataFrame([{'variable':col,'category':value,'n':int(n),'percent':100*n/len(clean)}
        for col in COLUMNS+['yield_class'] for value,n in clean[col].value_counts().items()])

def associations(clean,variables,permutations=1999):
    from scipy.stats import chi2_contingency
    if not 99<=permutations<=9999:raise ValueError('Use 99–9,999 permutations.')
    rng=np.random.default_rng(SEED);rows=[]
    target=pd.Categorical(clean.yield_class,categories=YIELD_CLASSES).codes
    for col in variables:
        valid=~clean[col].str.startswith('unknown')
        x=pd.Categorical(clean.loc[valid,col]).codes;y=target[valid.to_numpy()]
        r,c=int(x.max()+1),3
        if r<2:continue
        table=np.bincount(x*c+y,minlength=r*c).reshape(r,c)
        chi,p,df,expected=chi2_contingency(table,correction=False)
        greater=0
        for iteration in range(permutations):
            perm=np.bincount(x*c+rng.permutation(y),minlength=r*c).reshape(r,c)
            greater+=((perm-expected)**2/expected).sum()>=chi-1e-12
        n=table.sum();phi=max(0,chi/n-(c-1)*(r-1)/(n-1))
        denom=min(r-(r-1)**2/(n-1)-1,c-(c-1)**2/(n-1)-1)
        rows.append({'variable':col,'effective_n':int(valid.sum()),'chi_square':chi,'df':df,
            'p_permutation':(greater+1)/(permutations+1),'p_asymptotic':p,
            'cramers_v_bias_corrected':float(np.sqrt(phi/denom)) if denom>0 else 0.,
            'expected_cells_under_5_percent':100*np.mean(expected<5)})
    result=pd.DataFrame(rows)
    if not len(result):raise ValueError('No selected variable has at least two observed categories.')
    p=result.p_permutation.to_numpy();order=np.argsort(p)
    adjusted=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    q=np.empty_like(adjusted);q[order]=np.minimum(adjusted,1)
    result['q_bh']=q
    return result.sort_values(['q_bh','cramers_v_bias_corrected'],ascending=[True,False])
