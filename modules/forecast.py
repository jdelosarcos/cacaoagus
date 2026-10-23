"""Pure forecasting functions. Dated numeric yield is required."""
import io
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
def ses_fit_forecast(values,horizon):
    values=np.asarray(values,dtype=float)
    def loss(alpha):
        level=values[0]; errors=[]
        for v in values[1:]:
            errors.append(v-level)
            level=alpha*v+(1-alpha)*level
        return float(np.mean(np.square(errors)))
    alpha=minimize_scalar(loss,bounds=(0.001,0.999),method='bounded').x
    level=values[0]
    for v in values[1:]: level=alpha*v+(1-alpha)*level
    return np.full(horizon,level)

def forecast_dated_yield(csv_path,horizon=3,seasonal_period=None,output_dir=None):
    if not isinstance(horizon,int) or horizon<1: raise ValueError('horizon must be a positive integer.')
    series=pd.read_csv(csv_path)
    required={'date','yield_kg_per_hectare'}
    if not required.issubset(series.columns): raise ValueError('Required columns: date, yield_kg_per_hectare.')
    series=series[['date','yield_kg_per_hectare']].copy()
    series['date']=pd.to_datetime(series.date,errors='raise')
    series['yield_kg_per_hectare']=pd.to_numeric(series.yield_kg_per_hectare,errors='raise')
    series=series.sort_values('date')
    if series.isna().any().any() or not np.isfinite(series.yield_kg_per_hectare).all():
        raise ValueError('Remove or resolve missing/nonfinite observations before forecasting.')
    if series.date.duplicated().any(): raise ValueError('Use one consistently defined value per date.')
    if (series.yield_kg_per_hectare<0).any(): raise ValueError('Yield cannot be negative.')
    if len(series)<max(15,2*horizon+6): raise ValueError('Provide at least 15 regular dated periods and enough history for the horizon.')
    freq=pd.infer_freq(pd.DatetimeIndex(series.date))
    if freq is None: raise ValueError('Dates must be regularly spaced; resolve missing periods explicitly.')
    values=series.yield_kg_per_hectare.to_numpy(dtype=float)
    methods={
        'Naive':lambda a,h:np.full(h,a[-1]),
        'Drift':lambda a,h:a[-1]+np.arange(1,h+1)*(a[-1]-a[0])/(len(a)-1),
        'Simple exponential smoothing':ses_fit_forecast}
    if seasonal_period is not None:
        if not isinstance(seasonal_period,int) or seasonal_period<2:
            raise ValueError('seasonal_period must be an integer of at least 2.')
        if len(values)<3*seasonal_period: raise ValueError('Seasonal modeling requires at least three complete cycles.')
        methods['Seasonal naive']=lambda a,h:np.resize(a[-seasonal_period:],h)
    minimum_train=max(8,2*seasonal_period if seasonal_period else 8)
    origins=range(max(minimum_train,len(values)-2*horizon),len(values)-horizon+1)
    if len(origins)<2: raise ValueError('More history is needed for at least two rolling-origin backtests.')
    records=[]
    for name,fun in methods.items():
        errors=[]
        for t in origins:
            pred=np.maximum(0,fun(values[:t],horizon))
            errors.extend(values[t:t+horizon]-pred)
        records.append({'model':name,'validation_mae':float(np.mean(np.abs(errors))),
                        'validation_rmse':float(np.sqrt(np.mean(np.square(errors)))),
                        'rolling_origins':len(origins),'horizon':horizon})
    performance=pd.DataFrame(records).sort_values('validation_mae')
    winner=performance.iloc[0]['model']
    future_dates=pd.date_range(start=series.date.iloc[-1],periods=horizon+1,freq=freq)[1:]
    prediction=pd.DataFrame({'date':future_dates,'forecast_yield_kg_per_hectare':
        np.maximum(0,methods[winner](values,horizon)),'method':winner})
    return performance,prediction

def forecast_bytes(content, horizon=3, seasonal_period=None):
    return forecast_dated_yield(io.BytesIO(content),horizon,seasonal_period)
