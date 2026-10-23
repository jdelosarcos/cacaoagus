import hashlib
import io
import pandas as pd
import streamlit as st
from modules.services import forecasting
from ui.common import csv_download

st.title('8. Forecasting')
st.info('The farmer survey has no dated numeric yield series. Upload a separate time-series file; respondent order, farmer age and tree age cannot replace dates.')
st.download_button('Download blank time-series template',b'date,yield_kg_per_hectare\n','forecast_data_template.csv','text/csv')
with st.form('forecast_form'):
    upload=st.file_uploader('Upload regular dated yield observations',type=['csv'],key='forecast_upload')
    horizon=st.number_input('Future periods',min_value=1,max_value=12,value=3,step=1)
    season=st.number_input('Season length (0 = no seasonal-naive candidate)',min_value=0,max_value=52,value=0,step=1)
    run=st.form_submit_button('Run forecasting',type='primary')
if run:
    if upload is None:st.warning('Upload a dated numeric series first.')
    else:
        try:
            content=upload.getvalue()
            with st.spinner('Running rolling-origin forecast comparisons...'):
                metrics,prediction=forecasting(content,int(horizon),int(season) if season else None)
            history=pd.read_csv(io.BytesIO(content));history['date']=pd.to_datetime(history.date)
            st.session_state['forecast_result']={'name':upload.name,'digest':hashlib.sha256(content).hexdigest(),
                'history':history,'metrics':metrics,'predictions':prediction,'horizon':int(horizon),'season':int(season)}
            st.session_state.pop('export_bytes',None)
        except (ValueError,KeyError,pd.errors.ParserError) as error:st.error(str(error))
result=st.session_state.get('forecast_result')
if result:
    st.caption(f"Last completed forecast: {result['name']}, horizon {result['horizon']}, season setting {result['season']}. Changing form inputs does not replace it until Run is pressed.")
    st.dataframe(result['metrics'],hide_index=True)
    st.dataframe(result['predictions'],hide_index=True)
    observed=result['history'].set_index('date')[['yield_kg_per_hectare']].rename(columns={'yield_kg_per_hectare':'Observed'})
    forecast=result['predictions'].set_index('date')[['forecast_yield_kg_per_hectare']].rename(columns={'forecast_yield_kg_per_hectare':'Forecast'})
    st.line_chart(pd.concat([observed,forecast],axis=1))
    csv_download(result['predictions'],'future_yield_forecast.csv','Download forecast values')
    csv_download(result['metrics'],'forecast_backtest_metrics.csv','Download forecast validation metrics')
    st.caption('MAE/RMSE come from expanding-window model selection, not an independent final accuracy test. No prediction interval or measured future outcome is fabricated.')
st.caption('Provide one consistent numeric yield measure per regular period, at least 15 periods and sufficient history for the chosen horizon. Seasonal-naive modeling requires an explicit cycle and at least three cycles. More history is preferable.')
