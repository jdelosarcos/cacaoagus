# Cacao Streamlit app — modular edition

This is one Streamlit application with nine independently loaded modules.
It converts the Cacao Colab workflow into reusable functions and separate pages.
Keep the entire folder together. Do not paste the whole notebook into Streamlit
or upload only app.py without its modules.

## Run on your computer

Use Python 3.12, or another supported Python version of at least 3.10.

1. Extract Cacao_Streamlit_Modular.zip.
2. Open a terminal inside the extracted cacao_streamlit folder.
3. Create and activate a virtual environment (recommended).

Windows:

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local address printed by Streamlit. In Module 1, upload CacaoY.csv
or click **Load included CacaoY sample**. The provided sample is the original
uploaded survey. Preprocessing is performed only when requested.

If your terminal does not recognize `streamlit`, use the full command
`python -m streamlit run app.py` with the virtual environment activated.

## Modules and recommended order

| Module | What it does | When computation starts |
|---|---|---|
| 1. Upload and preprocess | Cleans headers, blanks, spelling, categories and flags repeated responses | Preprocess or Load sample |
| 2. Statistical analysis | Frequency tables, exact training-count summaries and permutation association tests | Descriptions are small/cached; association tests require Run |
| 3. Exploratory analysis | One selected distribution or cross-tab chart at a time | Selected chart only |
| 4. Clustering | Training-only k-modes selection and assignment; optional seed stability | Run clustering; stability has a separate button |
| 5. Prediction models | Baseline, two trees, cluster-only tree, logistic regression, random forest and SVM | Train one selected model |
| 6. Prediction analysis | Three default holdout explanations, selected batches, new-farmer prediction, local sensitivity and optional importance | Analyze/Predict; importance has a separate button |
| 7. Decision dashboard | Reads completed model results, filters records, interprets errors and lists review questions | No training occurs here |
| 8. Forecasting | Compares naive, drift, exponential smoothing and optional seasonal naive | Run after uploading a separate dated series |
| 9. Export results | Packages only completed tasks into an in-memory ZIP | Prepare results ZIP |

For the complete original comparison, train each model in Module 5 using
**5 folds — notebook comparison**. Start with the majority baseline, then the
plain decision tree and cluster-assisted tree, followed by logistic regression,
random forest and SVM. The cluster-only tree is a supplementary comparison.

Use **3 folds — faster preview** when you want a quicker exploratory run.
Three-fold selection can choose different settings and produce different
scores. Both configurations remain separately labeled; compare like fold counts.

## Why loading is lighter

- `app.py` registers pages without importing the Colab script or running its code.
- Only the selected page executes. The app does not place every heavy task in
  tabs or expanders, whose contents would otherwise run together.
- Expensive operations require a button. Forms batch training settings into
  one request instead of retraining after every control change.
- `st.cache_data` caches preprocessing, frequency tables, association tests and
  forecast results with bounded entries.
- `st.cache_resource` caches fitted models and clustering results. Fitted
  resources are read-only during prediction and interpretation.
- Dataset identity, actual input content and model settings are part of cache
  keys. Switching datasets clears old model/explanation session references.
- One model runs at a time, with one training worker. There is no automatic
  seven-model training job on startup or dashboard refresh.
- The silhouette distance matrix uses all training records up to 1,000 rows;
  larger training sets use a fixed sample of 1,000 for silhouette scoring.
  Cluster modes still fit all training records. The 302-record survey reproduces
  the original notebook calculation without sampling.
- Record explanations are limited to requested batches of at most 100. They
  do not automatically generate every model’s explanation for every farmer.
- ZIP creation and feature importance are optional tasks.
- No external dashboard framework, public tunnel, GPU, pickle upload or
  heavy optional ML library is required.

Switching a page still causes Streamlit to rerun its light entrypoint and the
selected page. Cached operations and session results prevent model refitting.
A server restart or browser-session reset may require loading data and rebuilding
session results. File/module splitting reduces unnecessary work; it cannot remove
the actual cost of cross-validation or large datasets.

## Data interpretation

The original survey contains 302 respondent records and 36 fields. Empty rows
are removed; valid literal “None” responses are preserved. Three buyer responses
containing product descriptions are flagged as unknown. Identical profiles are
retained because no unique farmer identifier is available and are grouped across
train/validation boundaries.

Yield is recorded as ranges, not measured exact kilograms. Modeling uses:

- Less than 100 kg per hectare per season
- 100–250 kg per hectare per season
- 251 kg and above per hectare per season

The 251–500 kg group and two records above 750 kg are pooled in the final class.
Original categories remain in the processed data. No exact yield is invented
from category midpoints. Preprocessing, cluster selection and cluster modes
stay inside training folds during predictive evaluation.

The original five-fold comparison is reproduced by this app:

| Model | Holdout accuracy | Holdout macro F1 |
|---|---:|---:|
| Majority baseline | 60.7% | 0.252 |
| Decision tree | 54.1% | 0.409 |
| Cluster-assisted decision tree | 54.1% | 0.409 |
| Cluster-only decision tree | 32.8% | 0.228 |
| Logistic regression | 47.5% | 0.381 |
| Random forest | 59.0% | 0.502 |
| SVM | 60.7% | 0.367 |

Random forest has the best original training CV macro F1, but its accuracy is
below the majority baseline. The two k-modes groups are weakly separated and
did not improve the decision tree. These limits remain visible in the app.

Class probabilities are uncalibrated outputs, not chances that predictions
are correct. SVM scores remain raw decision scores. Local replacement
sensitivity describes the fitted model, not a causal farming intervention.
Dashboard review questions are not funding rules or invented urgency scores.

## New farmer predictions

1. Train a model in Module 5.
2. Open Module 6 and select **New farmers CSV**.
3. Download the input example to see the 28 required predictor column names.
   Its three rows are real held-out examples, not newly collected farmers.
4. Replace the rows with genuine new farmer responses and upload the CSV.
5. Click **Predict and interpret new farmers**.

Actual yield and correctness stay unknown for these new farmers. Raw uploaded
responses are normalized with the same rules before using the training-fitted
encoder. Unknown categories are handled without retraining the estimator.
Analysis batches are limited to 100 rows to keep the interface responsive.

## Forecasting

CacaoY.csv has no dated numeric yield series. Module 8 therefore requires a
separate CSV with `date` and `yield_kg_per_hectare`, one consistent observation
per regular period. Download the blank template in that module. Do not turn
respondent order, tree age or farmer age into dates.

At least 15 regular periods and enough observations for the requested horizon
are required. Several years of actual dated yields are preferable. Seasonal
naive needs an explicit season length and at least three cycles. Validation
chooses a method using rolling-origin MAE/RMSE and produces three future values
by default. Missing/irregular dates, duplicate periods and negative yields must
be resolved before forecasting. The sample survey alone creates no forecast.

## Hosting the app

For Streamlit hosting, keep this structure in your project/repository and select
**app.py** as the main file. Include **requirements.txt**, **modules/**,
**ui/**, **sample_data/** and **.streamlit/config.toml**. A `.ipynb` file is not
the Streamlit entrypoint. The source package is intended for deployment; this
delivery does not publish or configure an external hosting account.

```text
cacao_streamlit/
  app.py
  requirements.txt
  README.md
  VERIFICATION.json
  .streamlit/config.toml
  sample_data/CacaoY.csv
  modules/
    config.py
    normalization.py
    data.py
    categorical.py
    modeling.py
    statistics.py
    explain_helpers.py
    explanations.py
    forecast.py
    services.py
    exporting.py
  ui/
    common.py
    01_preprocessing.py
    02_statistics.py
    03_eda.py
    04_clustering.py
    05_models.py
    06_prediction_analysis.py
    07_dashboard.py
    08_forecasting.py
    09_exports.py
```

## Verification and implementation references

The application was exercised through real Streamlit AppTest on the supplied
dataset. Checks cover all nine pages, data preparation, on-demand execution,
seven model results against the notebook, cache reuse without refitting,
three prediction explanations, dashboard errors, exports, dataset replacement,
and new-record/forecast validation. Details and measured timings are in
VERIFICATION.json. These timings are environment-specific, not load-time guarantees.

Streamlit supports Python 3.10 or later for this source package; the verified
runtime used Python 3.12 and Streamlit 1.66.1. Full browser visual layout was not
verified by AppTest.

Primary documentation:

- https://docs.streamlit.io/develop/concepts/multipage-apps/page-and-navigation
- https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_data
- https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource
- https://docs.streamlit.io/develop/api-reference/execution-flow/st.form
- https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest
