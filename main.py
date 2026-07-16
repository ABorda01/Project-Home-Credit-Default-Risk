"""
------------------------------------------------------------------------------
Project I: Home Credt Default Risk.
------------------------------------------------------------------------------

This script takes best models and strategies from EDA.ipynb and (mainly) model_selection_improv.ipynb
and condenses it into a cleaner .py script which produces the most relevant plots.

The justifications for the exploratory data analysis can be found in EDA.ipynb and the ones for the 
feature-engineering/model election related decisions can be found in model_selection_improv.ipynb.
"""

# Imports: (save sklearn methods)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import optbinning
import time
import threading
from contextlib import contextmanager

#--------------------------------------------------------------------------------------------
# Progress logging config
#--------------------------------------------------------------------------------------------
PROGRESS_LOG = True          # set False to silence all progress output
HEARTBEAT_INTERVAL_SEC = 60  # how often the heartbeat pings during long silent calls

_SCRIPT_START = time.time()

def _log(msg):
    """Timestamped checkpoint print, called after each major stage below."""
    if PROGRESS_LOG:
        elapsed = time.time() - _SCRIPT_START
        print(f'[{elapsed/60:6.1f} min] {msg}', flush=True)

@contextmanager
def _heartbeat(label, interval=HEARTBEAT_INTERVAL_SEC):
    """Background ticker for the few steps that run silently for a long time
    (e.g. grid_search.fit) so the process doesn't look hung."""
    if not PROGRESS_LOG:
        yield
        return
    stop_event = threading.Event()
    def _tick():
        while not stop_event.wait(interval):
            elapsed = time.time() - _SCRIPT_START
            print(f'[{elapsed/60:6.1f} min] ...still running: {label}', flush=True)
    t = threading.Thread(target=_tick, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop_event.set()

#--------------------------------------------------------------------------------------------
# Data reading of main and all aditional csv's. Can be found in: 
# https://www.kaggle.com/competitions/home-credit-default-risk/data
#--------------------------------------------------------------------------------------------

X_base = pd.read_csv('application_train.csv')
X_base = X_base.set_index('SK_ID_CURR')
y = X_base['TARGET']
X_base = X_base.drop(columns=['TARGET']) 

#Replace anomalous DAYS_EMPLOYED value
X_base['DAYS_EMPLOYED_ANOM'] = (X_base['DAYS_EMPLOYED'] == 365243)
X_base['DAYS_EMPLOYED'] = X_base['DAYS_EMPLOYED'].replace(365243, np.nan)

_log('Base application data loaded and cleaned')

""" Auxiliary csv's: 
NOTE: The index 'SK_ID_CURR' is not unique in the suplementary CSV's
so we have to groupby after taking only the relevant indexes from the clients in "X".
"""

bureau = pd.read_csv('bureau.csv')
bureau = bureau[bureau['SK_ID_CURR'].isin(X_base.index)].set_index('SK_ID_CURR')

ccb = pd.read_csv('credit_card_balance.csv')
ccb = ccb[ccb['SK_ID_CURR'].isin(X_base.index)].set_index('SK_ID_CURR')

install_pay = pd.read_csv('installments_payments.csv')
install_pay = install_pay[install_pay['SK_ID_CURR'].isin(X_base.index)].set_index('SK_ID_CURR')

prev_app = pd.read_csv('previous_application.csv')
prev_app = prev_app[prev_app['SK_ID_CURR'].isin(X_base.index)].set_index('SK_ID_CURR')

bureau_bal = pd.read_csv('bureau_balance.csv') # DOES NOT HAVE THE index column from others

id_map = pd.Series(bureau.index, index=bureau.SK_ID_BUREAU)
bureau_bal['SK_ID_CURR'] = bureau_bal['SK_ID_BUREAU'].map(id_map)
bureau_bal = bureau_bal.dropna(subset=['SK_ID_CURR'])
bureau_bal['SK_ID_CURR'] = bureau_bal['SK_ID_CURR'].map(lambda p: int(p))
bureau_bal = bureau_bal.set_index('SK_ID_CURR')

_log('Auxiliary CSVs loaded and linked')

# Aggregate data of auxiliary CSV's

def primary_mode(x): # Helper function since 'mode' does not work in pandas
    mode_vals = x.mode()
    return mode_vals.iloc[0] if not mode_vals.empty else None


ccb_agg = ccb.groupby('SK_ID_CURR').agg({
    'AMT_BALANCE': ['mean', 'max', 'sum'],
    'AMT_PAYMENT_CURRENT': ['mean'],
    'AMT_TOTAL_RECEIVABLE': ['mean'],
    'CNT_INSTALMENT_MATURE_CUM': ['mean', 'sum'] })

install_agg = install_pay.groupby('SK_ID_CURR').agg({
    'NUM_INSTALMENT_NUMBER': ['mean', 'sum'],
    'DAYS_INSTALMENT' : ['mean', 'max'],
    'AMT_INSTALMENT' : ['mean', 'sum'],
    'AMT_PAYMENT' : ['mean', 'sum'] })

b_balance_agg = bureau_bal.groupby('SK_ID_CURR').agg({
    'STATUS' : [primary_mode],
    'MONTHS_BALANCE': ['min', 'max', 'mean']    })

bureau_agg = bureau.groupby('SK_ID_CURR').agg({
    'CREDIT_ACTIVE' : [primary_mode],
    'DAYS_CREDIT' : ['mean'],
    'CREDIT_DAY_OVERDUE' : ['mean', 'max'],
    'AMT_CREDIT_SUM': ['mean'],
    'AMT_CREDIT_SUM_DEBT': ['mean'],
    'CREDIT_TYPE' : [primary_mode]    })

prev_agg = prev_app.groupby('SK_ID_CURR').agg({
    'NAME_CONTRACT_TYPE' : [primary_mode],
    'RATE_DOWN_PAYMENT' : ['mean'],
    'NAME_CASH_LOAN_PURPOSE': [primary_mode],
    'NAME_CONTRACT_STATUS': [primary_mode],
    'CNT_PAYMENT' : ['mean']
})

data = pd.concat([X_base, ccb_agg, bureau_agg, b_balance_agg, install_agg, prev_agg], axis=1)

# Add domain knowledge features 

data['CREDIT_INCOME_PERCENT'] = data['AMT_CREDIT'] / data['AMT_INCOME_TOTAL']
data['ANNUITY_INCOME_PERCENT'] = data['AMT_ANNUITY'] / data['AMT_INCOME_TOTAL']
data['CREDIT_TERM'] = data['AMT_ANNUITY'] / data['AMT_CREDIT']
data['DAYS_EMPLOYED_PERCENT'] = data['DAYS_EMPLOYED'] / data['DAYS_BIRTH']

_log('Aggregated features merged and engineered')

#--------------------------------------------------------------------------------------------
# Imputing/encoding of data used for tree-based methods:
#--------------------------------------------------------------------------------------------

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

"""
We apply the following transformation policy:
 - Numeric data is imputed via medians
 - Obj data with k=2 categories is encoder ordinally
 - Obj data with k>2 categories is encoded with OH encoding
"""

def _clean(df):
    df = df.copy()
    df.columns = ['_'.join(map(str, col)).strip() if isinstance(col, tuple) else str(col) for col in df.columns]
    return df.replace([np.inf, -np.inf], np.nan)

def fit_transformer(df):
    """Fit column groupings + encoders on TRAIN only."""
    df = _clean(df)
    cat = df.select_dtypes(exclude='number')
    ord_cols = [c for c in cat.columns if cat[c].fillna('Missing').nunique() <= 2]
    oh_cols  = [c for c in cat.columns if cat[c].fillna('Missing').nunique() > 2]
    num_cols = list(df.select_dtypes(include='number').columns)

    numeric_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')),
                              ('scaler', MinMaxScaler(feature_range=(0, 1)))])
    ord_pipe = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
                          ('encoder', OrdinalEncoder())])
    oh_pipe = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
                         ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    preprocess = ColumnTransformer([('num', numeric_pipe, num_cols),
                                     ('ord', ord_pipe, ord_cols),
                                     ('oh',  oh_pipe,  oh_cols)])

    arr = preprocess.fit_transform(df)
    if oh_cols:
        oh_names = preprocess.named_transformers_['oh'].named_steps['encoder'].get_feature_names_out(oh_cols)
        feature_names = num_cols + ord_cols + list(oh_names)
    else:
        feature_names = num_cols + ord_cols

    return preprocess, feature_names, pd.DataFrame(arr, columns=feature_names, index=df.index)

def apply_transformer(df, preprocess, feature_names):
    """Transform any other split (e.g. test) with the ALREADY-FITTED preprocessor."""
    df = _clean(df)
    arr = preprocess.transform(df)
    return pd.DataFrame(arr, columns=feature_names, index=df.index)


from sklearn.model_selection import train_test_split #Split before transforming to prevent leakage

train_base, test_base, y_train_base, y_test_base = train_test_split(
    X_base, y, test_size=0.2, random_state=10, stratify=y) # Used for comparison later on, does not have aux csv's

preprocess_base, feature_names_base, train_ts_base = fit_transformer(train_base)
test_ts_base = apply_transformer(test_base, preprocess_base, feature_names_base)

train, test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=10, stratify=y)
preprocess, feature_names, train_ts = fit_transformer(train)
test_ts = apply_transformer(test, preprocess, feature_names)

_log('Encoding/imputation complete (base + full feature sets)')

#--------------------------------------------------------------------------------------------
# WoE transformation of data used for final Logistic regression
#--------------------------------------------------------------------------------------------

from optbinning import BinningProcess

train_raw = _clean(train)
test_raw  = _clean(test)

# Identify categoricals from the RAW (unencoded) frame -- optbinning bins
# numeric and categorical variables in one pass, so we skip separate
# ordinal/one-hot handling entirely for this track
cat_cols = list(train_raw.select_dtypes(exclude='number').columns)
variable_names = list(train_raw.columns)


binning_process = BinningProcess(
    variable_names=variable_names,
    categorical_variables=cat_cols,
    min_bin_size=0.05,
    selection_criteria={'iv': {'min': 1e-6}},  # drops only iv==0.0 columns, not "weak" ones
)

# Fit ONLY on train
_log('Starting WoE binning fit...')
with _heartbeat('WoE binning fit'):
    binning_process.fit(train_raw[variable_names], y_train)

# Apply the learned bin edges/WoE values to both splits
train_woe = binning_process.transform(train_raw[variable_names], metric='woe')
test_woe  = binning_process.transform(test_raw[variable_names], metric='woe')

_log('WoE binning complete')

""" NOTE: train/test samples guide:

 - test_base/train_base: Unprocessed data with the base features from application_... .csv
 - test/train: Unprocessed (no imputing/encoding) data with the full feature set
 - test_ts_base/train_ts_base: Enconded and imputed data wit base features
 - test_ts/train_ts: Encoded and imputed data with all features
 - test_woe/train_woe: WoE trasnformed data with full features and processing done by WoE itself
"""

#--------------------------------------------------------------------------------------------
# Naive Random Forest and Logistic regression
#--------------------------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

naive_forest = RandomForestClassifier(random_state=10)
naive_logistic = LogisticRegression()

naive_forest.fit(train_ts_base, y_train_base)
naive_forest_probas = naive_forest.predict_proba(test_ts_base)[:, 1]
naive_forest_auc = roc_auc_score(y_test_base, naive_forest_probas)

naive_logistic.fit(train_ts_base, y_train_base)
naive_logistic_probas = naive_logistic.predict_proba(test_ts_base)[:, 1]
naive_logistic_auc = roc_auc_score(y_test_base, naive_logistic_probas)

_log(f'Naive models done - RF AUC={naive_forest_auc:.4f}, LogReg AUC={naive_logistic_auc:.4f}')


#--------------------------------------------------------------------------------------------
# Improved Logistic regression (via WoE transformation and HP search)
#--------------------------------------------------------------------------------------------



from sklearn.model_selection import GridSearchCV

log_reg = LogisticRegression(max_iter=1000) 

param_grid = [
    {
        'solver': ['saga'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'l1_ratio': [0.0, 0.25, 0.5, 1.0] 
    }
]

grid_search = GridSearchCV(log_reg, param_grid, cv=3, scoring='roc_auc', return_train_score=True,n_jobs=-1)

_log('Starting grid search (WoE logistic regression)...')
with _heartbeat('grid search (WoE logistic)'):
    grid_search.fit(train_woe, y_train)

best_woe_model = grid_search.best_estimator_  # GridSearchCV refits on full train by default
woe_logistic_probas = best_woe_model.predict_proba(test_woe)[:, 1]
woe_logistic_auc = roc_auc_score(y_test, woe_logistic_probas)

_log(f'Grid search done - WoE logistic AUC={woe_logistic_auc:.4f}')


#--------------------------------------------------------------------------------------------
# Improved tree-based method (via boosting) and SHAP analysis
#--------------------------------------------------------------------------------------------

from xgboost import XGBClassifier
# We use an early stopping set instead of HP search
# We do NOT employ the WoE transform here, just transformed data (see guide on train/test samples)

X_tr, X_val, y_tr, y_val = train_test_split(train_ts, y_train, test_size=0.15, stratify=y_train)

model_fixed = XGBClassifier(
    n_estimators=2000,       # set high; early stopping finds the real number
    learning_rate=0.03,
    max_depth=5,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective='binary:logistic',
    eval_metric='auc',
    early_stopping_rounds=50,
    tree_method='hist',
    random_state=42,
)

_log('Starting XGBoost early-stopping fit...')
with _heartbeat('xgboost early-stopping fit'):
    model_fixed.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

optimal_trees = model_fixed.best_iteration + 1
_log(f'Early stopping found optimal_trees={optimal_trees}')

boost_fin = XGBClassifier(
    n_estimators=optimal_trees, # Using the exact number found above
    learning_rate=0.03,
    max_depth=5,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective='binary:logistic',
    tree_method='hist',
    random_state=42,
)

with _heartbeat('xgboost final fit'):
    boost_fin.fit(train_ts, y_train)
boost_preds = boost_fin.predict_proba(test_ts)[:, 1]
boost_auc = roc_auc_score(y_test, boost_preds)

_log(f'Boosted model done - AUC={boost_auc:.4f}')


import shap


explainer = shap.TreeExplainer(boost_fin)

with _heartbeat('SHAP value computation'):
    shap_values_sample = explainer(test_ts.sample(2000, random_state=10))

_log('SHAP values computed - all stages complete')

print(f'Naive Tree-based classifier (RF): {naive_forest_auc}')
print(f'Naive Logistic regressiom: {naive_logistic_auc}')
print(f'WoE transformed logistic regression: {woe_logistic_auc}')
print(f'Boosted Tree-based classifier: {boost_auc}')


#--------------------------------------------------------------------------------------------
# Figures
#--------------------------------------------------------------------------------------------