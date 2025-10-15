import pandas as pd
import numpy as np
import joblib, os
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, classification_report
from imblearn.over_sampling import SMOTE

from utils import evaluate_threshold_metrics

# update as required
version = 'v1'
input_file_path = 'bank/bank-full.csv'
output_file_path = 'bank/outputs/'

# ensure the output directory exists
os.makedirs(output_file_path, exist_ok=True)

# these should remain consistent across datasets - do not adjust
cols_to_cat = ['job', 'marital',  'education', 'contact', 'month', 'poutcome']
cols_to_binary = ['loan','housing','default','y']
target_col = 'y'


### DATA ENGINEERING

# read in data
df = pd.read_csv(input_file_path, sep=";")
# df.head()

# handle missing values in new data
missing_summary = df.isnull().sum()

if missing_summary.any():
    print("Warning: Missing values detected. Applying default handling...")

    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(-1, inplace=True)
            else:
                df[col].fillna('Unknown', inplace=True)
else:
    print("No missing values detected.")

# handle any dupes in new data
num_duplicates = df.duplicated().sum()
if num_duplicates > 0:
    print(f"Warning: {num_duplicates} duplicate rows found. Dropping...")
    df = df.drop_duplicates()
else:
    print("No duplicate rows detected.")

# convert column types
df[cols_to_cat] = df[cols_to_cat].astype('category')

for col in cols_to_binary:
    df[col] = df[col].map({'no': 0, 'yes': 1}) 

# drop duration (model leakage)
df = df.drop('duration', axis=1)

# create column ID for predictions mapping
df['id'] = range(len(df))

# feature engineering (limited scope)
df['previously_contacted'] = (df['pdays'] != -1).astype(int)

# model setup
X_raw = df.drop(columns=[target_col, 'id'])
y = df[target_col]
ids = df['id']

#  encode categorical columns
cat_cols = X_raw.select_dtypes(include=['category']).columns.tolist()
num_cols = X_raw.select_dtypes(include=['number']).columns.tolist()

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_cat_encoded = encoder.fit_transform(X_raw[cat_cols])
encoded_col_names = encoder.get_feature_names_out(cat_cols)

X_cat_df = pd.DataFrame(X_cat_encoded, columns=encoded_col_names, index=X_raw.index)

# combine encoded and numericall features
X_train = pd.concat([X_cat_df, X_raw[num_cols]], axis=1)

X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X_train, y, ids,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# apply SMOTE to training data (only training data so we don't cause data leakage which will lead to over optimistic performance metrics)
X_train_cols = X_train.columns.tolist()
cat_col_indices = [X_train_cols.index(col) for col in cat_cols if col in X_train_cols]
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f'Original training set shape: {X_train.shape}, {y_train.shape}')
print("X_train shape after SMOTE:", X_train_resampled.shape)

### MODELLING

### MODEL SANDBOX ###

# # RF
# from sklearn.ensemble import RandomForestClassifier
#
# model = RandomForestClassifier(
#     n_estimators=100,
#     max_features=10,
#     # class_weight='balanced',
#     random_state=42
#     )
# model.fit(X_train_resampled, y_train_resampled)

# # XGB
# import xgboost as xgb

# model = xgb.XGBClassifier(
#     scale_pos_weight=1,
#     learning_rate=0.1, 
#     max_depth=5,
#     )
# model.fit(X_train_resampled, y_train_resampled)

## assess performance
# y_pred = model.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
# print(classification_report(y_test, y_pred))

### END MODEL SANDBOX ###

## selected model - hyperparameter tuning
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats
from scipy.stats import uniform, randint

param_dist = {
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.05),
    'n_estimators': randint(100, 300),
}

xgb_model = xgb.XGBClassifier(random_state=7)

random_search = RandomizedSearchCV(xgb_model, param_distributions=param_dist, n_iter=10, cv=5, scoring='f1', random_state=7)
random_search.fit(X_train_resampled, y_train_resampled)

model = random_search.best_estimator_

# Save model to file
joblib.dump(model, f'{output_file_path}/xgb_model_{version}.pkl')

### INFERENCE

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

## feature importances and interpretation

importances = model.feature_importances_

feat_imp = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

feat_imp.to_csv(f'{output_file_path}/feature_importance_{version}.csv', index=False)

# Plot
plt.figure(figsize=(6, 10))
plt.barh(feat_imp['Feature'][::-1], feat_imp['Importance'][::-1])
plt.title('Top Feature Importances')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig(f'{output_file_path}/feature_importance_chart_{version}.png')
plt.close()

print(feat_imp.head(10))

### OUTPUTS 

## Prediction outputs: we want to prioritise candidates

# return a list of leads ordered by propensity
y_probs = model.predict_proba(X_test)[:, 1]
# compute F1 score for each threshold to find the optimal cutoff for probs
best_threshold, best_f1 = evaluate_threshold_metrics(
    y_test, 
    y_probs, 
    show_plot=False,
    save_path=f'{output_file_path}/pr_threshold_plot_{version}.png'
)
# export ranked list of predictions for insights team to reach out to - also filtered by best threshold for key prospects
results = pd.DataFrame({
    'id': ids_test,
    'predicted_prob': y_probs
}).sort_values(by='predicted_prob', ascending=False)

prospects = results[results['predicted_prob']>=best_threshold]

# export probs
results.to_csv(f'{output_file_path}/full_probs_{version}.csv', index=False)
prospects.to_csv(f'{output_file_path}/prospects_{version}.csv', index=False)

# print(results.sort_values('predicted_prob', ascending=False).head(10))

