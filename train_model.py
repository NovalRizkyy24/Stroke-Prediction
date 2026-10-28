import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'data', 'dataset-stroke.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pkl')
LABEL_ENCODERS_PATH = os.path.join(MODEL_DIR, 'fitted_label_encoders.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'fitted_min_max_scaler.pkl')
METRICS_PATH = os.path.join(MODEL_DIR, 'model_metrics.json')

# Probabilitas minimum untuk dikategorikan "Risiko Stroke Tinggi" (harus sama dengan app.py)
THRESHOLD = 0.6

# 1. Memuat dan membersihkan data
Data = pd.read_csv(DATASET_PATH)
Data = Data[Data['gender'] != 'Other']  # hanya 1 baris
Data['bmi'] = Data['bmi'].fillna(Data['bmi'].median()).clip(10.3, 43)  # sesuai batas form web

bins = [0, 18, 36, 54, 72, 90]
labels = ['0-18', '18-36', '36-54', '54-72', '72-90']
Data['age'] = pd.cut(Data['age'], bins=bins, labels=labels).astype(str)

# 2. Encoding: satu LabelEncoder per kolom agar bisa dipakai ulang di app.py
cat_cols = ['gender', 'age', 'hypertension', 'heart_disease',
            'ever_married', 'work_type', 'Residence_type', 'smoking_status']
fitted_label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    Data[col] = le.fit_transform(Data[col])
    fitted_label_encoders[col] = le

# 3. Split SEBELUM penanganan ketidakseimbangan; data uji tetap berproporsi asli
X = Data.drop(['id', 'stroke'], axis=1)
y = Data['stroke']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
print(f"Data latih: {len(y_train)} ({y_train.sum()} stroke) | Data uji: {len(y_test)} ({y_test.sum()} stroke)")

# 4. Normalisasi: scaler hanya di-fit pada data latih
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train.values)
X_test_scaled = scaler.transform(X_test.values)

# 5. Logistic Regression dengan class_weight='balanced', C dipilih lewat 5-fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    LogisticRegression(class_weight='balanced', max_iter=1000),
    {'C': [0.01, 0.1, 1, 10]},
    scoring='roc_auc',
    cv=cv,
)
grid.fit(X_train_scaled, y_train)
model = grid.best_estimator_
print(f"Hyperparameter terbaik: {grid.best_params_} (CV ROC-AUC: {grid.best_score_:.3f})")

# Pembanding: algoritma lain dengan CV yang sama, untuk menunjukkan alasan pemilihan model
comparison_models = {
    'SVC (linear)': (SVC(kernel='linear', class_weight='balanced', probability=True, random_state=42),
                     {'C': [0.1, 1]}),
    'Random Forest': (RandomForestClassifier(class_weight='balanced', n_estimators=200, random_state=42),
                      {'max_depth': [4, 6, 8], 'min_samples_leaf': [5, 20]}),
    'Gradient Boosting': (HistGradientBoostingClassifier(class_weight='balanced', random_state=42),
                          {'max_depth': [2, 3], 'learning_rate': [0.03, 0.1]}),
}
comparison = [{'model': 'Logistic Regression', 'cv_roc_auc': grid.best_score_, 'selected': True}]
for name, (estimator, params) in comparison_models.items():
    other = GridSearchCV(estimator, params, scoring='roc_auc', cv=cv, n_jobs=-1).fit(X_train_scaled, y_train)
    comparison.append({'model': name, 'cv_roc_auc': other.best_score_, 'selected': False})
    print(f"Pembanding {name}: CV ROC-AUC {other.best_score_:.3f}")

# 6. Evaluasi sekali pada data uji dengan threshold yang dipakai aplikasi
proba = model.predict_proba(X_test_scaled)[:, 1]
pred = (proba >= THRESHOLD).astype(int)

print(f"\n--- Evaluasi Data Uji (threshold {THRESHOLD}) ---")
print(f"Recall   : {recall_score(y_test, pred):.3f}")
print(f"Precision: {precision_score(y_test, pred):.3f}")
print(f"F1-Score : {f1_score(y_test, pred):.3f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, proba):.3f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, target_names=['Risiko Rendah', 'Risiko Tinggi']))

coef = pd.Series(model.coef_[0], index=X.columns).sort_values(ascending=False)
print("Koefisien fitur (positif = menaikkan risiko):")
print(coef.round(3).to_string())

# 7. Menyimpan model dan transformer untuk app.py
os.makedirs(MODEL_DIR, exist_ok=True)
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)
with open(LABEL_ENCODERS_PATH, 'wb') as f:
    pickle.dump(fitted_label_encoders, f)
with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)

# 8. Menyimpan hasil evaluasi untuk ditampilkan di website
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
metrics = {
    'model': 'Logistic Regression',
    'params': {**grid.best_params_, 'class_weight': 'balanced'},
    'threshold': THRESHOLD,
    'n_train': int(len(y_train)),
    'n_test': int(len(y_test)),
    'n_test_stroke': int(y_test.sum()),
    'recall': recall_score(y_test, pred),
    'precision': precision_score(y_test, pred),
    'f1': f1_score(y_test, pred),
    'roc_auc': roc_auc_score(y_test, proba),
    'cv_roc_auc': grid.best_score_,
    'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
    'coefficients': coef.round(3).to_dict(),
    'comparison': sorted(comparison, key=lambda item: -item['cv_roc_auc']),
}
with open(METRICS_PATH, 'w', encoding='utf-8') as f:
    json.dump(metrics, f, indent=2)

print(f"\nModel, transformer, dan metrik disimpan di folder: {MODEL_DIR}")
