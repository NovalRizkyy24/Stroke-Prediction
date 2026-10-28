import os
import json
from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pkl')
LABEL_ENCODERS_PATH = os.path.join(MODEL_DIR, 'fitted_label_encoders.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'fitted_min_max_scaler.pkl')
METRICS_PATH = os.path.join(MODEL_DIR, 'model_metrics.json')

# Nama fitur untuk ditampilkan di bagian evaluasi model
FEATURE_LABELS = {
    'age': 'Usia',
    'avg_glucose_level': 'Kadar Glukosa',
    'hypertension': 'Hipertensi',
    'heart_disease': 'Penyakit Jantung',
    'ever_married': 'Status Pernikahan',
    'Residence_type': 'Tempat Tinggal',
    'bmi': 'BMI',
    'smoking_status': 'Status Merokok',
    'work_type': 'Jenis Pekerjaan',
    'gender': 'Jenis Kelamin',
}

# Fitur kategorikal tanpa urutan: LabelEncoder memberi angka menurut abjad, sehingga arah koefisiennya tidak bermakna
NOMINAL_FEATURES = {'work_type', 'smoking_status'}

# Probabilitas minimum untuk dikategorikan "Risiko Stroke Tinggi" (harus sama dengan train_model.py)
THRESHOLD = 0.6

try:
    with open(MODEL_PATH, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(LABEL_ENCODERS_PATH, 'rb') as le_file:
        fitted_label_encoders = pickle.load(le_file)
    with open(SCALER_PATH, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    print("Model dan transformer berhasil dimuat.")
except FileNotFoundError as e:
    print(f"Error: File tidak ditemukan: {e}. Pastikan Anda telah menjalankan train_model.py terlebih dahulu.")
    print("Aplikasi tidak dapat berjalan tanpa file-file ini.")
    exit()
except Exception as e:
    print(f"Terjadi kesalahan saat memuat model atau transformer: {e}")
    exit()

# Metrik evaluasi bersifat opsional: website tetap berjalan tanpa bagian evaluasi jika file tidak ada
try:
    with open(METRICS_PATH, encoding='utf-8') as metrics_file:
        model_metrics = json.load(metrics_file)
    max_coef = max(abs(v) for v in model_metrics['coefficients'].values())
    model_metrics['features'] = [
        {'label': FEATURE_LABELS.get(name, name), 'coef': coef, 'width': abs(coef) / max_coef * 100,
         'nominal': name in NOMINAL_FEATURES}
        for name, coef in sorted(model_metrics['coefficients'].items(), key=lambda item: -abs(item[1]))
    ]
    model_metrics['prevalence'] = model_metrics['n_test_stroke'] / model_metrics['n_test']
    model_metrics['precision_lift'] = model_metrics['precision'] / model_metrics['prevalence']
    model_metrics['comparison'] = model_metrics.get('comparison', [])
    scores = [row['cv_roc_auc'] for row in model_metrics['comparison']]
    model_metrics['comparison_spread'] = max(scores) - min(scores) if scores else 0
except (FileNotFoundError, KeyError, ValueError) as e:
    print(f"Metrik evaluasi tidak dimuat: {e}")
    model_metrics = None

@app.context_processor
def inject_model_metrics():
    """Menyediakan metrik evaluasi model untuk semua halaman."""
    return {'metrics': model_metrics}

@app.route('/')
def home():
    """Menampilkan halaman utama dengan formulir input."""
    return render_template('index.html', form_data={})

@app.route('/predict', methods=['POST'])
def predict():
    """Menerima input dari formulir, melakukan preprocessing, dan memprediksi risiko stroke."""
    if request.method == 'POST':
        try:
            gender = request.form['gender']
            age = request.form['age']
            hypertension = int(request.form['hypertension']) # 0 or 1
            heart_disease = int(request.form['heart_disease']) # 0 or 1
            ever_married = request.form['ever_married']
            work_type = request.form['work_type']
            residence_type = request.form['residence_type']
            avg_glucose_level = float(request.form['avg_glucose_level'])
            bmi = float(request.form['bmi'])
            smoking_status = request.form['smoking_status']

            input_df = pd.DataFrame([[
                gender, age, hypertension, heart_disease, ever_married,
                work_type, residence_type, avg_glucose_level, bmi, smoking_status
            ]], columns=[
                'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
                'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
            ])

            print(f"\n--- Input Mentah (DataFrame): ---\n{input_df}")

            cols_to_transform_with_le = ['gender', 'age', 'hypertension', 'heart_disease',
                                          'ever_married', 'work_type', 'Residence_type', 'smoking_status']

            for col in cols_to_transform_with_le:
                val_to_encode = input_df[col].iloc[0]
                input_df[col] = fitted_label_encoders[col].transform(np.array([val_to_encode]).reshape(1, -1))[0]

            print(f"\n--- Input Setelah Label Encoding (DataFrame): ---\n{input_df}")

            scaled_input = scaler.transform(input_df.values)

            print(f"\n--- Input Setelah MinMax Scaling (Array NumPy): ---\n{scaled_input}")
            print(f"--- Bentuk Input Setelah Scaling: {scaled_input.shape} ---")

            prediction_proba = model.predict_proba(scaled_input)[0]
            prediction = 1 if prediction_proba[1] >= THRESHOLD else 0

            result_text = "Risiko Stroke Tinggi" if prediction == 1 else "Risiko Stroke Rendah"

            result_color_class = "text-red" if prediction == 1 else "text-green"

            confidence_stroke = prediction_proba[1] * 100

            return render_template('index.html',
                                   prediction_text=f'Berdasarkan data yang Anda masukkan, hasil skrining: {result_text}.',
                                   result_color_class=result_color_class,
                                   confidence_stroke=f'Skor Risiko Stroke: {confidence_stroke:.2f}% (batas risiko tinggi: {THRESHOLD * 100:.0f}%)',
                                   confidence_no_stroke='Hasil ini bukan diagnosis medis. Konsultasikan dengan dokter untuk pemeriksaan lebih lanjut.',
                                   form_data=request.form.to_dict()) 

        except ValueError as ve:
            return render_template('index.html', prediction_text=f"Input tidak valid: {ve}. Pastikan semua kolom numerik diisi dengan angka.", form_data=request.form.to_dict())
        except Exception as e:
            return render_template('index.html', prediction_text=f"Terjadi kesalahan saat memproses prediksi: {e}", form_data=request.form.to_dict())

if __name__ == '__main__':
    app.run(debug=True)
