import os
from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

MODEL_PATH = 'best_model.pkl'
LABEL_ENCODERS_PATH = 'fitted_label_encoders.pkl'
SCALER_PATH = 'fitted_min_max_scaler.pkl'

try:
    with open(MODEL_PATH, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(LABEL_ENCODERS_PATH, 'rb') as le_file:
        fitted_label_encoders = pickle.load(le_file)
    with open(SCALER_PATH, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    print("Model dan transformer berhasil dimuat.")
except FileNotFoundError as e:
    print(f"Error: File tidak ditemukan: {e}. Pastikan Anda telah menjalankan prepare_models_and_transformers.py terlebih dahulu.")
    print("Aplikasi tidak dapat berjalan tanpa file-file ini.")
    exit()
except Exception as e:
    print(f"Terjadi kesalahan saat memuat model atau transformer: {e}")
    exit()

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

            prediction = model.predict(scaled_input)[0]
            prediction_proba = model.predict_proba(scaled_input)[0]

            result_text = "Terkena Stroke" if prediction == 1 else "Tidak Terkena Stroke"
            
            result_color_class = "text-red" if prediction == 1 else "text-green"

            confidence_stroke = prediction_proba[1] * 100
            confidence_no_stroke = prediction_proba[0] * 100

            return render_template('index.html',
                                   prediction_text=f'Berdasarkan data yang Anda masukkan, prediksi adalah: {result_text}.',
                                   result_color_class=result_color_class, 
                                   confidence_stroke=f'Keyakinan Terkena Stroke: {confidence_stroke:.2f}%',
                                   confidence_no_stroke=f'Keyakinan Tidak Pernah Stroke: {confidence_no_stroke:.2f}%',
                                   form_data=request.form) 

        except ValueError as ve:
            return render_template('index.html', prediction_text=f"Input tidak valid: {ve}. Pastikan semua kolom numerik diisi dengan angka.", form_data=request.form)
        except Exception as e:
            return render_template('index.html', prediction_text=f"Terjadi kesalahan saat memproses prediksi: {e}", form_data=request.form)

if __name__ == '__main__':
    if not os.path.exists('templates'):
        os.makedirs('templates')

    app.run(debug=True)
