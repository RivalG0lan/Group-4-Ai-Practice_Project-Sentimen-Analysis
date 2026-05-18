# ─── Auto-install dependencies ────────────────────────────────────────────────
import subprocess
import sys

REQUIRED = [
    'flask',
    'flask-cors',
    'joblib',
    'numpy',
    'pandas',
    'scikit-learn',
    'openpyxl',
]

def install_if_missing(packages):
    import importlib
    # mapping nama pip → nama import
    pip_to_import = {
        'flask-cors': 'flask_cors',
        'scikit-learn': 'sklearn',
        'pillow': 'PIL',
    }
    for pkg in packages:
        import_name = pip_to_import.get(pkg, pkg)
        try:
            importlib.import_module(import_name)
        except ImportError:
            print(f"[AUTO-INSTALL] '{pkg}' tidak ditemukan, menginstall...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])
            print(f"[AUTO-INSTALL] '{pkg}' berhasil diinstall!")

install_if_missing(REQUIRED)
# ──────────────────────────────────────────────────────────────────────────────

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
# ... sisa import seperti biasa
import joblib
import numpy as np
import pandas as pd
import os
import re
from collections import Counter

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
tfidf = joblib.load(os.path.join(BASE_DIR, 'model_tfidf.joblib'))
model = joblib.load(os.path.join(BASE_DIR, 'sentiment_model.joblib'))

LABEL_MAP   = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
LABEL_EMOJI = {0: '😡', 1: '😐', 2: '😊'}

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_one(text: str) -> dict:
    cleaned = clean_text(text)
    vec     = tfidf.transform([cleaned])
    label   = int(model.predict(vec)[0])
    proba   = model.predict_proba(vec)[0]
    return {
        'review'    : text,
        'label'     : label,
        'sentimen'  : LABEL_MAP[label],
        'emoji'     : LABEL_EMOJI[label],
        'confidence': {
            'Negatif': round(float(proba[0]) * 100, 1),
            'Netral' : round(float(proba[1]) * 100, 1),
            'Positif': round(float(proba[2]) * 100, 1),
        }
    }

def top_words(texts, n=15):
    all_words = []
    for t in texts:
        words = clean_text(str(t)).split()
        all_words.extend([w for w in words if len(w) > 3])
    counter = Counter(all_words)
    return [{'word': w, 'count': c} for w, c in counter.most_common(n)]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/text', methods=['POST'])
def predict_text():
    data = request.get_json()
    text = data.get('review', '').strip()
    if not text:
        return jsonify({'error': 'Review kosong'}), 400
    return jsonify(predict_one(text))

@app.route('/predict/file', methods=['POST'])
def predict_file():
    print("=== FILE DITERIMA ===")
    if 'file' not in request.files:
        return jsonify({'error': 'File tidak ditemukan'}), 400

    f = request.files['file']
    print(f"Nama file: {f.filename}")

    try:
        if f.filename.lower().endswith('.csv'):
            df = pd.read_csv(f)
        else:
            df = pd.read_excel(f)
        print(f"Shape: {df.shape}, Kolom: {df.columns.tolist()}")
    except Exception as e:
        print(f"ERROR baca file: {e}")
        return jsonify({'error': f'Gagal membaca file: {str(e)}'}), 400

    text_col = None
    for candidate in ['review_cleaned', 'review', 'ulasan', 'text', 'comment']:
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col is None:
        text_col = df.columns[0]
    print(f"Kolom dipakai: {text_col}")

    df = df.dropna(subset=[text_col])
    df[text_col] = df[text_col].astype(str)
    print(f"Jumlah baris: {len(df)}")

    try:
        texts = df[text_col].tolist()
        cleaned_texts = [clean_text(t) for t in texts]
        vecs       = tfidf.transform(cleaned_texts)
        labels_arr = model.predict(vecs)
        probas     = model.predict_proba(vecs)

        results = []
        for i, text in enumerate(texts):
            label = int(labels_arr[i])
            proba = probas[i]
            results.append({
                'review'    : text,
                'label'     : label,
                'sentimen'  : LABEL_MAP[label],
                'emoji'     : LABEL_EMOJI[label],
                'confidence': {
                    'Negatif': round(float(proba[0]) * 100, 1),
                    'Netral' : round(float(proba[1]) * 100, 1),
                    'Positif': round(float(proba[2]) * 100, 1),
                }
            })
        print(f"Prediksi selesai: {len(results)} hasil")
    except Exception as e:
        print(f"ERROR prediksi: {e}")
        return jsonify({'error': f'Gagal prediksi: {str(e)}'}), 500

    labels    = [r['label'] for r in results]
    sentimens = [r['sentimen'] for r in results]

    dist  = Counter(sentimens)
    total = len(results)
    distribusi = {s: {'count': dist.get(s, 0),
                      'pct'  : round(dist.get(s, 0) / total * 100, 1)}
                  for s in ['Positif', 'Netral', 'Negatif']}

    pos_texts = [texts[i] for i, l in enumerate(labels) if l == 2]
    neg_texts = [texts[i] for i, l in enumerate(labels) if l == 0]

    return jsonify({
        'total'      : total,
        'distribusi' : distribusi,
        'top_positif': top_words(pos_texts),
        'top_negatif': top_words(neg_texts),
        'samples'    : results[:5],
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
    
    
    
#     Loop satu-satu (sebelumnya):
# setiap predict_one() dipanggil 67906 kali, dan tiap kali itu Python harus: buat sparse matrix baru → panggil fungsi predict → panggil fungsi proba → return. Overhead pemanggilan fungsi Python itu mahal kalau diulang puluhan ribu kali.
# Batch (sekarang):
# tfidf.transform(cleaned_texts) memproses semua 67906 teks sekaligus menjadi satu sparse matrix besar, lalu model.predict(vecs) dan model.predict_proba(vecs) beroperasi pada matrix itu dalam satu operasi numpy/scipy.
# Intinya, TF-IDF dan Naive Bayes di balik layar adalah operasi matrix multiplication — dan numpy/scipy sangat dioptimasi untuk perkalian matrix besar sekaligus menggunakan BLAS/LAPACK yang berjalan di level C, bukan Python. Jadi daripada 67906 perkalian matrix kecil yang masing-masing punya overhead, kamu melakukan 1 perkalian matrix besar yang jauh lebih efisien secara CPU cache dan memory access.
# Analogi sederhananya: seperti perbedaan antara bolak-balik ke dapur 67906 kali untuk ambil 1 piring vs sekali jalan bawa semua 67906 piring sekaligus pakai troli.