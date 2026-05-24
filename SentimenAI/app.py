#Auto-install dependencies library python
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
    pip_to_import = {
        'flask-cors': 'flask_cors',
        'scikit-learn': 'sklearn',
        'pillow': 'PIL',
    }
    all_ok = True
    for pkg in packages:
        import_name = pip_to_import.get(pkg, pkg)
        try:
            importlib.import_module(import_name)
        except ImportError:
            all_ok = False
            print(f"[AUTO-INSTALL] '{pkg}' tidak ditemukan, menginstall...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])
            print(f"[AUTO-INSTALL] '{pkg}' berhasil diinstall!")
 
    if all_ok:
        print("[✓] Semua dependencies sudah terinstall.")
    else:
        print("[✓] Dependencies selesai diinstall, siap jalan.")
 
install_if_missing(REQUIRED)

 
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
import re
import urllib.request
import urllib.error
import json
import time
from collections import Counter
 
app = Flask(__name__)
CORS(app)
 
# ngeload model dan vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
tfidf = joblib.load(os.path.join(BASE_DIR, 'model_tfidf.joblib'))
model = joblib.load(os.path.join(BASE_DIR, 'sentiment_model.joblib'))
 
LABEL_MAP   = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
LABEL_EMOJI = {0: '😡', 1: '😐', 2: '😊'}
 
# helper
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
 
#rute
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
 
    # cari kolom teks
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
        texts         = df[text_col].tolist()
        cleaned_texts = [clean_text(t) for t in texts]
        vecs          = tfidf.transform(cleaned_texts)
        labels_arr    = model.predict(vecs)
        probas        = model.predict_proba(vecs)
 
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
    distribusi = {
        s: {
            'count': dist.get(s, 0),
            'pct': round(dist.get(s, 0) / total * 100, 1) if total > 0 else 0
        }
        for s in ['Positif', 'Netral', 'Negatif']
    }
 
    pos_texts = [texts[i] for i, l in enumerate(labels) if l == 2]
    neg_texts = [texts[i] for i, l in enumerate(labels) if l == 0]
 
    return jsonify({
        'total'       : total,
        'distribusi'  : distribusi,
        'top_positif' : top_words(pos_texts),
        'top_negatif' : top_words(neg_texts),
        'samples'     : results[:5],   # kompatibilitas fallback
        'samples_all': results,       # semua data, saranku results[:500] 500 data aja, biar gak berat ram browser, bisa aja sih semua wkwk
    })


# rute baru buat sistem pendukung keputusan berbasis ai gemini. perlu urlib request, import json dan time 
# disini gemini ada limit request per menit, jadi kita buat mekanisme retry kalau kena 429 Too Many Requests. 
# Kita coba sampai 3x dengan jeda yang meningkat (20s, 40s, 60s) sebelum akhirnya menyerah dan minta user coba 
# lagi nanti.

# Cache insight agar tidak spam Gemini
_insight_cache = {'teks': None, 'key': None}

@app.route('/insight/gemini', methods=['POST'])
def insight_gemini():
    data = request.get_json()
    
    distribusi  = data.get('distribusi', {})
    top_positif = data.get('top_positif', [])
    top_negatif = data.get('top_negatif', [])
    total       = data.get('total', 0)
    # Frontend akan kirim ini — tambahkan di payload
    sampel_negatif = data.get('sampel_negatif', [])
    sampel_positif = data.get('sampel_positif', [])

    cache_key = str(hash(json.dumps({
        'distribusi': distribusi,
        'top_positif': top_positif[:5],
        'top_negatif': top_negatif[:5],
        'neg': sampel_negatif[:3]
    }, sort_keys=True)))
    
    if _insight_cache['key'] == cache_key and _insight_cache['teks']:
        print("[Gemini] Pakai cache")
        return jsonify({'insight': _insight_cache['teks']})

    kata_positif = ', '.join([w['word'] for w in top_positif[:8]])
    kata_negatif = ', '.join([w['word'] for w in top_negatif[:8]])
    
    # Format sampel teks - bersihkan karakter problematik
    def sanitasi(teks):
        return teks.replace('"', "'").replace('\\', '').replace('\n', ' ').replace('\r', '').strip()

    contoh_neg = '\n'.join([
        f'- "{sanitasi(t[:150])}"'
        for t in sampel_negatif[:8]
        if t.strip()    
    ])

    contoh_pos = '\n'.join([
        f'- "{sanitasi(t[:150])}"'
        for t in sampel_positif[:4]
        if t.strip()
    ])

    prompt = f"""Kamu adalah konsultan bisnis e-commerce yang berpengalaman.
    Hasil analisis sentimen dari {total} ulasan pelanggan:

    - Positif: {distribusi.get('Positif', {}).get('count', 0)} ({distribusi.get('Positif', {}).get('pct', 0)}%)
    - Netral: {distribusi.get('Netral', {}).get('count', 0)} ({distribusi.get('Netral', {}).get('pct', 0)}%)
    - Negatif: {distribusi.get('Negatif', {}).get('count', 0)} ({distribusi.get('Negatif', {}).get('pct', 0)}%)

    Kata yang sering muncul di ulasan POSITIF: {kata_positif}
    Kata yang sering muncul di ulasan NEGATIF: {kata_negatif}

    Berikan rekomendasi bisnis spesifik untuk penjual dalam bahasa Indonesia.
    Fokus: kualitas produk, pengemasan, pengiriman, layanan pelanggan.
    Format: paragraf mengalir, maksimal 3 paragraf."""

    GEMINI_KEY = "YOUR_GEMINI_API_KEY_HERE"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent?key={GEMINI_KEY}"

    # debug prompt Gemini
    print(f"[Gemini] Prompt length: {len(prompt)} chars")
    print(f"[Gemini] Contoh neg: {contoh_neg[:200]}")
    print(f"[Gemini] Contoh pos: {contoh_pos[:200]}")
    
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}]
    }).encode('utf-8')

    for percobaan in range(3):
        try:
            req = urllib.request.Request(
                url, data=body,
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode('utf-8'))

            teks = result['candidates'][0]['content']['parts'][0]['text']
            _insight_cache['teks'] = teks
            _insight_cache['key']  = cache_key
            return jsonify({'insight': teks})

        except urllib.error.HTTPError as e:
            if e.code == 429:
                tunggu = (percobaan + 1) * 25
                print(f"[Gemini] 429, tunggu {tunggu}s (percobaan {percobaan+1}/3)")
                time.sleep(tunggu)
                continue
            # error selain 429 (400, 403, dst)
            try:
                err_body = e.read().decode('utf-8')
                print(f"[Gemini] HTTP Error {e.code}: {err_body}")
            except:
                print(f"[Gemini] HTTP Error {e.code}: {e.reason}")
            return jsonify({'error': f'HTTP Error {e.code}: {e.reason}'}), 500
            print(f"[Gemini] HTTP Error {e.code}: {e.reason}")
            return jsonify({'error': f'HTTP Error {e.code}: {e.reason}'}), 500
        except Exception as e:
            print(f"[Gemini] Error: {e}")
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Gemini sedang sibuk. Tunggu 1 menit lalu coba lagi.'}), 429
    
    
if __name__ == '__main__':
    app.run(debug=True, port=5000)
    
    
#     Loop satu-satu (sebelumnya):
# setiap predict_one() dipanggil 67906 kali, dan tiap kali itu Python harus: buat sparse matrix baru → panggil fungsi predict → panggil fungsi proba → return. Overhead pemanggilan fungsi Python itu mahal kalau diulang puluhan ribu kali.
# Batch (sekarang):
# tfidf.transform(cleaned_texts) memproses semua 67906 teks sekaligus menjadi satu sparse matrix besar, lalu model.predict(vecs) dan model.predict_proba(vecs) beroperasi pada matrix itu dalam satu operasi numpy/scipy.
# Intinya, TF-IDF dan Naive Bayes di balik layar adalah operasi matrix multiplication — dan numpy/scipy sangat dioptimasi untuk perkalian matrix besar sekaligus menggunakan BLAS/LAPACK yang berjalan di level C, bukan Python. Jadi daripada 67906 perkalian matrix kecil yang masing-masing punya overhead, kamu melakukan 1 perkalian matrix besar yang jauh lebih efisien secara CPU cache dan memory access.
# Analogi sederhananya: seperti perbedaan antara bolak-balik ke dapur 67906 kali untuk ambil 1 piring vs sekali jalan bawa semua 67906 piring sekaligus pakai troli.