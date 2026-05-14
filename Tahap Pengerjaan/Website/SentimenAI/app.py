from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
import re
from collections import Counter

app = Flask(__name__, static_folder='.')
CORS(app)

# ─── Load Model ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
tfidf = joblib.load(os.path.join(BASE_DIR, 'model_tfidf.joblib'))
model = joblib.load(os.path.join(BASE_DIR, 'sentiment_model.joblib'))  # ComplementNB

LABEL_MAP   = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
LABEL_EMOJI = {0: '😡', 1: '😐', 2: '😊'}

# ─── Helper ───────────────────────────────────────────────────────────────────
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
    """Ambil kata yang paling sering muncul dari list teks."""
    all_words = []
    for t in texts:
        words = clean_text(str(t)).split()
        # filter kata terlalu pendek
        all_words.extend([w for w in words if len(w) > 3])
    counter = Counter(all_words)
    return [{'word': w, 'count': c} for w, c in counter.most_common(n)]

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict/text', methods=['POST'])
def predict_text():
    data = request.get_json()
    text = data.get('review', '').strip()
    if not text:
        return jsonify({'error': 'Review kosong'}), 400
    return jsonify(predict_one(text))

@app.route('/predict/file', methods=['POST'])
def predict_file():
    if 'file' not in request.files:
        return jsonify({'error': 'File tidak ditemukan'}), 400

    f = request.files['file']
    fname = f.filename.lower()

    try:
        if fname.endswith('.csv'):
            df = pd.read_csv(f)
        else:
            df = pd.read_excel(f)
    except Exception as e:
        return jsonify({'error': f'Gagal membaca file: {str(e)}'}), 400

    # cari kolom teks
    text_col = None
    for candidate in ['review_cleaned', 'review', 'ulasan', 'text', 'comment']:
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col is None:
        text_col = df.columns[0]

    df = df.dropna(subset=[text_col])
    df[text_col] = df[text_col].astype(str)

    results   = [predict_one(t) for t in df[text_col].tolist()]
    labels    = [r['label'] for r in results]
    sentimens = [r['sentimen'] for r in results]

    # distribusi
    from collections import Counter
    dist = Counter(sentimens)
    total = len(results)
    distribusi = {s: {'count': dist.get(s, 0),
                      'pct'  : round(dist.get(s, 0)/total*100, 1)}
                  for s in ['Positif', 'Netral', 'Negatif']}

    # kata sering per sentimen
    pos_texts = [df[text_col].iloc[i] for i, l in enumerate(labels) if l == 2]
    neg_texts = [df[text_col].iloc[i] for i, l in enumerate(labels) if l == 0]

    return jsonify({
        'total'     : total,
        'distribusi': distribusi,
        'top_positif': top_words(pos_texts),
        'top_negatif': top_words(neg_texts),
        'samples'   : results[:5],   # preview 5 baris pertama
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)