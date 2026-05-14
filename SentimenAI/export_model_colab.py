# ============================================================
# Jalankan cell ini di Google Colab SETELAH training selesai
# Tujuan: export model ComplementNB supaya bisa dipakai di web
# ============================================================

# Pastikan model sudah di-training di cell sebelumnya
# (tfidf dan model sudah ada di memory)

import joblib
from google.colab import files

# Simpan TF-IDF vectorizer
joblib.dump(tfidf, 'model_tfidf.joblib')

# Simpan model ComplementNB
joblib.dump(model, 'sentiment_model.joblib')

print("Model berhasil disimpan!")
print(f"  - model_tfidf.joblib    (TF-IDF vectorizer)")
print(f"  - sentiment_model.joblib (ComplementNB classifier)")

# Download keduanya
files.download('model_tfidf.joblib')
files.download('sentiment_model.joblib')

print("\nFile sudah ter-download. Pindahkan ke folder project website kamu.")