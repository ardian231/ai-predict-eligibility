import firebase_admin
from firebase_admin import credentials, db, firestore
import pandas as pd
import joblib
import re
import shap
import time
import threading

# ========== Setup ==========
FIREBASE_CREDENTIALS = "firebase-key.json"
DATABASE_URL = "https://reseller-form-a616f-default-rtdb.asia-southeast1.firebasedatabase.app/"
SLEEP_INTERVAL = 15  # detik

# ========== Init Firebase ==========
cred = credentials.Certificate(FIREBASE_CREDENTIALS)
firebase_admin.initialize_app(cred, {'databaseURL': DATABASE_URL})
fs = firestore.client()

# ========== Load Assets ==========
model = joblib.load("loan_risk_model.pkl")
le_job = joblib.load("label_encoder_job.pkl")
le_item = joblib.load("label_encoder_item.pkl")
explainer = shap.TreeExplainer(model)

# ========== Constants ==========
feature_explanations = {
    "income_amount": {
        "positive": "Pendapatan tinggi meningkatkan peluang persetujuan.",
        "negative": "Pendapatan rendah menurunkan kemungkinan disetujui."
    },
    "installment_amount": {
        "positive": "Cicilan tinggi menaikkan risiko penolakan.",
        "negative": "Cicilan rendah mendukung kemampuan membayar."
    },
    "job": {
        "positive": "Pekerjaan dianggap stabil.",
        "negative": "Pekerjaan dianggap berisiko."
    },
    "item": {
        "positive": "Barang yang dibiayai dianggap bernilai baik.",
        "negative": "Barang yang dibiayai dianggap kurang bernilai."
    }
}

label_map = {0: "reject", 1: "approve", 2: "dipertimbangkan"}

fixed_explanations = {
    "approve": ["Pendapatan memenuhi syarat.", "Barang dibiayai dianggap layak."],
    "reject": ["Pendapatan kurang memenuhi syarat.", "Risiko kredit tinggi."],
    "dipertimbangkan": ["Data perlu ditinjau manual.", "Beberapa kriteria belum optimal."]
}

# ========== Normalisasi Mapping ==========
item_normalization_map = {
    "Motor baru": "motor baru",
    "Motor Bekas": "motor bekas",
    "Mobil Baru": "mobil Baru",
    "Mobil Bekas": "mobil Bekas",
    "amanah (adira multi dana syariah)": "AMANAH (Adira Multi Dana Syariah)"
}

job_normalization_map = {
    "pegawai negeri": "Pegawai Negeri",
    "karyawan swasta": "Karyawan Swasta",
    "wirausaha": "Wirausaha",
    "mahasiswa": "Mahasiswa",
    "dokter": "Dokter",
    "guru": "Guru",
    "lainnya": "Lainnya"
    # Tambahkan job lainnya di sini jika perlu
}

# ========== Utils ==========
def extract_amount(text):
    if not text:
        return 0
    text = text.lower()
    match = re.search(r'(\d+)\s*(juta|ribu)', text)
    if match:
        num, satuan = int(match.group(1)), match.group(2)
        return num * 1_000_000 if satuan == 'juta' else num * 1_000
    return 0

def normalize_item(text):
    if not isinstance(text, str):
        return text
    text = text.strip().lower()
    return item_normalization_map.get(text, text.title())

def normalize_job(text):
    if not isinstance(text, str):
        return text
    text = text.strip().lower()
    return job_normalization_map.get(text, text.capitalize())

def safe_label_encode(le, value, type_='general'):
    if not isinstance(value, str):
        return -1
    value = value.strip()
    if type_ == 'item':
        value = normalize_item(value)
    elif type_ == 'job':
        value = normalize_job(value)
    else:
        value = value.capitalize()
    try:
        return le.transform([value])[0]
    except ValueError:
        print(f"⚠️ Label {type_} tidak dikenali: {value}")
        return -1

def get_realtime_data():
    ref = db.reference('orders')
    data = ref.get()
    return pd.DataFrame.from_dict(data, orient='index') if data else pd.DataFrame()

def preprocess_data(df):
    df['income_amount'] = df['income'].apply(extract_amount)
    df['installment_amount'] = df['installment'].apply(extract_amount)
    df['job'] = df['job'].apply(lambda x: safe_label_encode(le_job, x, type_='job'))
    df['item'] = df['item'].apply(lambda x: safe_label_encode(le_item, x, type_='item'))
    return df[['job', 'item', 'income_amount', 'installment_amount']]

def generate_explanations(predictions, shap_values, X, mode):
    results = []
    for i, pred in enumerate(predictions):
        try:
            status = label_map[pred]
            if mode == "dynamic":
                shap_row = shap_values[pred][i]
                impacts = sorted(zip(X.columns, shap_row), key=lambda x: abs(x[1]), reverse=True)
                reasons = []
                for feature, impact in impacts:
                    if len(reasons) >= 2:
                        break
                    direction = "positive" if impact > 0 else "negative"
                    reason = feature_explanations.get(feature, {}).get(direction)
                    if reason and reason not in reasons:
                        reasons.append(reason)
                if not reasons:
                    reasons = fixed_explanations.get(status, ["Tidak ada alasan ditemukan."])
            else:
                reasons = fixed_explanations.get(status, ["Tidak ada alasan ditemukan."])

            results.append({'status': status, 'alasan': reasons})
        except Exception as e:
            print(f"❌ Error generate_explanations data ke-{i}: {e}")
            results.append({'status': "unknown", 'alasan': ["Gagal menganalisa data."]})
    return results

def save_predictions_to_firestore(results, ids):
    for doc_id, result in zip(ids, results):
        fs.collection('loan_predictions').document(doc_id).set(result)

# ========== Pipeline ==========
def run_pipeline(mode="dynamic"):
    print("🚀 Mulai cek data baru...")
    df = get_realtime_data()
    if df.empty:
        print("🟡 Tidak ada data baru.")
        return

    ids = df.index.tolist()
    try:
        X = preprocess_data(df)
        predictions = model.predict(X)
        shap_values = explainer.shap_values(X)
        results = generate_explanations(predictions, shap_values, X, mode)
        save_predictions_to_firestore(results, ids)
        print(f"✅ {len(results)} prediksi disimpan ke Firestore.")
    except Exception as e:
        print(f"❌ Error di run_pipeline: {e}")

# ========== Main Loop ==========
def start_loop(mode="dynamic"):
    last_ids = set()
    while True:
        try:
            df = get_realtime_data()
            if not df.empty:
                current_ids = set(df.index)
                new_ids = current_ids - last_ids
                if new_ids:
                    df_new = df.loc[list(new_ids)]
                    print(f"📥 {len(df_new)} data baru ditemukan.")
                    run_pipeline(mode)
                    last_ids = current_ids
                else:
                    print("🟡 Tidak ada data baru masuk.")
            else:
                print("🟡 Database kosong.")
        except Exception as e:
            print(f"❌ Error di loop utama: {e}")
        time.sleep(SLEEP_INTERVAL)

# ========== Entry Point ==========
if __name__ == '__main__':
    threading.Thread(target=start_loop, kwargs={'mode': 'fixed'}).start()
