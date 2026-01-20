import json
import joblib
import sqlite3
# import smtplib                         # EMAIL DISABLED
# from email.mime.text import MIMEText  # EMAIL DISABLED
# from email.mime.multipart import MIMEMultipart  # EMAIL DISABLED
from datetime import datetime, timedelta, timezone
from flask import Flask, request, jsonify, render_template, g
from pathlib import Path
from dotenv import dotenv_values
import numpy as np

IST = timezone(timedelta(hours=5, minutes=30))

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

config = dotenv_values(ENV_PATH)

# EMAIL_SENDER = config.get("EMAIL_SENDER")      # EMAIL DISABLED
# EMAIL_PASSWORD = config.get("EMAIL_PASSWORD")  # EMAIL DISABLED
# ALERT_EMAIL = config.get("ALERT_EMAIL")        # EMAIL DISABLED

# print("EMAIL CONFIG CHECK →", EMAIL_SENDER, ALERT_EMAIL)  # EMAIL DISABLED

MODEL_PATH = "final_isolation_forest_model.pkl"
SCALER_PATH = "scaler.pkl"
DB_PATH = "predictions_history.db"

app = Flask(__name__, template_folder="templates")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# def send_email_alert(subject, message):   # EMAIL DISABLED
#     if not EMAIL_SENDER or not EMAIL_PASSWORD or not ALERT_EMAIL:
#         print("Email config missing")
#         return
#
#     try:
#         msg = MIMEMultipart()
#         msg["From"] = EMAIL_SENDER
#         msg["To"] = ALERT_EMAIL
#         msg["Subject"] = subject
#         msg.attach(MIMEText(message, "plain"))
#
#         server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
#         server.login(EMAIL_SENDER, EMAIL_PASSWORD)
#         server.sendmail(EMAIL_SENDER, ALERT_EMAIL, msg.as_string())
#         server.quit()
#
#         print(f"Email sent ({subject})")
#
#     except Exception as e:
#         print("Email sending failed:", e)

def get_db():
    if "_db" not in g:
        g._db = sqlite3.connect(DB_PATH)
        g._db.row_factory = sqlite3.Row
    return g._db

def init_db():
    db = get_db()
    db.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            state TEXT,
            district TEXT,
            latitude REAL,
            longitude REAL,
            features_json TEXT,
            raw_prediction INTEGER,
            decision_score REAL,
            label TEXT
        )
    """)
    db.commit()

@app.teardown_appcontext
def close_db(error):
    db = g.pop("_db", None)
    if db:
        db.close()

features_order = [
    "Recharge from rainfall During Monsoon Season",
    "Recharge from other sources During Monsoon Season",
    "Recharge from rainfall During Non Monsoon Season",
    "Recharge from other sources During Non Monsoon Season",
    "Total Natural Discharges",
    "Current Annual Ground Water Extraction For Irrigation",
    "Current Annual Ground Water Extraction For Domestic & Industrial Use",
    "Net Ground Water Availability for future use",
    "Stage of Ground Water Extraction (%)"
]

critical_thresholds = {
    "Recharge from rainfall During Monsoon Season": 40075,
    "Recharge from other sources During Monsoon Season": 9366,
    "Recharge from rainfall During Non Monsoon Season": 4850,
    "Recharge from other sources During Non Monsoon Season": 12124,
    "Total Natural Discharges": 5597,
    "Current Annual Ground Water Extraction For Irrigation": 35530,
    "Current Annual Ground Water Extraction For Domestic & Industrial Use": 4215,
    "Net Ground Water Availability for future use": 26498,
    "Stage of Ground Water Extraction (%)": 60.7
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    X = np.array([float(data[f]) for f in features_order]).reshape(1, -1)
    X_scaled = scaler.transform(X)

    raw_pred = int(model.predict(X_scaled)[0])
    score = float(model.decision_function(X_scaled)[0])

    threshold_trigger = any(float(data[f]) > critical_thresholds[f] for f in critical_thresholds)
    final_label = "CRITICAL" if raw_pred == -1 or threshold_trigger else "SAFE"

    subject = f"{'🚨' if final_label == 'CRITICAL' else '🟢'} Groundwater Status: {final_label}"

    message = f"""
Groundwater Status Report

Status: {final_label}
Decision Score: {score}

Location:
Latitude: {data.get('latitude')}
Longitude: {data.get('longitude')}

Timestamp:
{datetime.now(IST).isoformat()}
""".strip()

    # send_email_alert(subject, message)   # EMAIL DISABLED

    db = get_db()
    db.execute("""
        INSERT INTO predictions
        (timestamp, state, district, latitude, longitude, features_json,
         raw_prediction, decision_score, label)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now(IST).isoformat(),
        None,
        None,
        float(data["latitude"]) if data.get("latitude") else None,
        float(data["longitude"]) if data.get("longitude") else None,
        json.dumps({f: data[f] for f in features_order}),
        raw_pred,
        score,
        final_label
    ))
    db.commit()

    return jsonify({
        "success": True,
        "label": final_label,
        "raw_prediction": raw_pred,
        "decision_score": score
    })

@app.route("/history")
def history():
    db = get_db()
    rows = db.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 100").fetchall()

    history_list = []
    for r in rows:
        history_list.append({
            "id": r["id"],
            "timestamp": r["timestamp"],
            "latitude": r["latitude"],
            "longitude": r["longitude"],
            "features": json.loads(r["features_json"]),
            "raw_prediction": r["raw_prediction"],
            "decision_score": r["decision_score"],
            "label": r["label"]
        })

    return jsonify({"success": True, "history": history_list})

if __name__ == "__main__":
    with app.app_context():
        init_db()
    app.run(host="0.0.0.0", port=7860)
