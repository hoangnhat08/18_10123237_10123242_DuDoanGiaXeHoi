# -*- coding: utf-8 -*-
"""
AI Service - nap CA 4 MODEL NGAY KHI container khoi dong (khong doi request dau tien).
Moi lan /predict, service chay ca 4 model va tra ve KET QUA CUA CA 4, kem model
duoc danh dau la "tot nhat" (best_model_key, lay tu metadata.json/bang so sanh o buoc 4).

Endpoints: POST /predict, GET /health, GET /model-info, GET /options
"""
import os
import json
import time
import logging
import uuid

import joblib
import pandas as pd
from flask import Flask, request, jsonify

HERE = os.path.dirname(__file__)
MODELS_DIR = os.path.join(HERE, "..", "models")
SCHEMA_PATH = os.path.join(MODELS_DIR, "schema.json")
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")

PORT = int(os.environ.get("AI_SERVICE_PORT", 8001))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s ai-service req=%(request_id)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ai-service")


def log(msg, request_id="-"):
    logger.info(msg, extra={"request_id": request_id})


# ============================================================
# Nap schema + metadata + CA 4 MODEL NGAY LUC IMPORT MODULE
# (chay ngay khi container khoi dong, truoc khi nhan request dau tien)
# ============================================================
START_TIME = time.time()

with open(SCHEMA_PATH, encoding="utf-8") as f:
    SCHEMA = json.load(f)
with open(METADATA_PATH, encoding="utf-8") as f:
    METADATA = json.load(f)

FEATURE_NAMES = [f["name"] for f in SCHEMA["features"]]
BEST_MODEL_KEY = METADATA.get("best_model_key") or METADATA.get("model_key")

# METADATA["models"]: danh sach 4 model, moi phan tu co "key", "name", "file", "is_best", metric...
# Day la NGUON SU THAT DUY NHAT cho biet co bao nhieu model va file nao ung voi model nao;
# them/bot model chi can sua metadata.json (do train.py sinh ra), khong can sua code o day.
MODEL_ENTRIES = METADATA.get("models")
if not MODEL_ENTRIES:
    # tuong thich nguoc: metadata cu chi co 1 model duy nhat (model.joblib)
    MODEL_ENTRIES = [{
        "key": METADATA.get("model_key", "model"),
        "name": METADATA.get("model_name", "Model"),
        "file": "model.joblib",
        "is_best": True,
        "metrics_test": METADATA.get("metrics_test", {}),
    }]

MODELS = {}
for entry in MODEL_ENTRIES:
    key = entry["key"]
    model_path = os.path.join(MODELS_DIR, entry["file"])
    log("Dang nap model '%s' tu %s ..." % (key, model_path))
    MODELS[key] = joblib.load(model_path)

log("Da nap xong %d model: %s (best=%s)" % (len(MODELS), list(MODELS.keys()), BEST_MODEL_KEY))

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "port": PORT,
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "models_loaded": len(MODELS),
        "model_keys": list(MODELS.keys()),
        "best_model_key": BEST_MODEL_KEY,
        "model_version": METADATA.get("model_version"),
    })


@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify(METADATA)


@app.route("/options", methods=["GET"])
def options():
    return jsonify(SCHEMA)


@app.route("/predict", methods=["POST"])
def predict():
    request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:8])
    t0 = time.time()
    body = request.get_json(silent=True) or {}
    features = body.get("features", body)  # chap nhan ca {"features": {...}} lan {...} truc tiep

    missing = [c for c in FEATURE_NAMES if c not in features]
    if missing:
        log("input thieu truong %s" % missing, request_id)
        return jsonify({"error": "invalid_input", "detail": f"Thieu truong: {missing}",
                         "request_id": request_id}), 400

    try:
        row = {c: features[c] for c in FEATURE_NAMES}
        X = pd.DataFrame([row])
    except Exception as exc:  # noqa: BLE001
        log("loi chuan bi du lieu: %s" % exc, request_id)
        return jsonify({"error": "invalid_input", "detail": str(exc),
                         "request_id": request_id}), 400

    # Chay LAN LUOT CA 4 MODEL tren cung 1 dong du lieu, moi model do rieng thoi gian du doan.
    results = []
    entries_by_key = {e["key"]: e for e in MODEL_ENTRIES}
    for key, model in MODELS.items():
        entry = entries_by_key.get(key, {})
        try:
            m0 = time.time()
            pred_million_vnd = float(model.predict(X)[0])
            predict_ms = round((time.time() - m0) * 1000, 2)
        except Exception as exc:  # noqa: BLE001
            log("model '%s' loi du doan: %s" % (key, exc), request_id)
            return jsonify({"error": "prediction_failed", "detail": f"[{key}] {exc}",
                             "request_id": request_id}), 400

        results.append({
            "key": key,
            "model_name": entry.get("name", key),
            "is_best": bool(entry.get("is_best", key == BEST_MODEL_KEY)),
            "prediction_million_vnd": round(pred_million_vnd, 1),
            "prediction_text": f"{pred_million_vnd:,.0f} triệu VNĐ",
            "predict_time_ms": predict_ms,
            "metrics_test": entry.get("metrics_test"),
        })

    # Sap xep de model tot nhat (is_best) len dau, phan con lai giu nguyen thu tu nap.
    results.sort(key=lambda r: (not r["is_best"],))

    best = next((r for r in results if r["is_best"]), results[0])
    elapsed_ms = round((time.time() - t0) * 1000, 1)
    log("predict xong %d model, best=%s (%.1f trieu) total=%sms" %
        (len(results), best["key"], best["prediction_million_vnd"], elapsed_ms), request_id)

    return jsonify({
        "best_model": best,
        "results": results,
        "model_version": METADATA.get("model_version"),
        "request_id": request_id,
        # cac truong duoi day giu de tuong thich nguoc voi code cu chi doc 1 ket qua
        "prediction_million_vnd": best["prediction_million_vnd"],
        "prediction_text": best["prediction_text"],
        "model_name": best["model_name"],
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
