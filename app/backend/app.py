# -*- coding: utf-8 -*-
"""
Backend - kiem tra du lieu theo schema.json (lay tu AI Service luc khoi dong),
goi AI Service, luu lich su du doan, phuc vu Frontend tinh.

Luu y: AI Service tra ve KET QUA CUA CA 4 MODEL trong mang "results" (kem "best_model").
Backend khong can biet chi tiet co bao nhieu model - chi kiem tra dau vao roi
chuyen tiep (forward) nguyen ket qua tu AI Service cho Frontend hien thi.

Endpoints: POST /api/predict, GET /api/history, GET /health
"""
import os
import time
import json
import uuid
import logging
from collections import deque

import requests
from flask import Flask, request, jsonify, send_from_directory

AI_SERVICE_URL = os.environ.get("AI_SERVICE_URL", "http://ai-service:8001")
PORT = int(os.environ.get("SERVER_PORT", 8000))
START_TIME = time.time()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s backend req=%(request_id)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backend")


def log(msg, request_id="-"):
    logger.info(msg, extra={"request_id": request_id})


# Luu lich su du doan trong bo nho (toi da 500 ban ghi gan nhat).
# Ghi chu: co the thay bang MongoDB (bien moi truong MONGODB_URI) khi trien khai
# that su can luu lau dai / chia se giua nhieu instance backend.
HISTORY = deque(maxlen=500)

app = Flask(__name__, static_folder="../frontend", static_url_path="")

SCHEMA_CACHE = {"schema": None, "fetched_at": 0}


def get_schema():
    """Lay schema.json tu AI Service (cache 60s) de validate dung 1 nguon su that."""
    if SCHEMA_CACHE["schema"] is None or time.time() - SCHEMA_CACHE["fetched_at"] > 60:
        resp = requests.get(f"{AI_SERVICE_URL}/options", timeout=5)
        resp.raise_for_status()
        SCHEMA_CACHE["schema"] = resp.json()
        SCHEMA_CACHE["fetched_at"] = time.time()
    return SCHEMA_CACHE["schema"]


def validate_features(features, schema):
    errors = []
    for f in schema["features"]:
        name = f["name"]
        if name not in features:
            errors.append(f"Thiếu trường '{name}'")
            continue
        val = features[name]
        if f["dtype"] in ("int", "float"):
            try:
                num = float(val)
            except (TypeError, ValueError):
                errors.append(f"'{name}' phải là số")
                continue
            if "min" in f and num < f["min"]:
                errors.append(f"'{name}' phải >= {f['min']}")
            if "max" in f and num > f["max"]:
                errors.append(f"'{name}' phải <= {f['max']}")
        elif f["dtype"] == "category" and "allowed_values" in f:
            if val not in f["allowed_values"]:
                errors.append(f"'{name}' phải thuộc {f['allowed_values']}")
    return errors


@app.route("/health", methods=["GET"])
def health():
    ai_ok = False
    try:
        r = requests.get(f"{AI_SERVICE_URL}/health", timeout=3)
        ai_ok = r.status_code == 200
    except requests.RequestException:
        ai_ok = False
    return jsonify({
        "status": "ok",
        "port": PORT,
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "ai_service_reachable": ai_ok,
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:8])
    t0 = time.time()
    body = request.get_json(silent=True) or {}
    features = body.get("features", body)

    try:
        schema = get_schema()
    except requests.RequestException as exc:
        log("khong ket noi duoc AI Service de lay schema: %s" % exc, request_id)
        return jsonify({"error": "ai_service_unreachable", "detail": str(exc),
                         "request_id": request_id}), 502

    errors = validate_features(features, schema)
    if errors:
        log("validate that bai: %s" % errors, request_id)
        return jsonify({"error": "invalid_input", "detail": "; ".join(errors),
                         "request_id": request_id}), 400
    log("validate OK -> goi ai-service", request_id)

    try:
        r = requests.post(f"{AI_SERVICE_URL}/predict", json={"features": features},
                           headers={"X-Request-ID": request_id}, timeout=10)
    except requests.RequestException as exc:
        log("loi goi ai-service: %s" % exc, request_id)
        return jsonify({"error": "ai_service_error", "detail": str(exc),
                         "request_id": request_id}), 502

    if r.status_code != 200:
        return (r.text, r.status_code, {"Content-Type": "application/json"})

    result = r.json()  # {"best_model": {...}, "results": [4 model], ...}
    n_models = len(result.get("results", []))
    HISTORY.appendleft({"request_id": request_id, "features": features, "result": result,
                         "timestamp": time.time()})
    total_ms = round((time.time() - t0) * 1000, 1)
    log("200 OK (%d model) total %sms, saved history" % (n_models, total_ms), request_id)
    return jsonify(result)


@app.route("/api/history", methods=["GET"])
def api_history():
    limit = int(request.args.get("limit", 20))
    return jsonify(list(HISTORY)[:limit])


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
