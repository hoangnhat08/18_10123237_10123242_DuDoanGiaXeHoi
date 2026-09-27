# -*- coding: utf-8 -*-
"""
train.py - Huan luyen 4 model hoi quy du doan gia xe cu.
Chay: python train.py
Dau ra (moi model la MOT FILE RIENG, AI Service se nap ca 4 file nay):
  - ai-models/models/model_linear_regression.joblib
  - ai-models/models/model_decision_tree.joblib
  - ai-models/models/model_random_forest.joblib
  - ai-models/models/model_svr.joblib
  - ai-models/models/model.joblib          (alias = ban sao cua model tot nhat,
                                             giu de tuong thich nguoc / xem nhanh)
  - ai-models/models/schema.json
  - ai-models/models/metadata.json          (liet ke ca 4 model: key, ten, duong dan
                                             file, metric, model tot nhat la ai)
  - docs/figures/08_model_comparison.png, 09_residuals.png
  - training metrics in ra man hinh + luu docs/metrics.csv
"""
import os
import sys
import json
import time
import platform
import numpy as np
import pandas as pd
import sklearn
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import (load_raw, clean_dataframe, FEATURE_COLUMNS, TARGET_COLUMN,
                         NUMERIC_FEATURES, CATEGORICAL_FEATURES)

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "..", "data", "data.csv")
MODELS_DIR = os.path.join(HERE, "..", "models")
FIG_DIR = os.path.join(HERE, "..", "..", "docs", "figures")
DOCS_DIR = os.path.join(HERE, "..", "..", "docs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
RANDOM_STATE = 42


def build_preprocessor():
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def build_models():
    """4 model, gom 1 baseline. TransformedTargetRegressor: log1p(price) khi fit, expm1 khi predict."""
    models = {
        "linear_regression": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(max_depth=12, min_samples_leaf=5, random_state=RANDOM_STATE),
        "random_forest": RandomForestRegressor(n_estimators=300, max_depth=16, min_samples_leaf=2,
                                                n_jobs=-1, random_state=RANDOM_STATE),
        "svr": SVR(kernel="rbf", C=10, epsilon=0.05),
    }
    return models


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def main():
    raw = load_raw(DATA_PATH)
    df = clean_dataframe(raw)
    print("Du lieu sach:", df.shape)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)
    print("Train:", X_train.shape, "Test:", X_test.shape)

    preprocessor = build_preprocessor()
    models = build_models()

    results = []
    fitted = {}
    labels = {
        "linear_regression": "Linear Regression (baseline)",
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
        "svr": "SVR",
    }

    for key, base_model in models.items():
        pipe = Pipeline(steps=[
            ("preprocess", build_preprocessor()),
            ("model", base_model),
        ])
        ttr = TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)

        t0 = time.time()
        ttr.fit(X_train, y_train)
        train_time = time.time() - t0

        t0 = time.time()
        y_pred_test = ttr.predict(X_test)
        predict_time = (time.time() - t0) / max(len(X_test), 1)

        y_pred_train = ttr.predict(X_train)

        mae_test, rmse_test, r2_test = evaluate(y_test, y_pred_test)
        mae_train, rmse_train, r2_train = evaluate(y_train, y_pred_train)

        # Luu MOI model thanh MOT FILE RIENG NGAY TAI DAY (khong chi luu model tot nhat).
        # AI Service se nap ca 4 file nay khi container khoi dong.
        model_filename = f"model_{key}.joblib"
        model_file_path = os.path.join(MODELS_DIR, model_filename)
        joblib.dump(ttr, model_file_path, compress=3)
        size_mb = os.path.getsize(model_file_path) / (1024 * 1024)

        fitted[key] = ttr
        results.append({
            "model": labels[key],
            "key": key,
            "file": model_filename,
            "MAE_test": round(mae_test, 2),
            "RMSE_test": round(rmse_test, 2),
            "R2_test": round(r2_test, 4),
            "MAE_train": round(mae_train, 2),
            "RMSE_train": round(rmse_train, 2),
            "R2_train": round(r2_train, 4),
            "train_time_s": round(train_time, 2),
            "predict_time_ms_per_row": round(predict_time * 1000, 4),
            "model_size_MB": round(size_mb, 2),
        })
        print(f"[{labels[key]}] MAE={mae_test:.1f} RMSE={rmse_test:.1f} R2={r2_test:.4f} "
              f"(train R2={r2_train:.4f}) time={train_time:.1f}s size={size_mb:.1f}MB")

    metrics_df = pd.DataFrame(results).sort_values("R2_test", ascending=False)
    metrics_df.to_csv(os.path.join(DOCS_DIR, "metrics.csv"), index=False)
    print("\n=== Bang so sanh (sap xep theo R2 test) ===")
    print(metrics_df.to_string(index=False))

    best_key = metrics_df.iloc[0]["key"]
    best_model = fitted[best_key]
    print(f"\n>>> Model tot nhat: {labels[best_key]} ({best_key})")

    # ---- Hinh 8: so sanh model ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    order = metrics_df["model"]
    axes[0].bar(order, metrics_df["MAE_test"], color="#3a7ca5")
    axes[0].set_title("MAE tren tap test (trieu VND, thap hon = tot hon)")
    plt.setp(axes[0].get_xticklabels(), rotation=20, ha="right")
    axes[1].bar(order, metrics_df["R2_test"], color="#2f6690")
    axes[1].set_title("R2 tren tap test (cao hon = tot hon)")
    axes[1].set_ylim(0, 1)
    plt.setp(axes[1].get_xticklabels(), rotation=20, ha="right")
    fig.suptitle("Hinh 8: So sanh 4 model hoi quy")
    fig.savefig(os.path.join(FIG_DIR, "08_model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Hinh 9: bieu do phan du cua model tot nhat ----
    y_pred_best = best_model.predict(X_test)
    residuals = y_test - y_pred_best
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(y_pred_best, residuals, s=8, alpha=0.3, color="#2f6690")
    axes[0].axhline(0, color="red", linewidth=1)
    axes[0].set_xlabel("Gia du doan (trieu VND)")
    axes[0].set_ylabel("Phan du (thuc te - du doan)")
    axes[0].set_title(f"Phan du - {labels[best_key]}")
    axes[1].hist(residuals, bins=60, color="#3a7ca5")
    axes[1].set_title("Phan bo phan du")
    fig.suptitle("Hinh 9: Phan tich loi cua model tot nhat")
    fig.savefig(os.path.join(FIG_DIR, "09_residuals.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Xuat schema + metadata. 4 file model_<key>.joblib da duoc luu o vong lap tren. ----
    # model.joblib duoc giu lai nhu MOT BAN SAO/ALIAS cua model tot nhat, de ai muon
    # xem/nap nhanh "model chinh" van tim thay theo duong dan cu; AI Service khong
    # dung file nay de tra ket qua ma nap rieng ca 4 file model_<key>.joblib.
    best_alias_path = os.path.join(MODELS_DIR, "model.joblib")
    joblib.dump(best_model, best_alias_path, compress=3)
    print("Da luu 4 model rieng:", [r["file"] for r in results])
    print("Da luu ban sao model tot nhat (alias):", best_alias_path,
          f"({os.path.getsize(best_alias_path)/1024/1024:.2f} MB)")

    schema = {
        "target": TARGET_COLUMN,
        "target_unit": "million_vnd",
        "features": [
            {"name": "brand", "dtype": "category", "example": "Toyota"},
            {"name": "model", "dtype": "category", "example": "Camry 2.5Q"},
            {"name": "series", "dtype": "category", "allowed_values": sorted(df["series"].unique().tolist())},
            {"name": "year", "dtype": "int", "min": int(df["year"].min()), "max": int(df["year"].max())},
            {"name": "driven_kms", "dtype": "float", "min": float(df["driven_kms"].min()), "max": float(df["driven_kms"].max())},
            {"name": "assemble_place", "dtype": "category", "allowed_values": sorted(df["assemble_place"].unique().tolist())},
            {"name": "engine_type", "dtype": "category", "allowed_values": sorted(df["engine_type"].unique().tolist())},
            {"name": "transmission", "dtype": "category", "allowed_values": sorted(df["transmission"].unique().tolist())},
            {"name": "num_of_door", "dtype": "int", "min": int(df["num_of_door"].min()), "max": int(df["num_of_door"].max())},
            {"name": "num_of_seat", "dtype": "int", "min": int(df["num_of_seat"].min()), "max": int(df["num_of_seat"].max())},
        ],
    }
    with open(os.path.join(MODELS_DIR, "schema.json"), "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    best_row = metrics_df.iloc[0].to_dict()
    # "models": danh sach ca 4 model - AI Service doc chinh phan nay de biet nap file nao,
    # ten hien thi tren giao dien la gi, va model nao la "best" (danh dau mac dinh/goi y).
    models_list = []
    for r in results:
        models_list.append({
            "key": r["key"],
            "name": r["model"],
            "file": r["file"],
            "is_best": (r["key"] == best_key),
            "metrics_test": {"MAE": r["MAE_test"], "RMSE": r["RMSE_test"], "R2": r["R2_test"]},
            "metrics_train": {"MAE": r["MAE_train"], "RMSE": r["RMSE_train"], "R2": r["R2_train"]},
            "train_time_s": r["train_time_s"],
            "predict_time_ms_per_row": r["predict_time_ms_per_row"],
            "model_size_MB": r["model_size_MB"],
        })

    metadata = {
        "model_version": "2.0.0",
        "trained_at": pd.Timestamp.now().isoformat(),
        "best_model_key": best_key,
        "best_model_name": labels[best_key],
        # Giu 2 truong nay (model_name/model_key + metrics_test/train o cap tren cung)
        # de tuong thich nguoc voi ma cu tung doc metadata["model_name"] truc tiep.
        "model_name": labels[best_key],
        "model_key": best_key,
        "metrics_test": {"MAE": best_row["MAE_test"], "RMSE": best_row["RMSE_test"], "R2": best_row["R2_test"]},
        "metrics_train": {"MAE": best_row["MAE_train"], "RMSE": best_row["RMSE_train"], "R2": best_row["R2_train"]},
        "models": models_list,
        "all_models_compared": results,
        "target_unit": "million_vnd",
        "n_rows_train": int(len(X_train)),
        "n_rows_test": int(len(X_test)),
        "random_state": RANDOM_STATE,
        "library_versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
    }
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Da luu schema.json va metadata.json trong", MODELS_DIR)


if __name__ == "__main__":
    main()
