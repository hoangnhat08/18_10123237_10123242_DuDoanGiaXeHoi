# -*- coding: utf-8 -*-
"""
data_utils.py
Ham dung chung de doc va lam sach du lieu xe cu tu bonbanh.com (Kaggle: danh911/gi-xe).
Dung lai o ca notebook EDA, notebook/train.py huan luyen, va AI Service khi can kiem tra dau vao.
"""
import re
import numpy as np
import pandas as pd

RAW_COLUMNS = [
    "car_name", "year", "price", "assemble_place", "series",
    "driven kms", "num_of_door", "num_of_seat", "engine_type",
    "transmission", "url",
]

FEATURE_COLUMNS = [
    "brand", "model", "series", "year", "driven_kms",
    "assemble_place", "engine_type", "transmission",
    "num_of_door", "num_of_seat",
]
TARGET_COLUMN = "price"
NUMERIC_FEATURES = ["year", "driven_kms", "num_of_door", "num_of_seat"]
CATEGORICAL_FEATURES = ["brand", "model", "series", "assemble_place", "engine_type", "transmission"]


def parse_price_to_million_vnd(text):
    """'2 Ty 700 Trieu' -> 2700 (trieu VND). Tra ve NaN neu khong parse duoc."""
    if pd.isna(text):
        return np.nan
    s = str(text).strip()
    if s == "" or s.lower() in {"nan", "thoa thuan", "giá thỏa thuận"}:
        return np.nan
    ty_match = re.search(r"([\d.,]+)\s*T[yỷ]", s, flags=re.IGNORECASE)
    trieu_match = re.search(r"([\d.,]+)\s*Tr[i1]ệu|Trieu|Triệu", s, flags=re.IGNORECASE)
    trieu_num_match = re.search(r"([\d.,]+)\s*Tri[eệ]u", s, flags=re.IGNORECASE)

    ty_val = 0.0
    trieu_val = 0.0
    if ty_match:
        ty_val = float(ty_match.group(1).replace(",", "."))
    if trieu_num_match:
        trieu_val = float(trieu_num_match.group(1).replace(",", "."))

    if ty_match or trieu_num_match:
        return ty_val * 1000 + trieu_val

    # fallback: chuoi chi co so (vd '450' hieu la 450 trieu), hoac so co dau cham/phay
    only_num = re.sub(r"[^\d.,]", "", s)
    if only_num == "":
        return np.nan
    try:
        return float(only_num.replace(",", ""))
    except ValueError:
        return np.nan


def split_brand_model(car_name):
    """Tach 'Mazda 3 1.5L Luxury' -> brand='Mazda', model='3 1.5L Luxury'."""
    if pd.isna(car_name):
        return np.nan, np.nan
    parts = str(car_name).strip().split(maxsplit=1)
    brand = parts[0] if len(parts) >= 1 else np.nan
    model = parts[1] if len(parts) >= 2 else np.nan
    return brand, model


def load_raw(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df


def clean_dataframe(df, min_price=30, max_price=60000, max_kms=1_000_000,
                     min_year=1980, max_year=2027):
    """
    Lam sach theo dung mo ta trong README:
    - parse gia ve trieu VND
    - bo dong thieu, bo trung theo url
    - bo dong nam khong hop le (loi crawl / lech cot)
    - tach brand/model tu car_name
    - loc ngoai lai: km, so cua/ghe, gia
    Tra ve DataFrame da lam sach voi FEATURE_COLUMNS + TARGET_COLUMN.
    """
    d = df.copy()
    d.columns = [c.strip() for c in d.columns]
    d = d.rename(columns={"driven kms": "driven_kms"})

    # 1) parse gia
    d["price"] = d["price"].apply(parse_price_to_million_vnd)

    # 2) ep kieu so, loai dong loi crawl (vd nam khong phai so)
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    d["driven_kms"] = pd.to_numeric(d["driven_kms"], errors="coerce")
    d["num_of_door"] = pd.to_numeric(d["num_of_door"], errors="coerce")
    d["num_of_seat"] = pd.to_numeric(d["num_of_seat"], errors="coerce")

    # 3) tach brand / model
    brand_model = d["car_name"].apply(split_brand_model)
    d["brand"] = brand_model.apply(lambda t: t[0])
    d["model"] = brand_model.apply(lambda t: t[1])

    # 4) bo dong thieu truong quan trong
    required = ["price", "year", "driven_kms", "brand", "series",
                "assemble_place", "engine_type", "transmission",
                "num_of_door", "num_of_seat", "url"]
    d = d.dropna(subset=required)

    # 5) bo trung theo url
    d = d.drop_duplicates(subset=["url"])

    # 6) loc ngoai lai
    d = d[(d["year"] >= min_year) & (d["year"] <= max_year)]
    d = d[(d["driven_kms"] >= 0) & (d["driven_kms"] <= max_kms)]
    d = d[(d["num_of_door"] >= 2) & (d["num_of_door"] <= 6)]
    d = d[(d["num_of_seat"] >= 2) & (d["num_of_seat"] <= 16)]
    d = d[(d["price"] >= min_price) & (d["price"] <= max_price)]

    d["year"] = d["year"].astype(int)
    d["num_of_door"] = d["num_of_door"].astype(int)
    d["num_of_seat"] = d["num_of_seat"].astype(int)

    out = d[FEATURE_COLUMNS + [TARGET_COLUMN]].reset_index(drop=True)
    return out


def load_clean(csv_path, **kwargs):
    return clean_dataframe(load_raw(csv_path), **kwargs)
