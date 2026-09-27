# -*- coding: utf-8 -*-
"""
eda.py - Phan tich kham pha du lieu (EDA)
Sinh >= 5 hinh ve luu vao docs/figures/, dung cho notebook 01_eda va bao cao/slide.
Chay: python eda.py  (tu thu muc ai-models/src, hoac sua duong dan DATA_PATH/OUT_DIR)
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_raw, clean_dataframe, NUMERIC_FEATURES

sns.set_theme(style="whitegrid")
PALETTE = "crest"

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "data.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)


def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved", path)


def main():
    raw = load_raw(DATA_PATH)
    df = clean_dataframe(raw)
    print("Raw:", raw.shape, "| Clean:", df.shape)

    # ---- Hinh 1: Phan bo bien muc tieu (price) ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(df["price"], bins=60, ax=axes[0], color="#2f6690")
    axes[0].set_title("Phan bo gia xe (trieu VND)")
    axes[0].set_xlabel("Gia (trieu VND)")
    sns.histplot(np.log1p(df["price"]), bins=60, ax=axes[1], color="#3a7ca5")
    axes[1].set_title("Phan bo log1p(gia)")
    axes[1].set_xlabel("log1p(gia)")
    fig.suptitle("Hinh 1: Phan bo bien muc tieu 'price'")
    savefig(fig, "01_target_distribution.png")

    # ---- Hinh 2: Phan bo & ngoai lai cac bien so ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    sns.boxplot(y=df["driven_kms"], ax=axes[0], color="#81c3d7")
    axes[0].set_title("So km da di")
    sns.histplot(df["year"], bins=30, ax=axes[1], color="#3a7ca5")
    axes[1].set_title("Nam san xuat")
    sns.boxplot(y=df["price"], ax=axes[2], color="#2f6690")
    axes[2].set_title("Gia (trieu VND)")
    fig.suptitle("Hinh 2: Phan bo va ngoai lai cac bien so")
    savefig(fig, "02_numeric_distributions.png")

    # ---- Hinh 3: Ban do du lieu thieu (tren du lieu goc) ----
    fig, ax = plt.subplots(figsize=(10, 5))
    missing = raw.isna().mean().sort_values(ascending=False) * 100
    sns.barplot(x=missing.values, y=missing.index, ax=ax, color="#2f6690")
    ax.set_xlabel("% gia tri thieu")
    ax.set_title("Hinh 3: Ty le gia tri thieu theo cot (du lieu goc)")
    savefig(fig, "03_missing_data.png")

    # ---- Hinh 4: Heatmap tuong quan cac bien so ----
    fig, ax = plt.subplots(figsize=(6, 5))
    corr_cols = NUMERIC_FEATURES + ["price"]
    corr = df[corr_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="crest", ax=ax, vmin=-1, vmax=1)
    ax.set_title("Hinh 4: Tuong quan cac bien so voi gia")
    savefig(fig, "04_correlation_heatmap.png")

    # ---- Hinh 5: Gia theo hang xe (top 12 hang nhieu tin nhat) ----
    top_brands = df["brand"].value_counts().head(12).index
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df[df["brand"].isin(top_brands)], x="brand", y="price",
                order=top_brands, ax=ax, color="#3a7ca5")
    ax.set_yscale("log")
    ax.set_title("Hinh 5: Phan bo gia theo hang xe (12 hang pho bien nhat, truc log)")
    ax.set_ylabel("Gia (trieu VND, log scale)")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    savefig(fig, "05_price_by_brand.png")

    # ---- Hinh 6: Gia theo loai hop so & xuat xu ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.boxplot(data=df, x="transmission", y="price", ax=axes[0], color="#81c3d7")
    axes[0].set_yscale("log")
    axes[0].set_title("Gia theo hop so")
    sns.boxplot(data=df, x="assemble_place", y="price", ax=axes[1], color="#2f6690")
    axes[1].set_yscale("log")
    axes[1].set_title("Gia theo xuat xu (lap rap)")
    plt.setp(axes[1].get_xticklabels(), rotation=15, ha="right")
    fig.suptitle("Hinh 6: Gia theo hop so va noi lap rap")
    savefig(fig, "06_price_by_transmission_origin.png")

    # ---- Hinh 7: Scatter km & nam vs gia ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sample = df.sample(min(4000, len(df)), random_state=42)
    axes[0].scatter(sample["driven_kms"], sample["price"], s=6, alpha=0.35, color="#2f6690")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("So km da di")
    axes[0].set_ylabel("Gia (trieu VND, log)")
    axes[0].set_title("Km da di vs Gia")
    axes[1].scatter(sample["year"], sample["price"], s=6, alpha=0.35, color="#3a7ca5")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Nam san xuat")
    axes[1].set_title("Nam san xuat vs Gia")
    fig.suptitle("Hinh 7: Quan he giua km/nam san xuat va gia")
    savefig(fig, "07_scatter_kms_year_vs_price.png")

    print("\nHoan tat EDA. Da luu", len(os.listdir(OUT_DIR)), "hinh vao", OUT_DIR)


if __name__ == "__main__":
    main()
