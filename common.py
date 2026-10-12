"""Shared preprocessing, feature engineering and evaluation helpers."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix,
                             ConfusionMatrixDisplay, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.preprocessing import label_binarize

SEED = 42
CAT_COLS = ["Waste_Type"]
NUM_COLS = ["Waste_Collected_Tons", "Moisture_Content_Percent",
            "Ambient_Temperature_C", "Plant_Efficiency_Percent"]
ENGINEERED = ["Dry_Waste_Tons", "Effective_Waste_Tons"]
REG_TARGET = "Energy_Generated_MWh"
CLF_TARGET = "Operational_Status"


def add_features(df):
    """Feature engineering (none of these use the targets -> no leakage)."""
    df = df.copy()
    df["Dry_Waste_Tons"] = df["Waste_Collected_Tons"] * (1 - df["Moisture_Content_Percent"] / 100)
    df["Effective_Waste_Tons"] = df["Dry_Waste_Tons"] * df["Plant_Efficiency_Percent"] / 100
    return df


def prepare_data(csv_path, test_size=0.2, seed=SEED):
    df = pd.read_csv(csv_path)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    miss = df.isna().sum()
    print("Missing values per column:\n", miss[miss > 0])

    # Rows without a target are unusable; missing features are imputed later
    df = df.dropna(subset=[REG_TARGET, CLF_TARGET]).drop_duplicates()
    df = add_features(df)

    feature_cols = CAT_COLS + NUM_COLS + ENGINEERED
    X = df[feature_cols]
    y_reg = df[REG_TARGET].values

    le = LabelEncoder()                      # only the TARGET is label-encoded
    y_clf = le.fit_transform(df[CLF_TARGET])
    print("Class distribution:\n", pd.Series(le.inverse_transform(y_clf)).value_counts())

    # ONE split for everything -> rows stay aligned across targets
    (X_train, X_test, yr_train, yr_test,
     yc_train, yc_test) = train_test_split(X, y_reg, y_clf, test_size=test_size,
                                           random_state=seed, stratify=y_clf)

    pre = ColumnTransformer([
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT_COLS),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM_COLS + ENGINEERED),
    ])
    # fit on TRAIN only (prevents data leakage)
    Xtr = pre.fit_transform(X_train)
    Xte = pre.transform(X_test)
    names = list(pre.get_feature_names_out())
    return dict(Xtr=Xtr, Xte=Xte, yr_train=yr_train, yr_test=yr_test,
                yc_train=yc_train, yc_test=yc_test, feature_names=names,
                class_names=[str(c) for c in le.classes_], preprocessor=pre, label_encoder=le)


def evaluate_classifier(name, y_true, y_pred, proba, class_names, outdir, save_plots=True):
    n = len(class_names)
    res = {"model": name,
           "accuracy": accuracy_score(y_true, y_pred),
           "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
           "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
           "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0)}
    try:
        if n == 2:
            res["roc_auc"] = roc_auc_score(y_true, proba[:, 1])
        else:
            res["roc_auc"] = roc_auc_score(y_true, proba, multi_class="ovr", average="weighted")
    except ValueError:
        res["roc_auc"] = np.nan

    if save_plots:
        tag = "".join(ch if ch.isalnum() else "_" for ch in name)
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"Confusion Matrix - {name}", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"cm_{tag}.png"), dpi=130); plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 4))
        if n == 2:
            fpr, tpr, _ = roc_curve(y_true, proba[:, 1])
            ax.plot(fpr, tpr, label=f"AUC={res['roc_auc']:.3f}")
        else:
            Y = label_binarize(y_true, classes=range(n))
            for i, c in enumerate(class_names):
                fpr, tpr, _ = roc_curve(Y[:, i], proba[:, i])
                ax.plot(fpr, tpr, label=f"{c}")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve - {name}", fontsize=9); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"roc_{tag}.png"), dpi=130); plt.close(fig)
    return res


def evaluate_regressor(name, y_true, y_pred, outdir=None):
    res = {"model": name,
           "MAE": mean_absolute_error(y_true, y_pred),
           "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
           "R2": r2_score(y_true, y_pred)}
    if outdir:
        tag = "".join(ch if ch.isalnum() else "_" for ch in name)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y_true, y_pred, s=8, alpha=0.5)
        lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "r--")
        ax.set_xlabel("Actual MWh"); ax.set_ylabel("Predicted MWh"); ax.set_title(f"Actual vs Predicted - {name}", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"reg_{tag}.png"), dpi=130); plt.close(fig)
    return res
