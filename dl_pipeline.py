"""LEVEL: DL - neural networks (Keras MLP) for classification & regression."""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from common import prepare_data, evaluate_classifier, evaluate_regressor, SEED


def plot_history(h, keys, title, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for k in keys:
        ax.plot(h.history[k], label=k)
    ax.set_xlabel("Epoch"); ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def build_mlp(n_in, n_out, task):
    m = keras.Sequential([
        layers.Input(shape=(n_in,)),
        layers.Dense(128, activation="relu"), layers.BatchNormalization(), layers.Dropout(0.3),
        layers.Dense(64, activation="relu"), layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(n_out, activation="softmax" if task == "clf" else "linear"),
    ])
    if task == "clf":
        m.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    else:
        m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return m


def main(data, outdir, epochs):
    os.makedirs(outdir, exist_ok=True)
    tf.random.set_seed(SEED); np.random.seed(SEED)
    d = prepare_data(data)
    Xtr, Xte = d["Xtr"].astype("float32"), d["Xte"].astype("float32")
    stop = keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)

    # ---------------- Classification ----------------
    n_cls = len(d["class_names"])
    cw = compute_class_weight("balanced", classes=np.unique(d["yc_train"]), y=d["yc_train"])
    clf = build_mlp(Xtr.shape[1], n_cls, "clf")
    h = clf.fit(Xtr, d["yc_train"], validation_split=0.2, epochs=epochs, batch_size=64,
                class_weight=dict(enumerate(cw)), callbacks=[stop], verbose=2)
    plot_history(h, ["loss", "val_loss"], "DL Classifier - Loss", os.path.join(outdir, "dl_clf_loss.png"))
    plot_history(h, ["accuracy", "val_accuracy"], "DL Classifier - Accuracy", os.path.join(outdir, "dl_clf_acc.png"))
    proba = clf.predict(Xte, verbose=0)
    res = evaluate_classifier("DL MLP Classifier", d["yc_test"], proba.argmax(1), proba,
                              d["class_names"], outdir)
    clf_df = pd.DataFrame([res]).round(4)
    clf_df.to_csv(os.path.join(outdir, "dl_classification_metrics.csv"), index=False)
    print("\n=== DL CLASSIFICATION ===\n", clf_df.to_string(index=False))

    # ---------------- Regression (target scaled for stable training) ----------------
    ysc = StandardScaler()
    ytr = ysc.fit_transform(d["yr_train"].reshape(-1, 1)).ravel()
    reg = build_mlp(Xtr.shape[1], 1, "reg")
    h = reg.fit(Xtr, ytr, validation_split=0.2, epochs=epochs, batch_size=64,
                callbacks=[stop], verbose=2)
    plot_history(h, ["loss", "val_loss"], "DL Regressor - Loss", os.path.join(outdir, "dl_reg_loss.png"))
    pred = ysc.inverse_transform(reg.predict(Xte, verbose=0)).ravel()
    reg_df = pd.DataFrame([evaluate_regressor("DL MLP Regressor", d["yr_test"], pred, outdir)]).round(4)
    reg_df.to_csv(os.path.join(outdir, "dl_regression_metrics.csv"), index=False)
    print("\n=== DL REGRESSION ===\n", reg_df.to_string(index=False))
    print(f"\nAll outputs saved in: {outdir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=r"C:\Users\prade\Downloads\waste_to_energy_ml_dl\waste_to_energy\src\waste_data_large.csv")
    p.add_argument("--out", default="outputs/dl")
    p.add_argument("--epochs", type=int, default=100)
    a = p.parse_args()
    main(a.data, a.out, a.epochs)
