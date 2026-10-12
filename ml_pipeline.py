"""LEVEL: ML - baseline + stronger traditional models (classification & regression)."""
import argparse, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from common import prepare_data, evaluate_classifier, evaluate_regressor, SEED


def main(data, outdir):
    os.makedirs(outdir, exist_ok=True)
    d = prepare_data(data)
    Xtr, Xte = d["Xtr"], d["Xte"]

    # ---------------- Classification: Operational_Status ----------------
    clf_models = {
        "Logistic Regression (baseline)": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows = []
    for name, m in clf_models.items():
        cvf1 = cross_val_score(m, Xtr, d["yc_train"], cv=cv, scoring="f1_weighted").mean()
        m.fit(Xtr, d["yc_train"])
        r = evaluate_classifier(name, d["yc_test"], m.predict(Xte), m.predict_proba(Xte),
                                d["class_names"], outdir)
        r["cv_f1_train"] = cvf1
        rows.append(r)
    clf_df = pd.DataFrame(rows).round(4)
    clf_df.to_csv(os.path.join(outdir, "classification_metrics.csv"), index=False)
    print("\n=== CLASSIFICATION RESULTS (test set) ===\n", clf_df.to_string(index=False))

    # Feature importance (Random Forest)
    rf = clf_models["Random Forest"]
    imp = pd.Series(rf.feature_importances_, index=d["feature_names"]).sort_values()
    imp.to_csv(os.path.join(outdir, "feature_importance_clf.csv"))
    fig, ax = plt.subplots(figsize=(7, 5)); imp.plot.barh(ax=ax)
    ax.set_title("Feature Importance - Random Forest (Operational_Status)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "feature_importance_clf.png"), dpi=130); plt.close(fig)

    # ---------------- Regression: Energy_Generated_MWh ----------------
    reg_models = {
        "Linear Regression (baseline)": LinearRegression(),
        "Random Forest Reg": RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
        "Gradient Boosting Reg": GradientBoostingRegressor(random_state=SEED),
    }
    rows = []
    for name, m in reg_models.items():
        m.fit(Xtr, d["yr_train"])
        rows.append(evaluate_regressor(name, d["yr_test"], m.predict(Xte), outdir))
    reg_df = pd.DataFrame(rows).round(4)
    reg_df.to_csv(os.path.join(outdir, "regression_metrics.csv"), index=False)
    print("\n=== REGRESSION RESULTS (test set) ===\n", reg_df.to_string(index=False))

    imp_r = pd.Series(reg_models["Random Forest Reg"].feature_importances_,
                      index=d["feature_names"]).sort_values()
    imp_r.to_csv(os.path.join(outdir, "feature_importance_reg.csv"))
    fig, ax = plt.subplots(figsize=(7, 5)); imp_r.plot.barh(ax=ax)
    ax.set_title("Feature Importance - Random Forest (Energy_Generated_MWh)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "feature_importance_reg.png"), dpi=130); plt.close(fig)
    print(f"\nAll outputs saved in: {outdir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=r"C:\Users\prade\Downloads\waste_to_energy_ml_dl\waste_to_energy\src\waste_data_large.csv")
    p.add_argument("--out", default="outputs/ml")
    a = p.parse_args()
    main(a.data, a.out)
