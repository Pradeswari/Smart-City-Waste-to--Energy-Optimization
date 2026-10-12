# Smart City Waste-to-Energy Optimization (ML + DL)

## Structure
```
data/            <- put waste_data_large.csv here
src/common.py            shared preprocessing, feature engineering, metrics/plots
src/ml_pipeline.py       ML level: LogReg, Decision Tree, Random Forest, Gradient Boosting
src/dl_pipeline.py       DL level: Keras MLP (classification + regression)
src/generate_sample_data.py   synthetic data, only to test the code
outputs/         <- metrics CSVs + plots are written here
```

## How to run
```
pip install -r requirements.txt
# copy your CSV to data/waste_data_large.csv  (or pass --data "full\path\file.csv")
cd src
python ml_pipeline.py --data "d:\PYTHON\SMART CITY -- DS\2nd try\waste_data_large.csv"
python dl_pipeline.py --data "d:\PYTHON\SMART CITY -- DS\2nd try\waste_data_large.csv"
```
(Python 3.9-3.12 recommended for TensorFlow.)

## Problem
- Classification: predict `Operational_Status`
- Regression: predict `Energy_Generated_MWh`
- Features: Waste_Type, Waste_Collected_Tons, Moisture_Content_Percent,
  Ambient_Temperature_C, Plant_Efficiency_Percent (+ engineered features)

## Preprocessing pipeline
1. Drop rows with missing targets and duplicate rows.
2. Impute remaining missing values (median for numeric, most-frequent for categorical).
3. One-hot encode `Waste_Type` (nominal, so no fake ordering as with LabelEncoder).
4. StandardScaler on numeric features.
5. A SINGLE stratified 80/20 split (random_state=42) for both targets.
6. Imputer/encoder/scaler are fit on the training set only (no data leakage).

## Feature engineering report
| Feature | Formula | Reason |
|---|---|---|
| Dry_Waste_Tons | Waste_Collected_Tons x (1 - Moisture/100) | Moisture lowers burnable mass |
| Effective_Waste_Tons | Dry_Waste_Tons x Efficiency/100 | Usable waste after plant losses |

Neither uses a target column, so there is no leakage. Check `outputs/ml/feature_importance_*.png`
to see which features matter and discuss it in your report.

## Models
- ML: Logistic Regression (baseline), Decision Tree, Random Forest, Gradient Boosting;
  class_weight="balanced" handles class imbalance; 5-fold stratified CV F1 on train.
- Regression: Linear Regression (baseline), Random Forest, Gradient Boosting.
- DL: MLP 128-64-32 (ReLU, BatchNorm, Dropout), Adam lr=1e-3, batch 64, up to 100 epochs,
  EarlyStopping(patience=15), class weights, scaled regression target.

## Metrics produced (in outputs/)
Accuracy, Precision, Recall, F1, ROC-AUC (weighted OvR for multiclass), confusion matrix,
ROC curves, feature importance, MAE / RMSE / R2, actual-vs-predicted plots, DL loss curves.

## Fix vs. the original script
The original called train_test_split twice without a seed, so classifier features and labels
were misaligned. Now there is one split for all targets.

## Report tips
- Explain WHY the best model won (compare against the baseline).
- If Plant_Efficiency or Effective_Waste_Tons dominate importance, say what that means operationally.
- Mention limits: synthetic/limited data, class imbalance, no hyperparameter tuning yet.
