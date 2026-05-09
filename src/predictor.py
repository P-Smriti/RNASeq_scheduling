"""Train and evaluate 5 regression models for runtime prediction."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score
from utils import STAGES, N_FEATURES, encode_features


def prepare_features(df):
    X = np.zeros((len(df), N_FEATURES))
    for i, r in enumerate(df.itertuples()):
        X[i] = encode_features(r.stage, r.input_size_factor)
    return X, df["runtime"].values


def train_all_models(X_train, y_train):
    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=1.0),
        "SVR": SVR(kernel="rbf", C=1000, gamma=0.01),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    }
    for m in models.values():
        m.fit(X_train, y_train)
    return models


def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        pred = model.predict(X_test)
        results[name] = {
            "mape": mean_absolute_percentage_error(y_test, pred) * 100,
            "mae": mean_absolute_error(y_test, pred),
            "r2": r2_score(y_test, pred),
            "predictions": pred,
        }
    return results


def run_prediction_pipeline(data_path="data/training_data.csv", test_size=0.2, seed=42):
    df = pd.read_csv(data_path)
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    print(f"Data: {len(df)} -> {len(X_train)} train + {len(X_test)} test")

    models = train_all_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)

    # stage average baseline
    sm = df.groupby("stage")["runtime"].mean().to_dict()
    sa_pred = np.array([sm[STAGES[int(X_test[i,:6].argmax())]] for i in range(len(X_test))])
    sa_mape = mean_absolute_percentage_error(y_test, sa_pred) * 100

    print(f"\n{'Model':15s} {'MAPE':>7s} {'MAE':>7s} {'R2':>7s}")
    print("-" * 38)
    for n, r in results.items():
        print(f"{n:15s} {r['mape']:6.1f}% {r['mae']:6.0f}s {r['r2']:6.3f}")
    print(f"{'Stage Avg':15s} {sa_mape:6.1f}%")

    best = min(results, key=lambda n: results[n]["mape"])
    print(f"\nBest: {best} ({results[best]['mape']:.1f}%)")
    return models, results, best, X_train, X_test, y_train, y_test, df, sa_mape

if __name__ == "__main__":
    run_prediction_pipeline()
