import os
import os
import glob
import logging
import json
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy.stats import randint, uniform

logging.basicConfig(level=logging.INFO)


def find_processed_csv():
    candidates = [
        os.path.join("weather_preprocessing", "seattle_weather_processed.csv"),
        "seattle_weather_processed.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    matches = glob.glob("**/*processed*.csv", recursive=True)
    if matches:
        logging.info("Found processed csv: %s", matches[0])
        return matches[0]

    raise FileNotFoundError(
        "Processed dataset tidak ditemukan. Pastikan file seattle_weather_processed.csv berada di folder proyek atau di folder 'weather_preprocessing'."
    )


def build_preprocessor(X):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # OneHotEncoder parameter changed name in scikit-learn >=1.2 (sparse -> sparse_output).
    # Create encoder in a try/except for compatibility.
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], remainder="drop")
    return preprocessor


def main(n_iter=50, test_size=0.2, random_state=42):
    data_path = find_processed_csv()
    df = pd.read_csv(data_path)

    if "weather" not in df.columns:
        raise ValueError("Kolom target 'weather' tidak ditemukan.")

    X = df.drop(columns=["weather"]).copy()
    y = df["weather"].copy()

    preprocessor = build_preprocessor(X)

    pipelines = {
        "logreg": Pipeline([
            ("pre", preprocessor),
            ("logreg", LogisticRegression(max_iter=2000, random_state=random_state)),
        ]),
        "rf": Pipeline([
            ("pre", preprocessor),
            ("rf", RandomForestClassifier(random_state=random_state)),
        ]),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Enable MLflow autologging for sklearn
    mlflow.sklearn.autolog()

    best_overall = None
    best_score = -float("inf")

    with mlflow.start_run(run_name="model_tuning"):
        for name, pipe in pipelines.items():
            if name == "logreg":
                param_dist = {
                    "logreg__C": uniform(0.01, 10.0),
                    "logreg__solver": ["lbfgs", "saga"],
                    "logreg__penalty": ["l2"],
                }
            else:
                param_dist = {
                    "rf__n_estimators": randint(50, 400),
                    "rf__max_depth": randint(3, 30),
                    "rf__min_samples_split": randint(2, 10),
                }

            rnd = RandomizedSearchCV(
                pipe,
                param_distributions=param_dist,
                n_iter=n_iter,
                scoring="f1_weighted",
                cv=cv,
                random_state=random_state,
                n_jobs=-1,
                verbose=1,
            )

            logging.info("Tuning %s with %s iterations", name, n_iter)
            rnd.fit(X_train, y_train)

            try:
                mlflow.log_param(f"{name}_best_params", json.dumps(rnd.best_params_))
            except Exception:
                mlflow.log_param(f"{name}_best_params", str(rnd.best_params_))

            mlflow.log_metric(f"{name}_best_cv_score", float(rnd.best_score_))

            y_pred = rnd.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            mlflow.log_metric(f"{name}_test_accuracy", float(acc))
            mlflow.log_metric(f"{name}_test_f1_weighted", float(f1))

            report = classification_report(y_test, y_pred)
            os.makedirs("artifacts", exist_ok=True)
            report_path = os.path.join("artifacts", f"classification_report_{name}.txt")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)
            mlflow.log_artifact(report_path)

            try:
                mlflow.sklearn.log_model(rnd.best_estimator_, artifact_path=f"model_{name}")
            except Exception:
                logging.exception("Gagal log_model ke MLflow untuk %s", name)

            model_joblib_path = os.path.join("artifacts", f"model_{name}.joblib")
            joblib.dump(rnd.best_estimator_, model_joblib_path)
            mlflow.log_artifact(model_joblib_path)

            if f1 > best_score:
                best_score = f1
                best_overall = (name, rnd.best_estimator_, f1, acc, rnd.best_params_)

        if best_overall:
            name, estimator, best_f1, best_acc, best_params = best_overall
            mlflow.log_param("best_model", name)
            mlflow.log_metric("best_model_test_f1_weighted", float(best_f1))
            mlflow.log_metric("best_model_test_accuracy", float(best_acc))
            try:
                mlflow.log_param("best_model_params", json.dumps(best_params))
            except Exception:
                mlflow.log_param("best_model_params", str(best_params))

            os.makedirs("artifacts", exist_ok=True)
            best_model_path = os.path.join("artifacts", "best_model.joblib")
            joblib.dump(estimator, best_model_path)
            mlflow.log_artifact(best_model_path)
            try:
                mlflow.sklearn.log_model(estimator, artifact_path="best_model")
            except Exception:
                logging.exception("Gagal log_model best_model ke MLflow")

        print("Tuning selesai. Best model:", best_overall[0] if best_overall else None)


if __name__ == "__main__":
    main()

