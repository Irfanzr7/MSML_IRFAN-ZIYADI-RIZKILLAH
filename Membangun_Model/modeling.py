import os
import glob
import logging
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

logging.basicConfig(level=logging.INFO)

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Membangun_Model")

dagshub.init(
    repo_owner="Irfanzr7",
    repo_name="MSML_IRFAN-ZIYADI-RIZKILLAH",
    mlflow=True
)


def find_processed_csv():
    # Coba beberapa lokasi umum lalu cari secara rekursif
    candidates = [
        os.path.join("weather_preprocessing", "seattle_weather_processed.csv"),
        "seattle_weather_processed.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    # Cari file processed secara rekursif di folder kerja
    matches = glob.glob("**/*processed*.csv", recursive=True)
    if matches:
        logging.info("Found processed csv: %s", matches[0])
        return matches[0]

    raise FileNotFoundError(
        "Processed dataset tidak ditemukan. Pastikan file seattle_weather_processed.csv berada di folder proyek atau di folder 'weather_preprocessing'."
    )


def main():
    data_path = find_processed_csv()
    df = pd.read_csv(data_path)

    if "weather" not in df.columns:
        raise ValueError("Kolom target 'weather' tidak ditemukan di dataset processed.")

    X = df.drop(columns=["weather"])
    y = df["weather"]

    # Tangani missing values: numeric -> median, categorical -> mode
    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(exclude=[np.number]).columns

    if len(num_cols) > 0:
        X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    for c in cat_cols:
        mode = X[c].mode()
        fill_val = mode.iloc[0] if not mode.empty else ""
        X[c] = X[c].fillna(fill_val)

    # Ubah fitur non-numerik menjadi dummy (one-hot)
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=list(cat_cols), drop_first=True)

    # Pastikan tidak ada kolom non-numeric tersisa
    if X.select_dtypes(exclude=[np.number]).shape[1] > 0:
        raise ValueError("Masih ada kolom non-numerik setelah preprocessing.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="logreg_baseline"):
        # Gunakan solver yang sesuai untuk multiclass dan lebih iterasi jika perlu
        model = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=42
        )
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            logging.exception("Gagal melatih model: %s", e)
            raise

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        mlflow.log_metric("test_accuracy_manual", acc)
        mlflow.log_metric("test_f1_weighted_manual", f1)

        report = classification_report(y_test, y_pred)
        os.makedirs("artifacts", exist_ok=True)
        report_path = os.path.join("artifacts", "classification_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        print("Training selesai.")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1-weighted: {f1:.4f}")


if __name__ == "__main__":
    main()