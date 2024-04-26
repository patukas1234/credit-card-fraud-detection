import numpy as np
import pandas as pd
import yaml
import mlflow
import os
import argparse
import importlib
import logging

from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from mlflow.models.signature import infer_signature

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


def train_val_test_split(X, y, test_size=0.2, val_size=0.2):
    test_size_val = val_size / (1 - test_size)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=1
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, stratify=y_train, test_size=test_size_val, random_state=1
    )  # 0.25 x 0.8 = 0.2
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_iso_forest_score(sklearn_score):
    return -1 * sklearn_score + 0.5


def add_feature_data(X, new_features):
    X_aug = np.hstack((X, np.array(new_features).reshape(-1, 1)))
    return X_aug


def get_iso_features(X, clf):
    sklearn_scores = clf.decision_function(X)
    iso_scores = [get_iso_forest_score(s) for s in sklearn_scores]
    return iso_scores


def get_iso_augmented_features(X, clf):
    new_features = get_iso_features(X, clf)
    X_aug = add_feature_data(X, new_features)
    return X_aug


def get_oversampled_data(X, y, sampling_method=SMOTE, oversample_ratio=2):
    num_minority_samples = int(y.sum() * oversample_ratio)
    oversample = sampling_method(
        sampling_strategy={0: y[y == 0].shape[0], 1: num_minority_samples},
        random_state=42,
    )
    X_smote, y_smote = oversample.fit_resample(X, y)
    return X_smote, y_smote


def task_preprocess(experiment_name, file_path):
    data_path = file_path
    output_dir = config["preprocess"]["output_dir"]
    augment_bool = config["preprocess"]["anomaly_detection"]
    oversample_bool = config["preprocess"]["class_imbalance"]["oversample"]
    oversample_ratio = config["preprocess"]["class_imbalance"]["ratio"]

    existing_exp = mlflow.get_experiment_by_name(experiment_name)
    if not existing_exp:
        mlflow.create_experiment(experiment_name, artifact_location="...")
        logger.info(f"Experiment {experiment_name} created")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="preprocess") as run:
        Path(output_dir).mkdir(exist_ok=True)
        data = pd.read_csv(data_path)

        feat_columns_names = data.columns.values[:30]
        X = data[feat_columns_names].values
        y = data["Class"].values
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
            X, y, test_size=0.2, val_size=0.2
        )

        # Augmenting features with the anomaly score from isolation forests
        if augment_bool:
            logger.info("Augmenting features with anomaly score")
            iso_model = IsolationForest(random_state=42).fit(X_train)
            signature = infer_signature(
                model_input=X_train, model_output=iso_model.predict(X_train)
            )
            mlflow.sklearn.log_model(
                iso_model,
                artifact_path="anomaly_detection",
                signature=signature,
                registered_model_name="isoforest",
            )

            X_train = get_iso_augmented_features(X_train, iso_model)
            X_val = get_iso_augmented_features(X_val, iso_model)
            X_test = get_iso_augmented_features(X_test, iso_model)

        if oversample_bool:
            oversampler = config["preprocess"]["class_imbalance"]["attr"]
            logger.info(f"Oversampling train data with {oversampler}")
            module = importlib.import_module(
                config["preprocess"]["class_imbalance"]["module"]
            )
            sampler = getattr(module, config["preprocess"]["class_imbalance"]["attr"])
            X_train, y_train = get_oversampled_data(
                X_train, y_train, sampler, oversample_ratio=oversample_ratio
            )

        train_processed = np.hstack((X_train, y_train.reshape(-1, 1)))
        val_processed = np.hstack((X_val, y_val.reshape(-1, 1)))
        test_processed = np.hstack((X_test, y_test.reshape(-1, 1)))

        np.savetxt(
            os.path.join(output_dir, "train_data_preprocessed.csv"),
            train_processed,
            delimiter=",",
        )
        np.savetxt(
            os.path.join(output_dir, "val_data_preprocessed.csv"),
            val_processed,
            delimiter=",",
        )
        np.savetxt(
            os.path.join(output_dir, "test_data_preprocessed.csv"),
            test_processed,
            delimiter=",",
        )

        mlflow.log_artifact(output_dir, artifact_path="data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-name", dest="experiment_name", type=str, default="test_exp"
    )
    parser.add_argument(
        "--file-path", dest="file_path", type=str, default=config["data"]["file_path"]
    )

    run_parameters = vars(parser.parse_args())
    task_preprocess(**run_parameters)
