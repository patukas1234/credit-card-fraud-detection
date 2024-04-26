import mlflow
import yaml
import pickle
import os
import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from mlflow.models.signature import infer_signature
from utils import get_eval_metrics, plot_model_performance_metrics

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


def task_test(experiment_name, data_path, model_path):
    existing_exp = mlflow.get_experiment_by_name(experiment_name)
    if not existing_exp:
        mlflow.create_experiment(experiment_name, artifact_location="...")
        logger.info(f"Experiment {experiment_name} created")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="test") as run:
        # Load processed data
        test_data = np.loadtxt(
            os.path.join(config["test"]["data_path"], "test_data_preprocessed.csv"),
            delimiter=",",
        )
        X_test, y_test = test_data[:, :-1], test_data[:, -1]

        model_path = config["test"]["model"]["path"]

        with open(os.path.join(model_path, "model.pkl"), "rb") as f:
            clf = pickle.load(f)

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        precision, recall, f1, precision_avg, recall_avg, f1_avg = get_eval_metrics(
            y_test, y_pred
        )

        # Plot confusion mtrx and PR curve
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        plot_model_performance_metrics(
            y_test, y_pred, y_proba, ax=ax, **{"model": config["test"]["model"]["name"]}
        )

        # Log model, parameters, metrics, plots as artifacts
        mlflow.log_metrics(
            {
                "precision_non-fraud": precision[0],
                "precision_fraud": precision[1],
                "recall_non-fraud": recall[0],
                "recall_fraud": recall[1],
                "f1_non-fraud": f1[0],
                "f1_fraud": f1[1],
                "avg_weighted_precision": precision_avg,
                "avg_weighted_recall": recall_avg,
                "avg_weighted_f1": f1_avg,
            }
        )
        mlflow.log_figure(
            fig,
            "{}_conf_mtrx_pr_curve_test.png".format(config["test"]["model"]["name"]),
        )
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-name", dest="experiment_name", type=str, default="test_exp"
    )
    parser.add_argument(
        "--data-path", dest="data_path", type=str, default=config["test"]["data_path"]
    )
    parser.add_argument(
        "--model-path",
        dest="model_path",
        type=str,
        default=config["test"]["model"]["path"],
    )

    run_parameters = vars(parser.parse_args())
    task_test(**run_parameters)
