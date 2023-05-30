import mlflow
import yaml
import importlib
import os
import argparse 

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from mlflow.models.signature import infer_signature
from utils import get_eval_metrics, plot_model_performance_metrics

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


def task_train(experiment_name, data_path = None):
    existing_exp = mlflow.get_experiment_by_name(experiment_name)
    if not existing_exp:
        mlflow.create_experiment(experiment_name, artifact_location="...")
        print("Experiment {} created".format(experiment_name))
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name = "train") as run:
        # Load processed data
        train_data = np.loadtxt(os.path.join(data_path, "train_data_preprocessed.csv"), delimiter=',')
        val_data = np.loadtxt(os.path.join(data_path, "val_data_preprocessed.csv"), delimiter=',')
        X_train, y_train = train_data[:, :-1], train_data[:, -1]
        X_val, y_val = val_data[:, :-1], val_data[:, -1]

        #importing clf module 
        module= importlib.import_module(config["train"]["model"]["module"])
        model = getattr(module, config["train"]["model"]["attr"])

        model_params = config["train"]["model"]["params"]
        clf = model(**model_params, random_state = 42).fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        y_proba = clf.predict_proba(X_val) 
        precision, recall, f1, precision_avg, recall_avg, f1_avg = get_eval_metrics(y_val, y_pred)
        
        #Plot confusion mtrx and PR curve
        fig, ax = plt.subplots(1,2, figsize = (12, 5))
        plot_model_performance_metrics(y_val, y_pred, y_proba, ax = ax, **{"model": config["train"]["model"]["name"]})

        #Log model, parameters, metrics, plots as artifacts
        signature = infer_signature(X_val, y_pred)
        mlflow.sklearn.log_model(
            sk_model = clf,
            artifact_path = "fraud_detection",
            signature = signature,
            registered_model_name = config["train"]["model"]["name"],
        )
    
        mlflow.log_params(model_params)
        mlflow.log_metrics(
                {
                    "precision_non-fraud": precision[0],
                    "precision_fraud": precision[1],
                    "recall_non-fraud": recall[0],
                    "recall_fraud": recall[1],
                    "f1_non-fraud": f1[0],
                    "f1_fraud": f1[1],
                    "avg_weighted_precision": precision_avg,
                    "avg_weighted_recall":recall_avg,
                    "avg_weighted_f1": f1_avg
                }
            )
        mlflow.log_figure(fig, '{}_conf_mtrx_pr_curve.png'.format(config["train"]["model"]["name"]))
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", dest='experiment_name', type=str, default="test_exp")
    parser.add_argument("--data-path", dest='data_path', type=str, default=config["train"]["data_path"])
    
    run_parameters = vars(parser.parse_args())
    task_train(**run_parameters)
