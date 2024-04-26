import mlflow
import os
import argparse
import yaml
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


def workflow(experiment_name, file_path):
  existing_exp = mlflow.get_experiment_by_name(experiment_name)
  if not existing_exp:
      mlflow.create_experiment(experiment_name, artifact_location="...")
      logger.info(f"Experiment {experiment_name} created")
  mlflow.set_experiment(experiment_name)

  current_experiment=dict(mlflow.get_experiment_by_name(experiment_name))
  experiment_id=current_experiment['experiment_id']
  artifact_path_local = os.path.join("./mlruns", experiment_id)

  with mlflow.start_run() as active_run:
    logger.info("Launching 'preprocess'")
    logger.info(f"experiment name {experiment_name}")

    preprocess_run = mlflow.run(".", "preprocess", parameters={"file-path": file_path,
                                                                "experiment-name": experiment_name})
    #preprocess_run = mlflow.tracking.MlflowClient().get_run(preprocess_run.run_id)
    processed_data_path_uri = os.path.join(artifact_path_local, preprocess_run.run_id, "artifacts/data/processed")


    logger.info("Launching 'train'")
    logger.info(f"experiment name {experiment_name}")
    train_run = mlflow.run(".", "train", parameters={"data-path": processed_data_path_uri,
                                                                   "experiment-name": experiment_name})
    #train_run = mlflow.tracking.MlflowClient().get_run(train_run.run_id)
    train_model_path_uri = os.path.join(artifact_path_local, train_run.run_id, "artifacts/fraud_detection")


    logger.info("Launching 'test'")
    logger.info(f"experiment name {experiment_name}")
    test_run = mlflow.run(".", "test", parameters={"data-path": processed_data_path_uri,
                                                    "model-path": train_model_path_uri,
                                                    "experiment-name": experiment_name})
    #test_run = mlflow.tracking.MlflowClient().get_run(test_run.run_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", dest='experiment_name', type=str, default="test_exp")
    parser.add_argument("--file-path", dest='file_path', type=str, default=config["data"]["file_path"])
    run_parameters = vars(parser.parse_args()) 

    workflow(**run_parameters)