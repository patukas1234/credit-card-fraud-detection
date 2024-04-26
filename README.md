# MLflow Pipeline for Fraud Detection Task


This repository implements an MLflow model training pipeline to experiment with the [credit card fraud kaggle task](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and a prediction api.


# Getting started

1. Install poetry 
2. Install dependencies and activate the virtual environment  `poetry install` and `poetry shell`
3. Download [kaggle credit card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) to `./train/data`

## Prediction API

For local runs, create a .env file (see .example.env) to specify a trained model file path, host (localhost default), port (8000 default) and api token. 

From project root directory run `python predict/app.py`

The api is implemented with FastAPI, which includes built-in Swagger UI suppoer. Go to `http://<host>:<port>/docs` to interact with the api.

## Training

Data preprocessing, model training and test parameters can be set in `config.yaml`

## Tracked experimentation with MLflow:

- `preprocess.py`: 
    - performs train-validation-test split 
    - optionally uses isolation forest to derive an anomaly score and add it to the feature vector. E.g. to add anomaly score to features set `config.yaml` params:
    ```
    ...
    anomaly_detection: True
    ...
    ```
    - optionally performs oversampling of the minority class. E.g. to use SMOTE oversampling set `config.yaml` params:
    ```
    ...
    class_imbalance:
        oversample: True
        module: "imblearn.over_sampling"
        attr: "SMOTE"
        ratio: 3
    ...
    ```
    - Perform preprocessing run and log preprocessed data as artifacts using:

    ``` 
    python preprocess.py --experiment-name <TEXT>
    ```
- `train.py`: 
    - Train a classifier specified in `config.yaml` params. E.g.:
    ```
    ...
    model: 
        module: "sklearn.linear_model"
        attr: "LogisticRegression" 
        name: "logistic_regression"
        params: {"penalty": "l2", "C": 0.6}
    
    ```
    - Train and log train and validation performance metrics, confusion matrix and PR curves:
    ``` 
    python train.py --experiment-name <TEXT>
    ```
- `test.py`: 
    - Test the trained classifier. Trained model file path specified in `config.yaml` params. E.g.:
    ```
    ...
    model: 
        path: "./mlruns/979822066418382618/e894710ffd014405a269337d32fc341c/artifacts/fraud_detection"
        name: "logistic_regression"
    ...
    ```
    - Log test set performance metrics, confusion matrix and PR curves:
    ``` 
    python test.py --experiment-name <TEXT>
    ```
- `main.py`: 
    - implements an end-to-end workflow of `preprocess`, `tain` and `test` entrypoints . 
    ``` 
    python main.py --experiment-name <TEXT>
    ```

