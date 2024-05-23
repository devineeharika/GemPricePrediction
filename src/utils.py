import os
import sys

import numpy as np 
import pandas as pd
import dill
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
import mlflow
import mlflow.sklearn
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import ParameterGrid
import dagshub
dagshub.init(repo_owner='devineeharika', repo_name='GemPricePrediction', mlflow=True)


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        MLFLOW_TRACKING_URI = "https://dagshub.com/devineeharika/GemPricePrediction.mlflow"
        MLFLOW_TRACKING_USERNAME ="devineeharika"
        MLFLOW_TRACKING_PASSWORD=os.getenv('DAGSHUB_TOKEN')
        mlflow.set_experiment('gem_price_prediction')
       
        


        report = {}

        overall_best_score = -float('inf')
        overall_best_model_name = None
        overall_best_model = None
        overall_best_params = None

        for model_name, model in models.items():
            params = param.get(model_name, {})
            
            gs = GridSearchCV(model, params, cv=3)
            gs.fit(X_train, y_train)
            
            best_model = gs.best_estimator_
            best_params = gs.best_params_

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

            with mlflow.start_run(run_name=f"best_{model_name}_model"):
                mlflow.log_params(best_params)
                mlflow.log_metric("best_train_r2_score", train_model_score)
                mlflow.log_metric("best_test_r2_score", test_model_score)
                mlflow.sklearn.log_model(best_model, f"best_{model_name}_model")

            if test_model_score > overall_best_score:
                overall_best_score = test_model_score
                overall_best_model_name = model_name
                overall_best_model = best_model
                overall_best_params = best_params

        with mlflow.start_run(run_name="overall_best_model"):
            mlflow.log_params(overall_best_params)
            mlflow.log_metric("best_test_r2_score", overall_best_score)
            mlflow.sklearn.log_model(overall_best_model, "overall_best_model")
            mlflow.log_param("best_model_name", overall_best_model_name)

            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        return report



    except Exception as e:
        raise CustomException(e, sys)

        
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)