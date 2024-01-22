# handle missing value
# outlier treatment
# handle imbalance dataset
# covert categorical columns into numerical columns

import os, sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

@dataclass
class DataTransformationConfig:
    preprocess_obj_file_path = os.path.join("artifacts/data_transformation","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            logging.info("Data Transformation started")
            numerical_features = ['age','workclass', 'education_num', 'marital_status', 'occupation',
                                  'relationship','race','sex','capital_gain','capital_loss',
                                  'hours_per_week', 'native_country']
            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = 'median')),
                    ('scaler', StandardScaler()),
                ]
            )
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_features)
            ])
            return preprocessor
        
        except Exception as e:
            raise CustomException
        
    def remove_outliers_IQR(self, col, df):
        try:
            Q1 = df[col].quartile(0.25)
            Q3 = df[col].quartile(0.75)
            IQR = Q3 - Q1
            upper_limit = Q3 + 1.5*IQR
            lower_limit = Q1 - 1.5*IQR

            df.loc[(df[col]>upper_limit), col] = upper_limit
            df.loc[(df[col]<lower_limit), col] = lower_limit

        except Exception as e:
            logging.info("outler handling code")
            raise CustomException
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            numerical_features = ['age','workclass', 'education_num', 'marital_status', 'occupation',
                                  'relationship','race','sex','capital_gain','capital_loss',
                                  'hours_per_week', 'native_country']
            for col in numerical_features:
                self.remove_outliers_IQR(col = col, df = train_data)
            logging.info("outliers capped on our train data")

            for col in numerical_features:
                self.remove_outliers_IQR(col = col, df = test_data)
            logging.info("outlier capped on our test data")

            preprocess_obj = self.get_data_transformation_obj()

        except Exception as e:
            raise CustomException