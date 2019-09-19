### Tools for linear regression ###

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


def simulate_data():
    nobs=500
    x1 = np.random.exponential(scale=1/9000, size=nobs)
    x2 = np.random.poisson(lam=15, size=nobs)
    alpha = np.ones((nobs,1))
    X = np.column_stack((alpha,x1,x2))
    beta = np.array([[5],[1],[1]])
    epsilon =  np.random.normal(0, 1, nobs)
    y=np.dot(X,beta) + epsilon.reshape(nobs,1)
    data={"y": y, "X": X, "beta": beta}
    return(data) 



def compare_models():
    """
    Compares output from different implementations of OLS.
    INPUT
        X (ndarray) the independent variables in matrix form
        y (array) the response variables vector
    RETURNS
        results (pandas.DataFrame) of estimated beta coefficients
    """
    pass


def load_hospital_data():
    """
    Loads the hospital charges data set found at data.gov.
    INPUT
        path_to_data (str) indicates the filepath to the hospital charge data (csv)
    RETURNS
        clean_df (pandas.DataFrame) containing the cleaned and formatted dataset for regression
    """
    pass


def prepare_data():
    """
    Prepares hospital data for regression (basically turns df into X and y).
    INPUT
        df (pandas.DataFrame) the hospital dataset
    RETURNS
        data (dict) containing X design matrix and y response variable
    """
    pass


def run_hospital_regression(filepath):
    """
    Loads hospital charge data and runs OLS on it.
    INPUT
        path_to_data (str) filepath of the csv file
    RETURNS
        results (str) the statsmodels regression output
    """
    df = load_hospital_data(filepath)
    data = prepare_data(df)

    y = data['y']
    X = data['X']

    output = sm.regression.linear_model.OLS(y,X)

    return output


### END ###
