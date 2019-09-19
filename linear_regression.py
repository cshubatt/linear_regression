### Tools for linear regression ###

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


def simulate_data():
    """
    Simulates data for testing linear_regression models.
    INPUT
        nobs (int) the number of observations in the dataset
    RETURNS
        data (dict) contains X, y, and beta vectors.
    """
    pass


def compare_models(X, y, beta=None):
    """
    Compares output from different implementations of OLS.
    INPUT
        X (ndarray) the independent variables in matrix form
        y (array) the response variables vector
    RETURNS
     results (pandas.DataFrame) of estimated beta coefficients


    """
    # data is a dictionary
    # get X and make it an array
        #X = np.ndarray(data.get('X'))
    # get y and make it an array
        #y = np.array(data.get('y'))

    # statsmodels
        ols_statsmodels = sm.OLS(y, X).fit() # run sm OLS
        coeff_statsmodels = ols_statsmodels.params # this is an array
            # get the beta coefficients and they should be in a dataframe
        #results_statsmodels = pd.DataFrame(data = coeff_statsmodels.flatten()) # convert array to dataframe

    # sklearn
        ols_sklearn = LinearRegression().fit(y, X) # run sklearn
        coeff_sklearn = ols_sklearn.coef_ # get the coefficients
        #results_sklearn = pd.DataFrame(data = coeff_sklearn.flatten()) # convert array to dataframe

    # put two results into "result"?
        results = pd.DataFrame() # start from an empty dataframe
        results['sklearn'] = coeff_sklearn
        results['statsmodels'] = coeff_statsmodels

    if beta is not None:
        results['truth'] = beta

    return results

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


def run_hospital_regression():
    """
    Loads hospital charge data and runs OLS on it.
    INPUT
        path_to_data (str) filepath of the csv file
    RETURNS
        results (str) the statsmodels regression output
    """
    pass


### END ###
