import numpy as np
from sklearn.utils.extmath import randomized_svd


def ols_regression(X, Y):
    """
    Performs standard linear regression
    :param X: Feature matrix (n, p)
    :param Y: Dependent data (n, m)
    :return: Linear regression coefficients (p, m)
    """
    return np.linalg.solve(np.matmul(X.T, X), np.matmul(X.T, Y))


def ridge_regression(X, Y, lamda):
    """
    Performs ridge regression
    :param X: Feature matrix (n, p)
    :param Y: Dependent data (n, m)
    :param lamda: Regularisation parameter
    :return: Regression coefficients (p, m)
    """
    return np.linalg.solve(np.matmul(X.T, X) + lamda * np.eye(X.shape[1]), np.matmul(X.T, Y))


def covariance_regression(X, Y, cov):
    """
    Performs regression with pre-computed prior belief covariance matrix
    :param X: Feature matrix (n, p)
    :param Y: Dependent data (n, m)
    :param cov: Prior belief covariance matrix
    :return: Regression coefficients (p, m)
    """
    return np.linalg.solve(np.matmul(X.T, X) + cov, np.matmul(X.T, Y))


class RRR:
    """
    Class for implementing reduced rank regression
    Includes methods for making predictions and generating the regression coefficients with
    any desired rank after fitting once
    """

    def __init__(self, rank=30):
        """
        Initialises variables that will be used later and sets the rank
        :param rank: Maximum rank for later analysis
        """
        self.V = 0
        self.bhat = 0
        self.rank = rank


    def fit(self, bhat, X=None, yhat=None):
        """
        Calculates the fitted values yhat if not passed and then an SVD of yhat for use later
        :param bhat: Fitted linear coefficients
        :param X: Feature matrix (optional, not required if yhat passed directly)
        :param yhat: Fitted values with full rank (optional, speeds up computation if passed)
        :return:
        """
        self.bhat = bhat

        if yhat is None:
            yhat = np.matmul(X, bhat)

        _, _, self.V = randomized_svd(yhat, n_components=self.rank)


    def predict(self, X, rank):
        """
        For a given feature matrix predicts values using low rank coefficients
        :param X: Feature matrix
        :param rank: Rank of linear coefficients
        :return: Predictions
        """
        if rank > self.rank:
            print(f'Requested prediction rank exceeds fitted rank of {self.rank}')
            return None
        else:
            W = self.V[0:rank, :].T
            A = np.matmul(self.bhat, W)
            return np.matmul(X, np.matmul(A, W.T))


    def gen_bhat(self, rank):
        """
        Outputs fitted linear coefficients for the desired rank
        :param rank: Rank of linear coefficients
        :return: Coefficients
        """
        if rank > self.rank:
            print(f'Requested prediction rank exceeds fitted rank of {self.rank}')
            return None
        else:
            W = self.V[0:rank, :].T
            A = np.matmul(self.bhat, W)
            return np.matmul(A, W.T)


def r2(y, yhat, var=None):
    """
    Function for calculating the r2 statistic per target variable. When the target variable is
    constant the returned value is 0
    :param y: Array of target variables (must be numpy array)
    :param yhat: Predicted values (must be numpy array)
    :param var: Optional, pre-computed variance of array to speed up calculations
    :return: Array of r2 values for each target variable
    """
    if var is None:
        var = y.var(axis=0)

    mask = var == 0
    return (var - np.mean((y - yhat) ** 2, axis=0)) / (var + mask)
