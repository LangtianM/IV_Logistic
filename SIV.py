import pandas as pd 
import numpy as np
from scipy import linalg
from sklearn.preprocessing import scale
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFECV

class SIV():
    """
    The SIV class represents the method of Synthetic Instrumental Variables.

    Attributes:
        q (int): The number of latent variables. Defaults to None.
        causal_model (object): The fitted causal model.
        selected_features (array-like): The names of the selected features.
        feature_names (array-like): The names of the input features.
        identifiability (bool): Indicates whether the model is identifiable.
        X_hat (array-like): The predicted values of the input features using SIV. 

    Methods:
        tuning_q(X): Static method that tunes the value of q based on the input features.
        fit(X, Y, n_jobs): Fits the SIV model to the given input data.
        predict(X): Predicts the target variable for the given input data.
        get_coef(): Returns the coefficients of the selected features.

    Example:
        siv = SIV()
        siv.fit(X_train, y_train, n_jobs=2)
        y_pred = siv.predict(X_test)
        coef = siv.get_coef()
    """
    
    
    def __init__(self, q = None):
        self.q = q
        self.causal_model = None
        self.selected_features = None
        self.feature_names = None
        self.identifiability = None
        self.X_hat = None
        
        
    @staticmethod
    def tuning_q(X):
        n, p = X.shape
        rmax =  np.min([10, (p-1)//2])
        
        X = scale(X, with_mean=True, with_std=False)
        Cov = np.cov(X, rowvar=False)
        
        eigenv = linalg.eigh(Cov, eigvals_only=True, subset_by_index=(p-rmax-3, p-1))
        eigenv = eigenv[::-1]
        teststat = np.zeros(rmax)
        
        for i in range(rmax):
            teststat[i] = (eigenv[i] - eigenv[i+1]) / (eigenv[i+1] - eigenv[i+2])
            
        return np.argmax(teststat) + 1
    
    def fit(self, X, Y, n_jobs = 1):
        """
        Fits the SIV model to the given input data.

        Parameters:
            X (array-like or DataFrame): The input features.
            Y (array-like): The target variable.
            n_jobs (int, optional): The number of parallel jobs to run for feature selection. Defaults to 1.

        Returns:
            None

        Raises:
            None

        Notes:
            - The `X` parameter can be either an array-like object or a DataFrame. If it is a DataFrame, the feature names will be extracted from the DataFrame columns.
            - The `Y` parameter should be an array-like object representing the target variable.
            - The `n_jobs` parameter controls the number of parallel jobs to run for feature selection. It determines the number of CPUs to use (-1 for all CPUs, 1 for single CPU).

        Example:
            siv = SIV()
            siv.fit(X_train, y_train, n_jobs=2)
        """
        
        # if X is a dataframe, get the names of the features
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns
        else:
            self.feature_names = None
            
        # Convert X into np arrary
        X = np.array(X)
        Y = np.array(Y)
        
        if self.q is None:
            self.q = self.tuning_q(X)

        X = scale(X, with_mean=True, with_std=False)
        p = X.shape[1]
        n = X.shape[0]
        
        if n > p:
            fa = FactorAnalysis(n_components = self.q)
            fa.fit(X)
            load = np.diag(np.sqrt(np.diag(np.cov(X, rowvar=False)))) @ fa.components_.T # p*p
        else:   
            pca = PCA()
            pca.fit(X)
            load = pca.components_[:self.q].T @\
                np.diag(np.sqrt(pca.explained_variance_[:self.q])) # p*q p > q

        B_orthogonal = linalg.null_space(load.T) # p*(p-q)
        X_SIV = X @ B_orthogonal
        reg = LinearRegression().fit(X_SIV, X)
        self.X_hat = reg.predict(X_SIV)
        
        
        rfecv = RFECV(estimator=LinearRegression(), step=1, cv=10, n_jobs=n_jobs)
        rfecv.fit(self.X_hat, Y)
        
        # Get the names of the selected features
        if self.feature_names is not None:
            self.selected_features = self.feature_names[rfecv.support_]
        else:
            self.selected_features = None
        
        lr = LinearRegression()
        lr.fit(self.X_hat[:, rfecv.support_], Y)
        
        self.causal_model = lr
        
        s = rfecv.n_features_
        
        if(s+1+self.q <= p):
            self.identifiability = True
        else:
            self.identifiability = False
            print('Warning: The model is not identifiable')
            
    def predict(self, X):
        X = np.array(X)
        X = scale(X, with_mean=True, with_std=False)
        return self.causal_model.predict(X)
    
    def get_coef(self):
        # Get coefficients of the selected features with their names
        if self.selected_features is not None:
            return pd.DataFrame({'Feature': self.selected_features, 'Coefficient': self.causal_model.coef_})
        else:
            return self.causal_model.coef_
        
    def get_X_hat(self):
        return pd.DataFrame(self.X_hat, columns=self.feature_names)

