from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import seaborn as sns
import numpy as np


class sim_data:
    def __init__(self, n, sigma_z=1, sigma_u=1, sigma_e=0.01, 
                 alpha=1, beta=1, gamma=1, eta = 1):
        super().__init__()
        self.Z = np.random.normal(0, sigma_z, n)
        self.U = np.random.normal(0, sigma_u, n)
        epsilon = np.random.normal(0, sigma_e, n)
        self.X = alpha * self.Z + gamma * self.U + epsilon
        self.g_p = beta * self.X + eta * self.U
        self.p = 1 / (1 + np.exp(-self.g_p))
        self.Y = np.random.binomial(1, self.p)
        
    def plot_Z_X(self):
        sns.scatterplot(x=self.Z, y=self.X)
        
    def plot_U_X(self):
        sns.scatterplot(x=self.U, y=self.X)
        
    def plot_p(self):
        sns.histplot(self.p)


class IVModel:
    def __init__(self) -> None:
        pass
    
    def fit(self):
        raise NotImplementedError
    
    def predict(self):
        raise NotImplementedError
    

class two_stage_logit(IVModel):
    
    def __init__(self, predictor=LinearRegression()):
        self.X_hat = None  
        self.predictor = predictor
    
    def fit(self, X, Y, Z):
        self.predictor.fit(Z.reshape(-1, 1), X)
        self.X_hat = self.predictor.predict(Z.reshape(-1, 1))
        lg = LogisticRegression()
        lg.fit(self.X_hat.reshape(-1, 1), Y)
        self.ypredictor = lg
        self.coef_ = lg.coef_
        self.intercept_ = lg.intercept_
        
    def predict(self, X):
        return self.ypredictor.predict(X.reshape(-1, 1))
    
    
class three_stage_logit(IVModel):
    
    def __init__(self, classifier=XGBClassifier(), regressor = LinearRegression()):
        self.X_hat = None
        self.g_p_hat = None
        self.classifier = classifier
        self.regressor = regressor
    
    def logit(self, p):
        return np.log( p / (1 - p) )
    
    def fit(self, X, Y, Z):
        self.classifier.fit(X.reshape(-1, 1), Y)
        E_Y_X = self.classifier.predict_proba(X.reshape(-1, 1))[:, 1]
        
        self.g_p_hat = self.logit(E_Y_X)
        
        lm_Z_X = LinearRegression()
        lm_Z_X.fit(Z.reshape(-1, 1), X)
        self.X_hat = lm_Z_X.predict(Z.reshape(-1, 1))
        lm_X_hat = LinearRegression()
        lm_X_hat.fit(self.X_hat.reshape(-1, 1), self.g_p_hat)
        self.coef_ = lm_X_hat.coef_
        self.intercept_ = lm_X_hat.intercept_