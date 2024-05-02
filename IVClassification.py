from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import seaborn as sns
import numpy as np
from scipy.special import expit
from scipy.special import logit
from scipy.optimize import root 
from sklearn.preprocessing import StandardScaler

class sim_data:
    def __init__(self, n, sigma_z=1, sigma_u=1, sigma_e=0.01, 
                 alpha=1, beta=1, gamma=1, eta = 1):
        
        super().__init__()
        self.Z = np.random.normal(0, sigma_z, n)
        self.U = np.random.normal(0, sigma_u, n)
        epsilon = np.random.normal(0, sigma_e, n)
        
        self.X = alpha * self.Z + gamma * self.U + epsilon
        
        self.g_p = beta * self.X + eta * self.U
        self.p = expit(self.g_p)  # Replaced custom logistic function with expit
        self.Y = np.random.binomial(1, self.p)
        
    def plot_Z_X(self):
        sns.scatterplot(x=self.Z, y=self.X)
        
    def plot_U_X(self):
        sns.scatterplot(x=self.U, y=self.X)
        
    def plot_p(self):
        sns.histplot(self.p)
        
    def set_Z(self, Z):
        self.Z = Z
        
class sim_scaled_data:
    def __init__(self, n, sigma_z=1, sigma_u=1, sigma_e=0.01, 
                 alpha=1, beta=1, gamma=1, eta = 1):
        super().__init__()
        self.Z = np.random.normal(0, sigma_z, n)
        self.U = np.random.normal(0, sigma_u, n)
        epsilon = np.random.normal(0, sigma_e, n)
        
        self.beta = ((2 * beta)/(beta + eta))
        self.eta = ((2 * eta)/(beta + eta))
        self.alpha = alpha / (gamma + alpha)
        self.gamma = gamma / (gamma + alpha)
        
        self.X = self.alpha * self.Z + self.gamma * self.U + epsilon 
        # (alpha / (gamma + alpha)) * self.Z + \
        #             (gamma / (gamma + alpha)) * self.U + epsilon
        self.g_p = self.beta * self.X + self.eta * self.U
        
        # self.g_p = ((2 * beta)/(beta + eta)) * self.X + \
        #                 ((2 * eta)/(beta + eta)) * self.U
        
        self.p = expit(self.g_p)  # Replaced custom logistic function with expit
        self.Y = np.random.binomial(1, self.p)
        
    def plot_Z_X(self):
        sns.scatterplot(x=self.Z, y=self.X)
        
    def plot_U_X(self):
        sns.scatterplot(x=self.U, y=self.X)
        
    def plot_p(self):
        sns.histplot(self.p)
        
    def set_Z(self, Z):
        self.Z = Z


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
        # Modified the data type of coef_ to be a scalar
        self.coef_ = lg.coef_[0][0]
        self.intercept_ = lg.intercept_
        
    def predict(self, X):
        return self.ypredictor.predict(X.reshape(-1, 1))
    
class Multivar_TSLS():
    """
    self.coef is ordered coefficient of X, V, W
    """
    def __init__(self):
        pass
    
    def fit(self, X, Y, Z, V, W=None):
        if W == None:
            Z_X_lm = LinearRegression().fit(Z, X)
            X_hat = Z_X_lm.predict(Z)
            
            adj_features = np.concatenate([X_hat, V], axis=1)
            lg = LogisticRegression().fit(adj_features, np.array(Y).ravel())
            
            self.coef = lg.coef_
            self.intercept = lg.intercept_
            self.causal_effect = lg.coef_[0][0]
        else:
            X_parents = np.concatenate([Z, W], axis=1)
            Z_X_lm = LinearRegression().fit(X_parents, X)
            X_hat = Z_X_lm.predict(Z)
            
            adj_features = np.concatenate([X_hat, V, W], axis=1)
            lg = LogisticRegression().fit(adj_features, np.array(Y).ravel())
            
            self.coef = lg.coef_
            self.intercept = lg.intercept_
            self.causal_effect = lg.coef_[0][0]

            
class residual_logit(IVModel):
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self, X, Y, Z):
        lm_Z_X = LinearRegression()
        lm_Z_X.fit(Z.reshape(-1, 1), X)
        self.lm_Z_X = lm_Z_X
        X_hat = lm_Z_X.predict(Z.reshape(-1, 1))
        U_hat = X - X_hat
        X_tilde = np.concatenate([X.reshape(-1, 1), 
                                  U_hat.reshape(-1, 1)], axis=1)
        lg = LogisticRegression()
        lg.fit(X_tilde, Y)
        self.ypredictor = lg
        self.coef_ = lg.coef_
        self.beta = lg.coef_[0][0]
        self.intercept_ = lg.intercept_
        
    def predict(self, X):
        residuals = X - self.lm_Z_X.predict(X.reshape(-1, 1))
        X_tilde = np.concatenate([X.reshape(-1, 1), 
                                  residuals.reshape(-1, 1)], axis=1)
        return self.ypredictor.predict(X_tilde)
        
class ResIV():
    def __init__(self) -> None:
        pass

    def fit(self, X, Y, Z, V, W=None):
        if W == None:
            Z_X_lm = LinearRegression().fit(Z, X)
            X_hat = Z_X_lm.predict(Z)
            U_hat = X - X_hat
            adj_features = np.concatenate([X_hat, U_hat, V], axis=1)
            
            lg = LogisticRegression().fit(adj_features, np.array(Y).ravel())
            self.coef = lg.coef_
            self.intercept = lg.intercept_
            self.causal_effect = lg.coef_[0][0]
        else:
            X_parents = np.concatenate([Z, W], axis=1)
            Z_X_lm = LinearRegression().fit(X_parents, X)
            X_hat = Z_X_lm.predict(Z)
            U_hat = X - X_hat
            adj_features = np.concatenate([X_hat, U_hat, V, W], axis=1)
            
            lg = LogisticRegression().fit(adj_features, np.array(Y).ravel())
            self.coef = lg.coef_
            self.intercept = lg.intercept_
            self.causal_effect = lg.coef_[0][0]
            

class three_stage_logit(IVModel):
    
    def __init__(self, classifier=XGBClassifier(), regressor = LinearRegression()):
        self.X_hat = None
        self.g_p_hat = None
        self.classifier = classifier
        self.regressor = regressor
    
    def fit(self, X, Y, Z):
        self.classifier.fit(X.reshape(-1, 1), Y)
        E_Y_X = self.classifier.predict_proba(X.reshape(-1, 1))[:, 1]
        
        self.g_p_hat = logit(E_Y_X)  # Replaced custom logistic function with np.log
        
        lm_Z_X = LinearRegression()
        lm_Z_X.fit(Z.reshape(-1, 1), X)
        self.X_hat = lm_Z_X.predict(Z.reshape(-1, 1))
        lm_X_hat = LinearRegression()
        lm_X_hat.fit(self.X_hat.reshape(-1, 1), self.g_p_hat)
        self.coef_ = lm_X_hat.coef_[0]
        self.intercept_ = lm_X_hat.intercept_

class GMM_logit(IVModel):
    
    def __init__(self) -> None:
        super().__init__()
        self.X_hat = None
        self.scaler = StandardScaler()

    def fit(self, X, Y, Z):
        
        # Z = self.scaler.fit_transform(Z.reshape(-1, 1))
        # X = self.scaler.fit_transform(X.reshape(-1, 1))

        lm_Z_X = LinearRegression()
        lm_Z_X.fit(Z.reshape(-1, 1), X)
        self.X_hat = lm_Z_X.predict(Z.reshape(-1, 1))
        
        def eq(beta):
            residuals = Y - expit(beta[0] + beta[1] * X)
            f = [np.mean(residuals), 
                 np.mean(residuals * self.X_hat)]
            
            #direvative of logistic function
            dls = expit(beta[0]+ beta[1]*X) * (1 - expit(beta[0]+ beta[1]*X))
            
            #Jacobian of f
            J = [[-np.mean(dls), -np.mean(dls * X)],
                 [-np.mean(dls * self.X_hat), -np.mean(dls * self.X_hat * X)]]
            
            return f, J
        
        beta = root(eq, [0, 0], jac=True).x
        self.coef_ = beta[1]
        self.intercept_ = beta[0]
        return beta[1]
        
class Multivar_GMM():
    def __init__(self):
        pass
    
    def fit(self, X, Y, Z, V, W=None):
        X = np.array(X).reshape(-1, 1)
        Y = np.array(Y).reshape(-1, 1)
        V = np.array(V)
        
        if W == None:
            Z_X_lm = LinearRegression().fit(Z, X)
            X_hat = Z_X_lm.predict(Z).reshape(-1, 1)
            residuals = X - X_hat
            features = np.concatenate([np.ones(len(X)).reshape(-1, 1), X, V], axis=1)
            p = features.shape[1]
            
            def eq(beta):
                
                beta = beta.reshape(-1, 1)
                # beta should be a p*1 vector
                # print('Beta:', beta.shape)
                # print('Features:', features.shape)
                
                y_hat = expit(features @ beta)
                
                # print('y_hat:', y_hat.shape)
                # print('Y:', Y.shape)
                
                residuals = Y - y_hat
                
                # print('Residuals:', residuals.shape)
                # print('X_hat:', X_hat.shape)
                
                f_1 = np.array([np.mean(residuals), 
                              np.mean(residuals * X_hat)]).reshape(-1, 1)
                
                # print('V:', V.shape)
                # print(np.mean(residuals * V, axis=0))

                f_2 = np.mean(residuals * V, axis=0).reshape(-1, 1)
         
                f = np.concatenate((f_1, f_2)).reshape(-1)
                
                # #direvative of logistic function
                # dls = expit(beta[0]+ beta[1]*X) * (1 - expit(beta[0]+ beta[1]*X))
                
                # #Jacobian of f
                # J = [[-np.mean(dls), -np.mean(dls * X)],
                #      [-np.mean(dls * self.X_hat), -np.mean(dls * self.X_hat * X)]]
                
                return f
            
            beta = root(eq, np.random.normal(size=p).reshape(-1, 1)).x
            self.coef = beta
            self.intercept = beta[0]
            self.causal_effect = beta[1]
        else:
            raise NotImplementedError