import numpy as np
import pandas as pd
import geopandas as gpd

import scipy
import statsmodels.api as sm
from copy import deepcopy

from spglm.iwls import _compute_betas_gwr
from mgwr.search import golden_section

from smoother import ConstantTerm, LinearTerm, DistanceWeighting, SpatialWeightSmoother

class GASS:
    def __init__(self, y, *args, constant = True):
        self.y = y
        self.z = None
        self.args = args # Input model terms
        self.term_mapping = {} # Dictionary to store the mapping of each covariate in 'X' to its corresponding term
        self.constant = constant # Intercept
        self.num_constant_term = 0 # NO. of intercept, it should be one.
        self.num_linear_terms = 0 # NO. of linear terms
        self.initial_sigmas = [] 
        self.sigmas = [] 
        self.initial_X = None
        self.final_X = None
        self.fit_function = None
        self.AWCI_sigmas = None
        self.RBCI_sigmas = None
        self.CI_betas = None
        self.fitted_y = None
        self.residuals = None
        self.std_err = None
        self.tvals = None
        self.zvals = None
        self.pvals = None
        self.AIC = None
        self.log_likelihood = None
        self.Deviance = None
        self.R_squared = None  
        self.R_squared_CS = None
        self.R_squared_McFadden = None
        self.percent_deviance = None
        
        self.initialize()

    def initialize(self):
        initial_X_matrices = []
        current_col_index = 0 # Current column index in initial_X
        
        if self.constant == True:
            constant = ConstantTerm(self.y.shape[0])
            initial_X_matrices.append(constant.X)
            current_col_index = 1
            self.num_constant_term = 1
            self.term_mapping [0] = (type(constant).__name__, constant)  
        
        for arg_idx, arg in enumerate(self.args):
            num_columns = 0  # Number of columns this arg will add to initial_X

            if isinstance(arg, LinearTerm):
                initial_X_matrices.append(arg.X)
                num_columns = arg.X.shape[1]  
                self.num_linear_terms += num_columns 
                
            elif isinstance(arg, SpatialWeightSmoother):
                initial_X_matrices.append(arg.cal(-1))
                num_columns = arg.cal(arg.initial_sigma).shape[1]
                self.initial_sigmas.append(arg.initial_sigma)
                
            elif isinstance(arg, DistanceWeighting):
                initial_X_matrices.append(arg.cal(-1))
                num_columns = arg.cal(arg.initial_sigma).shape[1]
                self.initial_sigmas.append(arg.initial_sigma)
                
            else:
                raise ValueError(f"Unsupported term type: {type(arg)}")

            # Record the term mapping for the new columns
            for col in range(current_col_index, current_col_index + num_columns):
                self.term_mapping [col] = (type(arg).__name__, arg)   # storing index and type name

            current_col_index += num_columns  # update current column index

        # Concatenate terms
        self.initial_X = np.hstack(initial_X_matrices)

    def backfit(self, y, X, w, sigs, verbose = False, max_iter = 50, tol = 1e-8):
        n,k = X.shape
        betas = _compute_betas_gwr(y, X, w.reshape((-1, 1)))[0]
        XB = np.multiply(betas.T, X)
        yhat = np.dot(X, betas)
        err = y.reshape((-1, 1)) - yhat
        crits = []
        delta = 1e6
        tmp_sigs = sigs

        for n_iter in range(1, max_iter + 1):
            new_XB = np.zeros_like(X)
            params = np.zeros_like(betas)

            for j in range(k):

                temp_y = XB[:, j].reshape((-1, 1))
                temp_y = temp_y + err.reshape((-1, 1))
                temp_X = X[:, j].reshape((-1, 1))
                type_name, term_instance = self.term_mapping[j]  
            
                if type_name not in ['LinearTerm', 'ConstantTerm']:
                    gscr = lambda x: sm.OLS(y, np.hstack((X[:, np.arange(X.shape[1]) != j], term_instance.cal(x)))).fit().aic
                    sig = golden_section(term_instance.lower_bound, term_instance.upper_bound, 0.3879, gscr, 1e-2, 50, 50)[0]
                    tmp_sigs[j-self.num_linear_terms-self.num_constant_term] = sig
                    sv = term_instance.cal(sig) #new smoothed values
                    X[:, j] = sv.flatten()
                    temp_X = sv.flatten().reshape((-1,1))

                beta = _compute_betas_gwr(temp_y, temp_X, w.reshape((-1,1)))[0]
                yhat = np.dot(temp_X, beta)
                new_XB[:, j] = yhat.flatten()
                err = (temp_y - yhat).reshape((-1, 1))
                params[j, :] = beta[0][0]

            crit = np.sum((y-XB)**2)/n
            XB = new_XB

            crits.append(deepcopy(crit))
            delta = crit

            if verbose:
                print("Current iteration:", n_iter, ",SOC:", np.round(score, 8))
            if delta < tol:
                break
        
        return params, X, tmp_sigs

    def fit_Gaussian(self, input_y = None, verbose = False, max_iter = 50, crit_threshold = 1e-8):
        self.fit_function = self.fit_Gaussian
        X = self.initial_X.copy()
        y = self.y.copy()
        if input_y is not None:
            y = input_y.copy()
        sigmas = self.initial_sigmas.copy()
        
        s_0 = np.mean(y)
        eta = s_0.reshape((-1, 1))
        s_old = np.ones_like(X)
        score = 9999
        n_iter = 0
        
        while score > crit_threshold and n_iter < max_iter:
            w = np.ones(X.shape[0])
            z = y.reshape((-1, 1))
            betas, X, tmp_sigmas = self.backfit(z, X, w, sigmas, verbose = verbose, max_iter = max_iter, tol = crit_threshold) 
            sigmas = deepcopy(tmp_sigmas)

            s_new = np.multiply(betas.T, X)
            inner = np.sum((s_old - s_new)**2, axis=1)
            num = np.sum(w*inner)
            den = np.sum(w*np.sum((1 + s_old)**2, axis=1).reshape((-1, 1)))
            score = num / den
            eta = np.sum(s_new, axis=1).reshape((-1, 1))
            s_old = s_new
            
            n_iter += 1  # increment the iteration counter
            
        self.coefficients = betas
        self.sigmas = sigmas
        self.final_X = X
        self.z = z
        
        pass
    
    def inference_Gaussian(self): 
        yhat = np.dot(self.final_X, self.coefficients)
        self.fitted_y = yhat
        
        n = self.final_X.shape[0] # number of observaions  
        k = self.final_X.shape[1] # number of parameters

        # Calculate residuals
        residuals = self.y - yhat
        self.residuals = residuals

        # Calculate standard error of the estimate
        s2 = np.sum(residuals**2) / (self.final_X.shape[0] - self.final_X.shape[1])

        # Calculate standard error of coefficients
        var_beta = s2 * (np.linalg.inv(np.dot(self.final_X.T, self.final_X)).diagonal())
        se_beta = np.sqrt(var_beta)
        self.std_err = se_beta

        # Calculate confidence intervals
        critical_value = scipy.stats.t.ppf(1 - 0.05 / 2, df=self.final_X.shape[0]-self.final_X.shape[1])
        coefs_lower_bound = self.coefficients.reshape(-1,) - critical_value * se_beta.reshape(-1,)
        coefs_upper_bound = self.coefficients.reshape(-1,) + critical_value * se_beta.reshape(-1,)
        self.CI_betas = []
        for i in range(k):
            self.CI_betas.append((round(coefs_lower_bound[i],4), round(coefs_upper_bound[i],4)))
            
        # Calculate t values
        self.tvals = self.coefficients.reshape(-1,) / se_beta.reshape(-1,)

        # Calculate p values
        self.pvals = 2 * (1 - scipy.stats.t.cdf(np.abs(self.tvals), df=n-k))
        
        # Calculate R squared
        SST = np.sum((self.y - np.mean(self.y))**2)
        SSR = np.sum(residuals**2)  
        self.R_squared = 1 - (SSR / SST)
        
        # Calculate log likelihood
        log_likelihood = -n/2 * (1 + np.log(2*np.pi)) - n/2 * np.log(s2)
        self.log_likelihood = log_likelihood
        
        # Calculate AIC
        # Another way:  AIC = sm.OLS(self.y, self.final_X).fit().aic
        self.AIC = 2*k - 2*log_likelihood 

        pass
    
    def calculate_AWCI_sigmas(self, level = 0.95, interval = 0.01):
        
        self.AWCI_sigmas = []
        for tidx, tsig in enumerate(self.sigmas):
            
            tsig_idx = int(tidx + self.num_linear_terms + self.num_constant_term)
            tsig_term_instance = self.term_mapping[tsig_idx][1]
            
            # create an array of candidate sigmas
            tsig_b4 = np.arange(tsig_term_instance.lower_bound, tsig, interval)
            tsig_af = np.arange(tsig, tsig_term_instance.upper_bound, interval)
            tsig_candidates = np.hstack((tsig_b4, tsig_af)).flatten()
            
            tsig_aics = []
            for sig in tsig_candidates:
                aic = sm.OLS(self.z, np.hstack((self.final_X[:, np.arange(self.final_X.shape[1]) != tsig_idx], tsig_term_instance.cal(sig)))).fit().aic 
                tsig_aics.append((sig, aic))
                
            tsig_awdf = pd.DataFrame(tsig_aics, columns=['Sigma', 'AIC'])  

            minAIC = np.min(tsig_awdf.AIC)
            deltaAICs = tsig_awdf.AIC - minAIC
            awsum = np.sum(np.exp(-0.5 * deltaAICs))
            tsig_awdf = tsig_awdf.assign(AW = np.exp(-0.5 * deltaAICs)/awsum)
            tsig_awdf = tsig_awdf.sort_values(by = 'AW',ascending=False)
            tsig_awdf = tsig_awdf.assign(cumAW = tsig_awdf.AW.cumsum())

            index = len(tsig_awdf[tsig_awdf.cumAW < level]) + 1
            tsig_min = tsig_awdf.iloc[:index,:].Sigma.min()
            tsig_max = tsig_awdf.iloc[:index,:].Sigma.max()
            
            self.AWCI_sigmas.append((round(tsig_min, 4), round(tsig_max,4)))
            
        pass
    
    def calculate_RBCI_sigmas(self, level=0.95, max_iter = 1000, crit_threshold = 1e-8):
        
        if self.fit_function is None:
            raise ValueError("No fit function has been set.")

        fitted_y = self.fitted_y.copy()
        residuals = self.residuals.copy()
        self.RBCI_sigmas = []
        sigdicts = {}
        lower = (1 - level) * 100 / 2.0
        upper = 100 - lower

        sigdicts = {i: [] for i in range(len(self.sigmas))}

        for n_iter in range(max_iter):
            
            np.random.seed(n_iter)
            bootstrap_residuals = np.random.choice(residuals[:, 0], size=len(residuals), replace=True).reshape(-1, 1)
            bootstrap_y = fitted_y + bootstrap_residuals
            tgass = deepcopy(self)

            tgass.fit_function(input_y = bootstrap_y, crit_threshold = crit_threshold)
            
            for tidx, tsig in enumerate(tgass.sigmas):
                sigdicts[tidx].append(tsig)

        for siglist in sigdicts.values():
            sigdf = pd.DataFrame(siglist)
            sigdf.columns = ['Sigma']
            sigdf = sigdf.sort_values(by=['Sigma'])

            minSig = np.percentile(sigdf, lower)
            maxSig = np.percentile(sigdf, upper)

            self.RBCI_sigmas.append((round(minSig, 4), round(maxSig, 4)))