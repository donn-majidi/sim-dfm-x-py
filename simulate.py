import scipy
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.multivariate.pca import PCA
from statsmodels.multivariate.multivariate_ols import _MultivariateOLS
from statsmodels.tsa.stattools import adfuller
from src.ABC_crit import ABC_crit

plt.rc('figure', figsize=(18,10))
sns.set_style('darkgrid')

seed = 1776
rng = np.random.default_rng(seed)

#### Simulation parameters
r = int(input('Number of common factors: '))
n = int(input('Number of observable variables: '))
T = int(input('Number of observations: '))

#### Simulate the Matrix of loadings
Lambda = rng.uniform(-8,8,(n,r))

#### Simulate a static exact factor model
F_T = rng.multivariate_normal(np.zeros(r), np.identity(r), T)
Xi_T = rng.multivariate_normal(np.zeros(n), np.identity(n), T)

X_T = F_T @ Lambda.T + Xi_T

Fdf = pd.DataFrame(F_T)
Xdf = pd.DataFrame(X_T)

## quick check for stationarity
nonstat_series = []
for c in Xdf.columns:
    ur_res = adfuller(Xdf[c])
    if ur_res[1] > 0.05:
        nonstat_series.append(c)
        
kmax = 8*int(min(n,T)/100)**0.25 if int(min(n,T)/100)**0.25 > 0 else 8*1

#### Estimate the number of factors from the ABC criterion
fig, ax = plt.subplots()
rhat1, rhat2, ax = ABC_crit(Xdf, kmax=kmax, ax=ax, demean=True, seed=seed)

model_pca = PCA(Xdf, standardize=True, demean=True, missing='fill-em')
model_pca.ic
model_pca.plot_scree(log_scale=True)

F_hat = model_pca.factors.iloc[:,:rhat2]

F_hat = (F_hat - F_hat.mean())/F_hat.std()
Fdf = (Fdf - Fdf.mean())/Fdf.std()

fig, ax = plt.subplots(3,2)
ax[0,0].plot(Fdf[0], color='blue', label='Factor #1')
ax[0,0].legend(loc='upper right')
ax[0,1].plot(F_hat['comp_00'], color='firebrick', label='PC #1')
ax[0,1].legend(loc='upper right')

ax[1,0].plot(Fdf[1], color='blue', label='Factor #2')
ax[1,0].legend(loc='upper right')
ax[1,1].plot(F_hat['comp_01'], color='firebrick', label='PC #2')
ax[1,1].legend(loc='upper right')

ax[2,0].plot(Fdf[2], color='blue', label='Factor #3')
ax[2,0].legend(loc='upper right')
ax[2,1].plot(F_hat['comp_02'], color='firebrick', label='PC #3')
ax[2,1].legend(loc='upper right')
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(3,1, linewidth=1.5)
ax[0].plot(Fdf[0], color='blue', label='Factor #1')
ax[0].plot(F_hat['comp_01'], color='firebrick', label='PC #2')
ax[0].legend(loc='upper right')

ax[1].plot(Fdf[1], color='blue', label='Factor #1')
ax[1].plot(F_hat['comp_02'], color='firebrick', label='PC #3')
ax[1].legend(loc='upper right')

ax[2].plot(Fdf[2], color='blue', label='Factor #1')
ax[2].plot(F_hat['comp_00'], color='firebrick', label='PC #1')
ax[2].legend(loc='upper right')
plt.suptitle('Factors vs. PCs')
fig.tight_layout()
plt.show()

### Compare the span of the PC estimated matrix of loadings with the original matrix
Lambda_hat = model_pca.eigenvecs.iloc[:,:rhat2].to_numpy()
ang_sep = scipy.linalg.subspace_angles(Lambda, Lambda_hat)


### Obtain the OLS estimate of the matrix of loadings by the linear projection of the
### observable variables onto the range space of the common factors
model_ols = sm.OLS(X_T, F_hat)
results_ols = model_ols.fit(cov_type="nonrobust")

Lambda_tilde = results_ols.params.transpose().to_numpy()
ang_sep_ols = scipy.linalg.subspace_angles(Lambda, Lambda_tilde)



