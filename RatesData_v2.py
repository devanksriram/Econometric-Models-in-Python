import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics import tsaplots
from arch.unitroot import ADF
import statsmodels.tsa.api as smt
from statsmodels.tsa.vector_ar import vecm
data = pd.read_excel(r'/Users/devanksriram/Documents/Time Series- Derek Bun/RatesData (1).xlsx', index_col=0)
data.index = pd.to_datetime(data.index)
data['D1_FF12']=data['FF12'].diff(1)
data['D1_GTII10']=data['GTII10'].diff(1)
data['D1_HYG_US']=data['HYG_US'].pct_change(1)
data=data[['FF12', 'D1_FF12', 'GTII10', 'D1_GTII10', 'HYG_US', 'D1_HYG_US']]
data=data.dropna()

#Plot of raw and differenced data
a = 3
b = 2
c = 1

fig = plt.figure(figsize=(10,7))
for i in data:
    plt.subplot(a, b, c)
    plt.title(i)
    plt.plot(data[i])
    c = c + 1

plt.tight_layout()
plt.show()

#ADF (include trend? Constant is default)
for i in data.columns:
    res = ADF(data[i], method='aic', max_lags=12)
    print()
    print(i)
    print(res.summary())

#ACF and PACF plots
lags=12
a=['FF12', 'GTII10', 'HYG_US']
D1a=['D1_FF12', 'D1_GTII10', 'D1_HYG_US']
b=[0,1,2]
c=[0,0,0]
d=[0,1,2]
e=[1,1,1]
axes=list(zip(a, b, c, d, e))
D1_axes=list(zip(D1a, b, c, d, e))

fig, ax = plt.subplots(3,2, figsize=(7.5,5), dpi=300, constrained_layout = True)
for a, b, c, d, e in axes:
        tsaplots.plot_acf(data[a], lags=lags, ax=ax[b, c])
        ax[b, c].set_title(a + ' ACF')
        tsaplots.plot_pacf(data[a],  lags=lags, ax=ax[d,e])
        ax[d, e].set_title(a + ' PACF')

fig, ax = plt.subplots(3,2, figsize=(7.5,5), dpi=300, constrained_layout = True)
for D1a, b, c, d, e in D1_axes:
        tsaplots.plot_acf(data[D1a], lags=lags, ax=ax[b, c])
        ax[b, c].set_title(D1a + ' ACF')
        tsaplots.plot_pacf(data[D1a],  lags=lags, ax=ax[d,e])
        ax[d, e].set_title(D1a + ' PACF')
plt.show()

#VAR model
model_VAR = smt.VAR(data[['D1_FF12', 'D1_GTII10', 'D1_HYG_US']])
res_VAR = model_VAR.fit(maxlags=2)
print(res_VAR.summary())

#VECM model
model_VECM = vecm.VECM(data[['FF12', 'GTII10', 'HYG_US']], k_ar_diff=2,coint_rank=1, deterministic='cili')
res_VECM = model_VECM.fit()
print(res_VECM.summary())

#fig, ax = plt.subplots(1, figsize=(7.5,5), dpi=300, constrained_layout = True)
#ax.plot(res_VECM.resid)
#dataVECMResid = pd.DataFrame(res_VECM.resid)
#ADF_VECM_Residuals = adfuller(res_VECM.resid, autolag='AIC', maxlag=12)
#print(ADF_VECM_Residuals)
