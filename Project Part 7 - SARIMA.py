from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from Help import GPAC
from Autocorrelation import Cal_autocorr_fn, Cal_autocorr_with_plot_fn, Cal_Q_fn
import statsmodels.api as sm
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import acf
from sklearn.model_selection import train_test_split
import copy
from scipy import signal
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("C:\\Users\\Trinh\\Desktop\\GWU\\Fall 2020\\Time Series\\Final Project\\AirQualityUCI.csv",
                 header=0, parse_dates=['Date'])
print(df.head())
df = df.iloc[:, :-2]
print(df.shape)
print(df.describe())
print(df.info())
print(df.isna().sum())
# There are 114 rows at the end without date and time, let's drop them
df = df[:-114]
print(df.shape)
# NA values are written as -200. Let's change it back to NaN
df = df.replace(-200, np.nan)
#print(df.head(15))

print(df.isna().sum())
# 8443 values in NMHC(GT) is NaN, so let's drop the column
df = df.drop(columns = ['NMHC(GT)'])
print(df.shape)

# Fill in nan with forward fill (replace nan with preceding value)
df = df.fillna(method='ffill')

time = pd.date_range('2004-03-10 18:00:00', periods=len(df.Time), freq='h')
# Split data into 80% train and 20% test
y = df['CO(GT)']
y_train, y_test = train_test_split(y, shuffle=False, test_size=0.2)
x_train, x_test = train_test_split(time, shuffle=False, test_size=0.2)

lags = 50


# Check ACF and PACF
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(y, lags=50, ax=ax1)
plt.title('ACF Plot')
plt.ylabel('Magnitude')

ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(y, lags=50, ax=ax2)
plt.title('PACF Plot')
plt.ylabel('Magnitude')
plt.xlabel('Lag')
plt.show()

# *********************************SARIMA(1,0,1)(1,0,1,24)*********************************
import statsmodels.api as sm
model = sm.tsa.statespace.SARIMAX(df['CO(GT)'], order=(1,0,1), seasonal_order=(1,0,1,24)).fit()

print(model.summary())

prediction = model.predict(start=0, end=len(y_train)-1)
# Plot prediction with original data
plt.figure()
plt.plot(x_train, y_train, label='Train Set')
plt.plot(x_train, prediction, label='Prediction of train set')
plt.legend()
plt.xticks(rotation=45)
plt.gcf().autofmt_xdate()
plt.xlabel('Time (Hourly)')
plt.ylabel('Concentration')
plt.title('CO Level - Train set and the prediction SARIMA(1,0,1)(1,0,1,24)')
plt.show()

res_e = y_train - prediction
Cal_autocorr_with_plot_fn(res_e, lags, 'ACF Plot of residuals - SARIMA(1,0,1)(1,0,1,24)')
print('MSE of residuals is', np.mean(res_e**2))
print('Variance of prediction error is', np.var(res_e))
print('Mean of prediction error is', np.mean(res_e))
print('Q of prediction error is', Cal_Q_fn(res_e, lags, len(y_train)))

# Q-value
Q = Cal_Q_fn(res_e,lags,len(y_train))

na=1
nb=1
# Chi-square test
from scipy.stats import chi2
DOF = lags - na - nb
alpha = 0.01
chi_critical = chi2.ppf(1-alpha, DOF)
if Q < chi_critical:
    print('The residual is white.')
else:
    print('The residual is NOT white.')

## Forecast
forecast = model.predict(start=len(y_train), end=len(y_train)+len(y_test)-1)
# Plot forecast
plt.figure()
plt.plot(x_test, y_test, label='Test Set')
plt.plot(x_test, forecast, label='Forecast of test set')
plt.legend()
plt.xticks(rotation=45)
plt.gcf().autofmt_xdate()
plt.xlabel('Time (Hourly)')
plt.ylabel('Concentration')
plt.title('CO Level - Test set and the forecast SARIMA(1,0,1)(1,0,1,24)')
plt.show()

f_e_arima = y_test - forecast
Cal_autocorr_with_plot_fn(f_e_arima, lags, 'ACF Plot of forecast error - SARIMA(1,0,1)(1,0,1,24)')
print('MSE of forecast is', np.mean(f_e_arima**2))
print('Variance of forecast error is', np.var(f_e_arima))
print('Mean of forecast error is', np.mean(f_e_arima))
print('Q of forecast error is', Cal_Q_fn(f_e_arima, lags, len(y_test)))
print('Mean of forecast', np.mean(forecast))
