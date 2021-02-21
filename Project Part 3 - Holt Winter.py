import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL
import statsmodels.tsa.holtwinters as ets
from Help import GPAC
from Autocorrelation import Cal_autocorr_fn, Cal_autocorr_with_plot_fn, Cal_Q_fn
import statsmodels.api as sm
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import acf
from sklearn.model_selection import train_test_split
import copy
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("C:\\Users\\Trinh\\Desktop\\GWU\\Fall 2020\\Time Series\\Final Project\\AirQualityUCI.csv",
                  header=0, parse_dates=['Date'])
print(df.head())
df = df.iloc[:, :-2]
print(df.shape)
print(df.describe())
print(df.isna().sum())
# There are 114 rows at the end without date and time, let's drop them
df = df[:-114]

# NA values are written as -200. Let's change it back to NaN
df = df.replace(-200, np.nan)
#print(df.head(15))

#df.isna().sum()
# 8557 values in NMHC(GT) is NaN, out of 9471 values, so let's drop the column
df = df.drop(columns = ['NMHC(GT)'])
print(df.shape)

# Fill in nan with forward fill (replace nan with preceding value)
df = df.fillna(method='ffill')

time = pd.date_range('2004-03-10 18:00:00', periods=len(df.Time), freq='h')

lags = 50
# Split the data into 80% train and 20% test
y = df['CO(GT)']
y_train, y_test = train_test_split(y, shuffle=False, test_size=0.2)
x_train, x_test = train_test_split(time, shuffle=False, test_size=0.2)

# ******************************Holt-Winter method*****************************

Holt_Winter = ets.ExponentialSmoothing(y_train, trend='add', seasonal='add', seasonal_periods=24).fit()
Holt_Winter_pr = Holt_Winter.fittedvalues
Holt_Winter_forecast = Holt_Winter.forecast(len(y_test))

# Plot the Holt-Winter Seasonal Method and Forecast
plt.figure()
plt.plot(x_train, y_train, label = 'Training set')
plt.plot(x_test, y_test, label = 'Test set')
plt.plot(x_test, Holt_Winter_forecast, label = 'h-step Forecast')
plt.legend()
plt.xlabel('Time (hourly)')
plt.ylabel('Concentration')
plt.title('CO level - Holt-Winter Seasonal Method and Forecast')
plt.show()

# Residual errors
Holt_Winter_pr_e = y_train - Holt_Winter_pr
print('Variance of prediction error - Holt Winter method is', np.var(Holt_Winter_pr_e))
Holt_Winter_MSE = np.mean(Holt_Winter_pr_e**2)
print('MSE of prediction error - Holt Winter method is', Holt_Winter_MSE)
print('Mean of prediction error is', np.mean(Holt_Winter_pr_e))
print('Q value of residual is', Cal_Q_fn(Holt_Winter_pr_e, lags, len(y_train)))

# ACF plot of residual errors
Cal_autocorr_with_plot_fn(Holt_Winter_pr_e, lags, 'ACF Plot for Holt Winter residuals')

# Forecast errors
Holt_Winter_f_e = y_test - Holt_Winter_forecast
print('Variance of forecast error - Holt Winter method is', np.var(Holt_Winter_f_e))
# MSE of the forecast error
Holt_Winter_f_MSE = np.mean(Holt_Winter_f_e**2)
print('MSE of forecast error - Holt Winter method is', Holt_Winter_f_MSE)
print('Q of forecast error is', Cal_Q_fn(Holt_Winter_f_e, lags, len(y_test)))
Cal_autocorr_with_plot_fn(Holt_Winter_f_e, lags, 'ACF Plot for Holt Winter forecast errors')