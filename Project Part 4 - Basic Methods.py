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

# *******************************AVERAGE METHOD******************************
print('**********************AVERAGE METHOD******************************')
from Help import average_1step_prediction, average_hstep_forecast
prediction, pr_e, pr_e_sq, pr_MSE = average_1step_prediction(y_train)
forecast, forecast_e, forecast_e_sq, forecast_MSE = average_hstep_forecast(y_train, y_test)

print('MSE of prediction errors:', pr_MSE)
print('Variance of prediction error:', np.var(pr_e))
print('Mean of prediction errors is', np.mean(pr_e))

# Plot the Average Method and Forecast
plt.figure()
plt.plot(x_train, y_train, label = 'Training set')
plt.plot(x_test, y_test, label = 'Test set')
plt.plot(x_test, forecast, label = 'h-step Forecast')
plt.legend()
plt.xlabel('Time (Hourly)')
plt.ylabel('Concentration')
plt.title('CO Level - Average Method and Forecast')
plt.show()

# ACF of prediction errors
Cal_autocorr_with_plot_fn(pr_e, lags, 'ACF Plot of prediction errors - Average method')
# ACF of forecast errors
Cal_autocorr_with_plot_fn(forecast_e, lags, 'ACF Plot of forecast errors - Average method')
# Q-value
print('Q value of residuals - average method:', Cal_Q_fn(pr_e, lags, len(y_train)))

print('MSE of forecast errors:', forecast_MSE)
print('Variance of forecast error:', np.var(forecast_e))
print('Mean of forecast errors is', np.mean(forecast_e))
print('Q of forecast error is', Cal_Q_fn(forecast_e, lags, len(y_test)))


# *******************************NAIVE METHOD******************************
print('**********************NAIVE METHOD******************************')
from Help import naive_1step_prediction, naive_hstep_forecast
naive_pr, naive_pr_e, naive_pr_e_sq, naive_pr_MSE = naive_1step_prediction(y_train)
naive_forecast, naive_f_e, naive_f_e_sq_, naive_f_MSE = naive_hstep_forecast(y_train, y_test)

print('MSE of prediction errors:', naive_pr_MSE)
print('Variance of prediction error:', np.var(naive_pr_e))
print('Mean of prediction errors is', np.mean(naive_pr_e))
print('Q value of naive method:', Cal_Q_fn(naive_pr_e, lags, len(y_train)))

# Plot the Naive Method and Forecast
plt.figure()
plt.plot(x_train, y_train, label = 'Training set')
plt.plot(x_test, y_test, label = 'Test set')
plt.plot(x_test, naive_forecast, label = 'h-step Forecast')
plt.legend()
plt.xlabel('Time (Hourly)')
plt.ylabel('Concentration')
plt.title('CO Level - Naive Method and Forecast')
plt.show()

# ACF of prediction errors
Cal_autocorr_with_plot_fn(naive_pr_e, lags, 'ACF Plot of prediction errors - Naive method')
# ACF of forecast errors
Cal_autocorr_with_plot_fn(naive_f_e, lags, 'ACF Plot of forecast errors - Naive method')

print('MSE of forecast errors:', naive_f_MSE)
print('Variance of forecast error:', np.var(naive_f_e))
print('Mean of forecast errors is', np.mean(naive_f_e))
print('Q of forecast error is', Cal_Q_fn(naive_f_e, lags, len(y_test)))


# *******************************DRIFT METHOD******************************
print('**********************DRIFT METHOD******************************')
from Help import drift_1step_prediction, drift_hstep_forecast
drift_pr, drift_pr_e, drift_pr_e_sq, drift_pr_MSE = drift_1step_prediction(y_train)
drift_forecast, drift_f_e, drift_f_e_sq_, drift_f_MSE = drift_hstep_forecast(y_train, y_test)

print('MSE of prediction errors:', drift_pr_MSE)
print('Variance of prediction error:', np.var(drift_pr_e))
print('Mean of prediction errors is', np.mean(drift_pr_e))
print('Q value of drift method:', Cal_Q_fn(drift_pr_e, lags, len(y_train)))

# Plot the Drift Method and Forecast
plt.figure()
plt.plot(x_train, y_train, label = 'Training set')
plt.plot(x_test, y_test, label = 'Test set')
plt.plot(x_test, drift_forecast, label = 'h-step Forecast')
plt.legend()
plt.xlabel('Time (Hourly)')
plt.ylabel('Concentration')
plt.title('CO Level - Drift Method and Forecast')
plt.show()

# ACF of prediction errors
Cal_autocorr_with_plot_fn(drift_pr_e, lags, 'ACF Plot of prediction errors - Drift method')
# ACF of forecast errors
Cal_autocorr_with_plot_fn(drift_f_e, lags, 'ACF Plot of forecast errors - Drift method')

print('MSE of forecast errors:', drift_f_MSE)
print('Variance of forecast error:', np.var(drift_f_e))
print('Mean of forecast errors is', np.mean(drift_f_e))
print('Q of forecast error is', Cal_Q_fn(drift_f_e, lags, len(y_test)))


# *******************************SES METHOD******************************
print('**********************SES METHOD******************************')
from Help import SES_1step_prediction, SES_hstep_forecast
SES_pr, SES_pr_e, SES_pr_e_sq, SES_pr_MSE = SES_1step_prediction(y_train, alpha=0.5)
SES_forecast, SES_f_e, SES_f_e_sq_, SES_f_MSE = SES_hstep_forecast(y_train, y_test, alpha=0.5)

print('MSE of prediction errors:', SES_pr_MSE)
print('Variance of prediction error:', np.var(SES_pr_e))
print('Mean of prediction errors is', np.mean(SES_pr_e))
print('Q value of SES method:', Cal_Q_fn(SES_pr_e, lags, len(y_train)))

# Plot the SES Method and Forecast
plt.figure()
plt.plot(x_train, y_train, label = 'Training set')
plt.plot(x_test, y_test, label = 'Test set')
plt.plot(x_test, SES_forecast, label = 'h-step Forecast')
plt.legend()
plt.xlabel('Time (Hourly)')
plt.ylabel('Concentration')
plt.title('CO Level - SES Method and Forecast with alpha=0.5')
plt.show()

# ACF of prediction errors
Cal_autocorr_with_plot_fn(SES_pr_e, lags, 'ACF Plot of prediction errors - SES method')
# ACF of forecast errors
Cal_autocorr_with_plot_fn(SES_f_e, lags, 'ACF Plot of forecast errors - SES method')

print('MSE of forecast errors:', SES_f_MSE)
print('Variance of forecast error:', np.var(SES_f_e))
print('Mean of forecast errors is', np.mean(SES_f_e))
print('Q of forecast error is', Cal_Q_fn(SES_f_e, lags, len(y_test)))
