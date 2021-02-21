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

df = pd.read_csv("C:\\Users\\Trinh\\Desktop\\GWU\\Fall 2020\\Time Series\\Final Project\\AirQualityUCI.csv",
                  header=0, parse_dates=['Date']) #parse_dates=[['Date', 'Time']], index_col=['Date_Time'])
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
# Plot CO level over time
plt.figure()
plt.plot(time, df['CO(GT)'])
plt.xticks(rotation=45)
plt.gcf().autofmt_xdate()
plt.xlabel('Time (Hourly)')
plt.ylabel('Concentration')
plt.title('CO level over time')
plt.show()


lags = 50
# ACF plot
Cal_autocorr_with_plot_fn(df['CO(GT)'], lags, 'ACF plot for CO level')


# Correlation matrix
corr = df.corr()
ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200), square=True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.gcf().autofmt_xdate()
plt.title('Correlation matrix')
plt.show()

# Split data into 80% train and 20% test
y = df['CO(GT)']
y_train, y_test = train_test_split(y, shuffle=False, test_size=0.2)
x_train, x_test = train_test_split(time, shuffle=False, test_size=0.2)


# Check for stationary
from Help import ADF_Cal
print(ADF_Cal(df['CO(GT)']))


# *******************TIME SERIES DECOMPOSITION*************************
# STL decomposition
data = pd.Series(np.array(df['CO(GT)']), index=pd.date_range('2004-3-10', periods=len(df['CO(GT)']), freq='h'), name='STL Decomposition for CO Level')
STL = STL(data)
res = STL.fit()
STL_plot = res.plot()
plt.xlabel('Time (Hourly)')
plt.show()

R = res.resid
T = res.trend
S = res.seasonal
# Plot trend, seasonal, and remainder in one plot
plt.figure()
plt.plot(time, T, label='Trend')
plt.plot(time, S, label='Seasonality')
plt.plot(time, R, label = 'Remainder')
plt.legend()
plt.xlabel('Time (Hourly)')
plt.ylabel('Magnitude')
plt.title('Components from STL decomposition')
plt.show()

# Calculate detrended data and plot it with original data
y_detrended = S + R
plt.figure()
plt.plot(time, data, label='Original CO data')
plt.plot(time, y_detrended, label='Detrended data')
plt.legend()
plt.xlabel('Time (Hourly)')
plt.ylabel('Magnitude')
plt.title('Original and Detrended CO data')
plt.show()

# Find seasonally adjusted data
seasonal_adj = T + R

# Plot seasonally adjusted data with original data
plt.figure()
plt.plot(time, data, label = 'Original data')
plt.plot(time, seasonal_adj, label='Seasonal Adjusted data')
plt.legend()
plt.title('Original and Seasonal Adjusted CO Data')
plt.xlabel('Time (Hourly)')
plt.ylabel('CO level')
plt.show()

# Calculate strength of trend
Ft = 1 - np.var(R)/np.var(T+R)
print('The strength of trend for the CO time series is', Ft)

# Calculate strength of seasonality
Fs = 1 - np.var(R)/np.var(S+R)
print('The strength of seasonality for the CO time series is', Fs)