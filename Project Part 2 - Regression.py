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

time = pd.date_range('2004-10-03 18:00:00', periods=len(df.Time), freq='h')

lags = 50

X = df[['PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)',
        'PT08.S5(O3)', 'T', 'RH', 'AH']]
Y = df['CO(GT)']
# Split the dataset into 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle=False, test_size=0.2)
time_train, time_test = train_test_split(time, shuffle=False, test_size=0.2)

X_train = sm.add_constant(X_train)
X_test = sm.add_constant((X_test))
model = sm.OLS(y_train, X_train).fit()

print('**************************Model 1 with all predictors and intercept**********************')
print(model.summary())

print('PT08.S3(NOx) has the highest p-value (0.852), so we are going to remove it from the model')
print('*****************************Model 2*****************************')

X_train = X_train[['PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'NO2(GT)', 'PT08.S4(NO2)',
        'PT08.S5(O3)', 'T', 'RH', 'AH']]
X_test = X_test[['PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'NO2(GT)', 'PT08.S4(NO2)',
        'PT08.S5(O3)', 'T', 'RH', 'AH']]
X_train = sm.add_constant(X_train)
X_test = sm.add_constant((X_test))
model2 = sm.OLS(y_train, X_train).fit()
forecast2 = model2.predict(X_test)

MSE_f_2 = np.square(np.subtract(y_test, forecast2)).mean()
print('MSE is', MSE_f_2)
print(model2.summary())

# Model 3 with 'AH' removed
print('************************************Model 3***************************')
X_train = X_train[['PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'NO2(GT)', 'PT08.S4(NO2)',
        'PT08.S5(O3)', 'T', 'RH']]
X_test = X_test[['PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'NO2(GT)', 'PT08.S4(NO2)',
        'PT08.S5(O3)', 'T', 'RH']]
X_train = sm.add_constant(X_train)
X_test = sm.add_constant((X_test))
model3 = sm.OLS(y_train, X_train).fit()
forecast3 = model3.predict(X_test)

f_e = y_test - forecast3
MSE_f_3 = np.square(np.subtract(y_test, forecast3)).mean()
print('MSE of forecast error is', MSE_f_3)
print('Variance of forecast error is', np.var(f_e))
print(model3.summary())

# Residual error
prediction = model3.predict(X_train)
# Plot original data and prediction
plt.figure()
plt.plot(time_train, y_train, label='Train Set')
plt.plot(time_train, prediction, label='Prediction of train set')
plt.legend()
plt.xticks(rotation=45)
plt.gcf().autofmt_xdate()
plt.xlabel('Time (Hourly)')
plt.ylabel('Concentration')
plt.title('CO Level - Train set and the prediction - Regression')
plt.show()

e = y_train - prediction
print('MSE of residuals is', np.mean(np.square(e)))
print('Variance of prediction error is', np.var(e))
print('Mean of prediction error is', np.mean(e))
print('Q value of prediction error is', Cal_Q_fn(e, lags, len(y_train)))
# ACF Plot
Cal_autocorr_with_plot_fn(e, lags, 'ACF plot of residual error')

# Plot forecast with test set
plt.figure()
plt.plot(time_test, y_test, label='Train Set')
plt.plot(time_test, forecast3, label='Prediction of train set')
plt.legend()
plt.xticks(rotation=45)
plt.gcf().autofmt_xdate()
plt.xlabel('Time (Hourly)')
plt.ylabel('Concentration')
plt.title('CO Level - Test set and the forecast - Regression')
plt.show()

# ACF Plot of forecast errors
Cal_autocorr_with_plot_fn(f_e, lags, 'ACF Plot of forecast errors')