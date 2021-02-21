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
from scipy.stats import chi2
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

# Estimate the order of ARMA
ry = Cal_autocorr_fn(df['CO(GT)'], lags)
acf_list = list(ry[::-1]) + list(ry[1:])

GPAC(acf_list, lags, 8, 8)
# Possible order set: (na, nb) = (1,1), (7,0)

mu_factor = 10
lags = 20
delta = 1e-6
epsilon = 0.001
dt = 1
mu = 0.01
mu_max = 1e10

na = 1
nb = 1
n = na + nb

theta = [0]*(n)

# Levenberg Marquardt Algorithm
# Step 1
def step1(data, theta, na, nb):
    max_order = max(na, nb)
    num = [0]*(max_order+1)
    den = [0]*(max_order+1)
    for i in range(na+1):
        if i==0:
            den[i] = 1
        else:
            den[i] = theta[i-1]

    for i in range(nb+1):
        if i==0:
            num[i] = 1
        else:
            num[i] = theta[na+i-1]

    system1 = (den, num, 1)
    _, e = signal.dlsim(system1, data)
    SSE = np.transpose(e).dot(e)

    X = []
    for i in range(n):
        theta_update = copy.copy(theta)
        theta_update[i] = theta_update[i] + delta
        for i in range(na + 1):
            if i == 0:
                den[i] = 1
            else:
                den[i] = theta_update[i - 1]

        for i in range(nb + 1):
            if i == 0:
                num[i] = 1
            else:
                num[i] = theta_update[na + i - 1]

        system = (den, num, 1)
        _, e_new = signal.dlsim(system, data)

        xi = (e - e_new) / delta
        X.append(xi)

    X = np.array(X)
    X = np.array([X[i].flatten() for i in range(len(X))]).T
    A = np.transpose(X).dot(X)
    g = np.transpose(X).dot(e)

    return SSE, X, A, g

# Step 2
def step2(data, A, g, theta, mu):
    I = np.identity(n)
    delta_theta = np.linalg.inv(A + mu*I).dot(g)
    delta_theta = np.reshape(delta_theta, (n,))
    theta_new = theta + delta_theta
    theta_new = theta_new.tolist()
    SSE_new, X_new, A_new, g_new = step1(data, theta_new, na, nb)

    if np.isinf(SSE_new):
        SSE_new = SSE
    if np.isnan(SSE_new):
        SSE_new = SSE*(10**8)

    return delta_theta, theta_new, SSE_new

# Compute steps 1 and 2
SSE, X, A, g = step1(y_train, theta, na, nb)
delta_theta, theta_new, SSE_new = step2(y_train, A, g, theta, mu)

# Step 3

max_iter = 50
i = 0
iteration_list = []
SSE_list = []

iteration_list.append(i) # initial iteration 0
SSE_list.append(SSE_new) # SSE at initial iteration 0

while i < max_iter:
    if SSE_new < SSE:
        if np.linalg.norm(delta_theta) < epsilon:
            theta = theta_new
            variance = SSE_new/(len(y_train)-n)
            covariance = variance*(np.linalg.inv(A))
            print('Estimated parameters:', theta)
            print('Variance:', variance)
            print('Covariance:', covariance)
            break
        else:
            theta = theta_new

    while SSE_new >= SSE:
        mu = mu*10
        delta_theta, theta_new, SSE_new = step2(y_train, A, g, theta, mu)
        if mu > mu_max:
            print(SSE_new)
            print('mu > mu_max. Check your code.')
            break


    i += 1
    iteration_list.append(i)
    if i > max_iter:
        print('i > 50. Check your code.')
        break

    theta = theta_new
    mu = mu / 10
    SSE, X, A, g = step1(y_train, theta, na, nb)
    delta_theta, theta_new, SSE_new = step2(y_train, A, g, theta, mu)
    SSE_list.extend(SSE_new)

# Plot SSE with iterations
plt.figure()
plt.plot(iteration_list, SSE_list)
plt.xlabel('Number of Iterations')
plt.ylabel('SSE')
plt.title('SSE vs number of iterations')
plt.show()

# Confidence interval
for i in range(n):
    low_limit = theta[i] - 2*np.sqrt(covariance[i][i])
    upper_limit = theta[i] + 2*np.sqrt(covariance[i][i])
    print('Confidence interval:', low_limit, ' < parameter {} < '.format(i+1), upper_limit)

# 1-step prediction
# y_hat(t+1|t) = 0.938y(t) +0.27y(t) - 0.27*yhat(t|t-1)
# so if t = 0, yhat(1|0) = 0.938*y(0) +0.27*y(0) - 0.27*yhat(0|-1) = 0.938*y(0) +0.27*y(0)
# if t=1, yhat(2|1) = 0.938*y(1) +0.27*y(1) - 0.27*yhat(1|0)
# and so on for the rest

prediction = []
yhat1 = -theta[0]*y_train[0] + theta[1] * y_train[0]
prediction.append(yhat1)
for i in range(1,len(y_train)-1):
    yhat = -theta[0]*y_train[i] + theta[1]*y_train[i] - theta[1]*prediction[i-1]
    prediction.append(yhat)

plt.figure()
plt.plot(x_train, y_train, label='Train Set')
plt.plot(x_train[1:], prediction, label='Prediction of train set')
plt.legend()
plt.xticks(rotation=45)
plt.gcf().autofmt_xdate()
plt.xlabel('Time (Hourly)')
plt.ylabel('Magnitude')
plt.title('Train set and the prediction')
plt.show()

e = y_train[1:] - prediction
Cal_autocorr_with_plot_fn(e, lags, 'ACF of residuals')
print('MSE of residuals is', np.mean(e**2))
print('Variance of prediction error is', np.var(e))
print('Mean of prediction error is', np.mean(e))
# Q-value
Q = Cal_Q_fn(e,lags,len(y_train))
print('Q value of residual is', Q)
# Chi-square test
DOF = lags - na - nb
alpha = 0.01
chi_critical = chi2.ppf(1-alpha, DOF)
if Q < chi_critical:
    print('The residual is white.')
else:
    print('The residual is NOT white.')

# Zero/Pole cancellation
roots_zeros = np.roots([1,theta[1]])
roots_poles = np.roots([1,theta[0]])
print('Roots of numerator:', roots_zeros)
print('Roots of denominator:', roots_poles)

# h-step forecast
forecast = []
yhat1 = -theta[0]*y_train.iloc[-1] + theta[1]*y_train.iloc[-1] - theta[1]*prediction[-1]
forecast.append(yhat1)
for i in range(len(y_test)-1):
    yhat = -theta[0]*forecast[i]
    forecast.append(yhat)

forecast_e = y_test - forecast
print('MSE of forecast error is', np.mean(forecast_e**2))
print('Variance of forecast error is', np.var(forecast_e))
print('Mean of forecast error is', np.mean(forecast_e))

plt.figure()
plt.plot(x_test, y_test, label='Test Set')
plt.plot(x_test, forecast, label='Forecast of test set')
plt.legend()
plt.xticks(rotation=45)
plt.gcf().autofmt_xdate()
plt.xlabel('Time (Hourly)')
plt.ylabel('Magnitude')
plt.title('Test set and its forecast')
plt.show()

Cal_autocorr_with_plot_fn(forecast_e, lags, 'ACF of forecast error')

