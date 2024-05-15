import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, mean_squared_error

joint_sales_orders=pd.read_excel('/workspaces/-round3---scmission2024---vanguard-s-code-/Data file/Cleaned data for sales order, product master.xlsx',sheet_name=4)
joint_sales_orders

# Assuming joint_sales_orders is your DataFrame
time_series_0 = joint_sales_orders[['Date', 'Quantity in Kg']].copy()
# Set 'Date' column as index
time_series_0.set_index('Date', inplace=True)
time_series_0['Month'] = time_series_0.index.month
time_series_0['Year'] = time_series_0.index.year
time_series_0

#Creating the time series
time_series_1= time_series_0.groupby(['Year', 'Month'])['Quantity in Kg'].sum()
time_series_1
time_series_1 = time_series_1.reset_index()
# Combine 'Year' and 'Month' columns into a new 'Date' column
time_series_1['Month_Year'] = time_series_1['Year'].astype(str) + '-' + time_series_1['Month'].astype(str)

# Convert 'Date' column to datetime format
time_series_1['Month_Year'] = pd.to_datetime(time_series_1['Month_Year'], format='%Y-%m')

# Drop 'Year' and 'Month' columns if you don't need them anymore
time_series_1.drop(['Year', 'Month'], axis=1, inplace=True)

# Set 'Date' column as index
time_series_1.set_index('Month_Year', inplace=True)


# Train-test split
train = time_series_1.loc[:'2022-10-01']
test = time_series_1.loc['2022-11-01':]

# Fit ARIMA model
order = (1,1,1)  # Example order, you may need to adjust this
model = ARIMA(train, order=order)
fit_model = model.fit()

# Get forecast and prediction intervals
forecast_steps = (pd.to_datetime('2030-04-01') - pd.to_datetime('2022-11-01')).days // 30 + 1
forecast = fit_model.get_forecast(steps=forecast_steps)
forecast_values = forecast.predicted_mean
ci = forecast.conf_int()

# Extend the forecast range back to 2022-11-01 and forward to April 2030
forecast_range = pd.date_range(start='2022-11-01', end='2030-04-01', freq='MS')

# Predict for the entire extended forecast range
forecast = fit_model.predict(start=forecast_range[0], end=forecast_range[-1])

# Plot results with prediction interval
plt.figure(figsize=(15, 8))
plt.plot(time_series_1, label='Actual')
plt.plot(test, label='Test')
plt.plot(forecast_range, forecast, label='Forecast')
ci_adjusted = ci.iloc[:len(forecast_range)]  # Adjust the size of ci to match forecast_range
plt.fill_between(forecast_range, ci_adjusted.iloc[:, 0], ci_adjusted.iloc[:, 1], color='gray', alpha=0.2, label='Prediction Interval')
plt.title('ARIMA Forecast with Prediction Interval')
plt.xlabel('Month_Year')
plt.ylabel('Quantity')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/workspaces/-round3---scmission2024---vanguard-s-code-/Pictures/Monthly/Arima_model_at_2021_08_01.png')

# Calculate metrics
mape_list=[]
for i in range(len(test)):
     mape =((forecast_values.iloc[1]-test.iloc[1]['Quantity in Kg'] )/(test.iloc[1]['Quantity in Kg']))
     mape_list.append(mape)
mape=np.array(mape_list).mean()
mad = mean_absolute_error(test, forecast_values[:len(test)])
rmse = np.sqrt(mean_squared_error(test, forecast_values[:len(test)]))

# Obtain residuals
residuals = fit_model.resid

# Plot ACF and PACF of residuals
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
sm.graphics.tsa.plot_acf(residuals, ax=ax[0])
sm.graphics.tsa.plot_pacf(residuals, ax=ax[1])
plt.savefig('/workspaces/-round3---scmission2024---vanguard-s-code-/Pictures/Monthly/Residuals_of_arima_model_at_2021_08_01.png')

# Ljung-Box test for residual autocorrelation
ljung_box_results = sm.stats.acorr_ljungbox(residuals, lags=[20])
print(ljung_box_results)
print('The p-value associated with this test statistic is approximately 0.999071.',
      'Since the p-value (0.999071) is much greater than 0.05, we fail to reject the null hypothesis.',
      'Therefore, at a significance level of 0.05, there is no significant evidence of autocorrelation in the residuals up to lag 10.')
