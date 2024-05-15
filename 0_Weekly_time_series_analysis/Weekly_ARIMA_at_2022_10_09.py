import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, mean_squared_error


joint_sales_orders=pd.read_excel('/workspaces/-round3---scmission2024---vanguard-s-code-/Data file/Cleaned data for sales order, product master.xlsx',sheet_name=4)
joint_sales_orders

# Create a copy of the joint_sales_orders DataFrame with only 'Date' and 'Quantity in Kg' columns
time_series_0 = joint_sales_orders[['Date', 'Quantity in Kg']].copy()

# Convert the 'Date' column to datetime format
time_series_0['Date'] = pd.to_datetime(time_series_0['Date'])

# Set the 'Date' column as the index
time_series_0.set_index('Date', inplace=True)

# Resample the data to weekly frequency, summing the 'Quantity in Kg' for each week
time_series_1 = time_series_0.resample('W').sum()

# Ensure the index is datetime
time_series_1.index = pd.to_datetime(time_series_1.index)

# Filter the training data for the specified date range
train_start_date = '2022-10-09'
train_end_date = '2024-03-31'
training_data = time_series_1[train_start_date:train_end_date].copy()

# Fit the ARIMA model
# (p, d, q) parameters need to be chosen. Here, we start with (1, 1, 1) as an example.
p, d, q = 2, 1, 2
arima_model = ARIMA(training_data['Quantity in Kg'], order=(p, d, q))
arima_fit = arima_model.fit()
print(arima_fit.summary())

# Get the fitted values
fitted_values = arima_fit.fittedvalues

# Forecast future values
forecast_start_date = pd.to_datetime('2024-04-01')
forecast_end_date = pd.to_datetime('2030-04-01')
num_weeks_forecast = (forecast_end_date - forecast_start_date).days // 7 + 1

forecast = arima_fit.get_forecast(steps=num_weeks_forecast)
forecast_index = pd.date_range(start=forecast_start_date, periods=num_weeks_forecast, freq='W')
forecast_values = forecast.predicted_mean

# Create DataFrame for the forecasted values
forecast_df = pd.DataFrame({'Forecasted Quantity in Kg': forecast_values}, index=forecast_index)

# Add fitted values to the training data
training_data['Fitted Quantity in Kg'] = fitted_values

# Calculate RMSE, MAD, and MAPE between the fitted values and the training data
rmse_train = np.sqrt(mean_squared_error(training_data['Quantity in Kg'], fitted_values))
mad_train = mean_absolute_error(training_data['Quantity in Kg'], fitted_values)
mape_train = np.mean(np.abs((training_data['Quantity in Kg'] - fitted_values) / training_data['Quantity in Kg'])) * 100

# Plot the original data, the fitted values, and the forecast
plt.figure(figsize=(12, 6))
plt.plot(time_series_1.index, time_series_1['Quantity in Kg'], label='Original Data')
plt.plot(training_data.index, training_data['Fitted Quantity in Kg'], label='Fitted ARIMA', linestyle='--')
plt.plot(forecast_df.index, forecast_df['Forecasted Quantity in Kg'], label='Forecast', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Quantity in Kg')
plt.legend()
plt.grid(True)
# Annotate the plot with the error metrics
error_text = f'RMSE: {rmse_train:.2f}\nMAD: {mad_train:.2f}\nMAPE: {mape_train:.2f}%'
plt.gca().text(0.30, 0.95, error_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
plt.savefig('Weekly_ARIMA_projection.png')

# Save forecasted data and fitted values to an Excel file
with pd.ExcelWriter('arima_fitted_and_forecasted_data.xlsx') as writer:
    training_data.to_excel(writer, sheet_name='Fitted Values')
    forecast_df.to_excel(writer, sheet_name='Forecasted Values')

# Resample to monthly frequency
fitted_monthly_df = training_data.resample('M').sum()
forecast_monthly_df = forecast_df.resample('M').sum()

# Save the monthly data to an Excel file
with pd.ExcelWriter('arima_fitted_and_forecasted_monthly_data.xlsx') as writer:
    fitted_monthly_df.to_excel(writer, sheet_name='Fitted Monthly Values')
    forecast_monthly_df.to_excel(writer, sheet_name='Forecasted Monthly Values')
