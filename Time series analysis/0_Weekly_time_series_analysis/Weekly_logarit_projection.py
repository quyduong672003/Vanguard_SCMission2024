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


joint_sales_orders=pd.read_excel('/workspaces/hello/Data/Cleaned data for sales order, product master.xlsx',sheet_name=4)
joint_sales_orders

# Create a copy of the joint_sales_orders DataFrame with only 'Date' and 'Quantity in Kg' columns
time_series_0 = joint_sales_orders[['Date', 'Quantity in Kg']].copy()

# Convert the 'Date' column to datetime format
time_series_0['Date'] = pd.to_datetime(time_series_0['Date'])

# Set the 'Date' column as the index
time_series_0.set_index('Date', inplace=True)

# Resample the data to weekly frequency, summing the 'Quantity in Kg' for each week
time_series_1 = time_series_0.resample('W').sum()

# 2022-10-09    71939.943053
# Ensure the index is datetime
time_series_1.index = pd.to_datetime(time_series_1.index)

# Filter the data for the specified date range
start_date = '2022-10-09'
end_date = '2024-03-31'
filtered_data = time_series_1[start_date:end_date]

# Define the log function
def log_function(x, a, b):
    return a * np.log(x) + b

# Create a time index (e.g., week number since start of the period)
filtered_data['Week_Number'] = np.arange(len(filtered_data)) + 1  # Week numbers starting from 1
x_data = filtered_data['Week_Number']
y_data = filtered_data['Quantity in Kg']

# Fit the log function to the data
params, _ = curve_fit(log_function, x_data, y_data)
a, b = params
print(f"Fitted parameters: a = {a}, b = {b}")

# Calculate the number of weeks to forecast
forecast_start_date = pd.to_datetime('2024-04-01')
forecast_end_date = pd.to_datetime('2030-04-01')
num_weeks_forecast = (forecast_end_date - forecast_start_date).days // 7 + 1

# Generate week numbers for the forecast period
last_week_number = x_data.iloc[-1]
forecast_week_numbers = np.arange(last_week_number + 1, last_week_number + num_weeks_forecast + 1)

# Use the fitted log function to forecast future values
forecasted_values = log_function(forecast_week_numbers, a, b)

# Create a DataFrame for the forecasted values
forecast_dates = pd.date_range(start=forecast_start_date, periods=num_weeks_forecast, freq='W')
forecast_df = pd.DataFrame({'Forecasted Quantity in Kg': forecasted_values}, index=forecast_dates)

# Calculate fitted values for the training data
fitted_values_train = log_function(x_data, a, b)
filtered_data['Fitted Quantity in Kg'] = fitted_values_train

# Save forecasted data and fitted values to an Excel file
with pd.ExcelWriter('fitted_and_forecasted_data.xlsx') as writer:
    filtered_data.to_excel(writer, sheet_name='Fitted Values')
    forecast_df.to_excel(writer, sheet_name='Forecasted Values')

# Calculate RMSE, MAD, and MAPE between the forecasted values and the training data
forecasted_values_train = log_function(x_data, a, b)
rmse_train = np.sqrt(mean_squared_error(y_data, forecasted_values_train))
mad_train = mean_absolute_error(y_data, forecasted_values_train)
mape_train = np.mean(np.abs((y_data - forecasted_values_train) / y_data)) * 100

print(f"RMSE between forecast and training data: {rmse_train:.2f}")
print(f"MAD between forecast and training data: {mad_train:.2f}")
print(f"MAPE between forecast and training data: {mape_train:.2f}%")

# Plot the original data, the fitted curve, and the forecast
plt.figure(figsize=(12, 6))
plt.plot(time_series_1.index, time_series_1['Quantity in Kg'], label='Original Data')
plt.plot(filtered_data.index, log_function(filtered_data['Week_Number'], a, b), label='Fitted Log Function', linestyle='--')
plt.plot(forecast_df.index, forecast_df['Forecasted Quantity in Kg'], label='Forecast', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Quantity in Kg')
plt.legend()
plt.grid(True)
plt.title('Logarit forecasting')
# Annotate the plot with the error metrics
error_text = f'RMSE: {rmse_train:.2f}\nMAD: {mad_train:.2f}\nMAPE: {mape_train:.2f}%'
plt.gca().text(0.30, 0.95, error_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

plt.savefig('/workspaces/hello/2_Pictures/Weekly/Logarit_projection_at_2022_41.png')

# Resample to monthly frequency
fitted_monthly_df = filtered_data.resample('M').sum()
forecast_monthly_df = forecast_df.resample('M').sum()

# Save the monthly data to an Excel file
with pd.ExcelWriter('/workspaces/hello/0_Weekly_time_series_analysis/Monthly logarit projection.xlsx') as writer:
    fitted_monthly_df.to_excel(writer, sheet_name='Fitted Monthly Values')
    forecast_monthly_df.to_excel(writer, sheet_name='Forecasted Monthly Values')