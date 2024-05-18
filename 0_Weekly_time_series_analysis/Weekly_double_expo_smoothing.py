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
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

joint_sales_orders=pd.read_excel('Data/Cleaned data for sales order, product master.xlsx',sheet_name=4)
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

#Filter the data for the specified date range
start_date = '2022-10-09'
end_date = '2024-03-31'
filtered_data = time_series_1[start_date:end_date]

# Fit the Holt's Exponential Smoothing model
model = ExponentialSmoothing(filtered_data['Quantity in Kg'], trend='add', seasonal=None)
fit = model.fit(optimized=True)

# Get the optimal alpha and beta values
alpha = fit.params['smoothing_level']
beta = fit.params['smoothing_trend']
print(f"Optimal alpha: {alpha}")
print(f"Optimal beta: {beta}")

# Calculate the number of weeks to forecast
forecast_start_date = pd.to_datetime('2024-04-01')
forecast_end_date = pd.to_datetime('2030-04-01')
num_weeks_forecast = (forecast_end_date - forecast_start_date).days // 7 + 1

# Forecast future values
forecast = fit.forecast(steps=num_weeks_forecast)

# Create a DataFrame for the forecasted values
forecast_dates = pd.date_range(start=forecast_start_date, periods=num_weeks_forecast, freq='W')
forecast_df = pd.DataFrame({'Forecasted Quantity in Kg': forecast}, index=forecast_dates)

# Create a DataFrame for the fitted values
fitted_df = filtered_data.copy()
fitted_df['Fitted Quantity in Kg'] = fit.fittedvalues

# Save fitted values and forecasted data to an Excel file
with pd.ExcelWriter('Weekly_double_expo_smoothing.xlsx') as writer:
    fitted_df.to_excel(writer, sheet_name='Fitted Values')
    forecast_df.to_excel(writer, sheet_name='Forecasted Values')


# Calculate RMSE, MAD, and MAPE between the fitted values and the training data
fitted_values = fit.fittedvalues
rmse_train = np.sqrt(mean_squared_error(filtered_data['Quantity in Kg'], fitted_values))
mad_train = mean_absolute_error(filtered_data['Quantity in Kg'], fitted_values)
mape_train = np.mean(np.abs((filtered_data['Quantity in Kg'] - fitted_values) / filtered_data['Quantity in Kg'])) * 100

# Plot the original data, the fitted values, and the forecast
plt.figure(figsize=(12, 6))
plt.plot(time_series_1.index, time_series_1['Quantity in Kg'], label='Original Data')
plt.plot(filtered_data.index, fitted_values, label='Fitted Holt Exponential Smoothing', linestyle='--')
plt.plot(forecast_df.index, forecast_df['Forecasted Quantity in Kg'], label='Forecast', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Quantity in Kg')
plt.legend()
plt.grid(True)
plt.title('Double exponential smooting (Alpha = 0.19, Beta = 0.04)')

# Annotate the plot with the error metrics

error_text = f'RMSE: {rmse_train:.2f}\nMAD: {mad_train:.2f}\nMAPE: {mape_train:.2f}%'
plt.gca().text(0.68, 0.95, error_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
plt.savefig('/workspaces/hello/2_Pictures/Weekly/Double_expo_smoothing_at_2022_41.png.png')

# Resample to monthly frequency
fitted_monthly_df = filtered_data.resample('M').sum()
forecast_monthly_df = forecast_df.resample('M').sum()

# Save the monthly data to an Excel file
with pd.ExcelWriter('Monthly double exponential smoothing.xlsx') as writer:
    fitted_monthly_df.to_excel(writer, sheet_name='Fitted Monthly Values')
    forecast_monthly_df.to_excel(writer, sheet_name='Forecasted Monthly Values')