import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, mean_squared_error

sales_orders = pd.read_excel('/workspaces/hello/Data/Cleaned data for sales order, product master.xlsx',sheet_name=None)
sales_orders
joint_sales_orders=pd.read_excel('/workspaces/hello/Data/Cleaned data for sales order, product master.xlsx',sheet_name=4)
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
# Set 'Date' column as index
time_series_1.set_index('Month_Year', inplace=True)
# Drop 'Year' and 'Month' columns if you don't need them anymore
time_series_1.drop(['Year', 'Month'], axis=1, inplace=True)
# Reset the index to convert 'Month_Year' index into a column
time_series_1.reset_index(inplace=True)
# Rename the index column to 'Month_Year'
time_series_1.rename(columns={'Month_Year': 'Month_Year_Index'}, inplace=True)

# Define the start date and end date for the next 60 periods
start_date = '2024-04-01'
end_date = '2030-03-01'
# Generate the date range
date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
# Create a DataFrame for the next 60 periods
next_60_periods = pd.DataFrame({'Month_Year_Index': date_range,
                                'Quantity in Kg': [None] * len(date_range)})
# Concatenate with the original time_series_1
time_series_2 = pd.concat([time_series_1, next_60_periods], ignore_index=True)



# Extract the training data
train_data = time_series_2.iloc[14:32]

# Extract the training data for both models
train_data_linear = time_series_2.iloc[14:32]
train_data_log = time_series_2.iloc[14:32]

# Define the logarithmic function
def log_function(x, a, b):
    return a * np.log(x) + b

# Fit a logarithmic regression model using training data
x_train_log = np.arange(1, len(train_data_log) + 1)
y_train_log = train_data_log['Quantity in Kg']

popt_log, _ = curve_fit(log_function, x_train_log, y_train_log)

# Fit a linear regression model using training data
x_train_linear = np.arange(len(train_data_linear)).reshape(-1, 1)
y_train_linear = train_data_linear['Quantity in Kg']

model_linear = LinearRegression().fit(x_train_linear, y_train_linear)

# Predict future values for logarithmic regression
future_periods = 72  # Number of periods from 32 to 103
future_index = pd.date_range(start=train_data_log['Month_Year_Index'].iloc[-1], periods=future_periods + 1, freq='MS')[1:]
future_numeric_index_log = np.arange(len(train_data_log) + 1, len(train_data_log) + future_periods + 1)
future_forecast_log = log_function(future_numeric_index_log, *popt_log)

# Predict future values for linear regression
future_numeric_index_linear = np.arange(len(train_data_linear), len(train_data_linear) + future_periods).reshape(-1, 1)
future_forecast_linear = model_linear.predict(future_numeric_index_linear)

# Calculate Mean Absolute Deviation (MAD) for linear regression model
mad_linear = mean_absolute_error(y_train_linear, model_linear.predict(x_train_linear))

# Calculate Mean Absolute Deviation (MAD) for logarithmic regression model
mad_log = mean_absolute_error(y_train_log, log_function(x_train_log, *popt_log))

# Calculate Root Mean Squared Error (RMSE) for linear regression model
rmse_linear = mean_squared_error(y_train_linear, model_linear.predict(x_train_linear), squared=False)

# Calculate Root Mean Squared Error (RMSE) for logarithmic regression model
rmse_log = mean_squared_error(y_train_log, log_function(x_train_log, *popt_log), squared=False)

# Define a function to calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Calculate MAPE for linear regression model
mape_linear = mean_absolute_percentage_error(train_data_linear['Quantity in Kg'], future_forecast_linear[0:18])

# Calculate MAPE for logarithmic regression model
mape_log = mean_absolute_percentage_error(train_data_log['Quantity in Kg'], future_forecast_log[0:18])

# Plot the actual data, linear regression model, and logarithmic regression model
plt.figure(figsize=(12, 8))
plt.plot(time_series_2['Month_Year_Index'], time_series_2['Quantity in Kg'], label='Actual Data')
plt.plot(train_data_log['Month_Year_Index'], log_function(x_train_log, *popt_log), label='Logarithmic Regression (Training)', linestyle='--', color='red')
plt.plot(future_index, future_forecast_log, label='Logarithmic Regression Forecast', linestyle='--', color='green')
plt.plot(train_data_linear['Month_Year_Index'], model_linear.predict(x_train_linear), label='Linear Regression (Training)', linestyle='--', color='blue')
plt.plot(future_index, future_forecast_linear, label='Linear Regression Forecast', linestyle='--', color='orange')
plt.xlabel('Date')
plt.ylabel('Quantity in Kg')
plt.title('Linear and Logarithmic Regression Forecast Comparison')
plt.legend()
plt.grid(True)

# Add MAD, RMSE, and MAPE annotations
plt.annotate(f'Linear Regression MAD: {mad_linear:.2f}\nLinear Regression RMSE: {rmse_linear:.2f}\nLinear Regression MAPE: {mape_linear:.2f}%', 
             xy=(0.01, 0.85), xycoords='axes fraction', color='black')
plt.annotate(f'Logarithmic Regression MAD: {mad_log:.2f}\nLogarithmic Regression RMSE: {rmse_log:.2f}\nLogarithmic Regression MAPE: {mape_log:.2f}%', 
             xy=(0.3, 0.85), xycoords='axes fraction', color='black')
plt.savefig('/workspaces/hello/2_Pictures/Monthly/Combined_projection.png')