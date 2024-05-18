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

# The final output
print(time_series_1)

#Detecting outliers 
plt.figure(figsize=(12, 8))
plt.boxplot(time_series_1['Quantity in Kg'])
plt.title('Boxplot of Time Series Data')
plt.savefig('/workspaces/-round3---scmission2024---vanguard-s-code-/Pictures/Weekly/Boxplot_of_Time_Series_Data.png')

# Calculate a moving average to smooth the data
time_series_1['Quantity in Kg (MA)'] = time_series_1['Quantity in Kg'].rolling(window=4).mean()

# Plotting the data
plt.figure(figsize=(14, 8))
plt.plot(time_series_1.index, time_series_1['Quantity in Kg'], marker='o', linestyle='-', label='Weekly Quantity')
plt.plot(time_series_1.index, time_series_1['Quantity in Kg (MA)'], marker='', linestyle='-', label='4-Week Moving Average', color='orange')

# Adding title and labels
plt.title('Weekly Sales Quantities')
plt.xlabel('Week_Year')
plt.ylabel('Quantity in Kg')

# Show fewer x-axis labels
plt.xticks(range(0, len(time_series_1.index), max(1, len(time_series_1.index) // 10)), rotation=45)

# Adding a legend
plt.legend()

# Adding grid for better readability
plt.grid(True)
plt.savefig('/workspaces/-round3---scmission2024---vanguard-s-code-/Pictures/Weekly/Time_series_plot.png')

#Creating the ACF and PACF from the begining
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
# Plot ACF
plot_acf(time_series_1['Quantity in Kg'], ax=ax[0])
ax[0].set_title('Autocorrelation Function (ACF)')
# Plot PACF
plot_pacf(time_series_1['Quantity in Kg'], ax=ax[1])
ax[1].set_title('Partial Autocorrelation Function (PACF)')
plt.tight_layout()
plt.savefig('/workspaces/-round3---scmission2024---vanguard-s-code-/Pictures/Weekly/ ACF_PACF_at_2021_31.png')
# Perform seasonal decomposition
decomposition = seasonal_decompose(time_series_1['Quantity in Kg'], model='mutiplicative', period=52)  # Assuming weekly data with yearly seasonality
# Plot the decomposed components
decomposition.plot()
plt.tight_layout()
plt.savefig('/workspaces/-round3---scmission2024---vanguard-s-code-/Pictures/Weekly/ Decomposition_at_2021_31.png')


#Creating the ACF and PACF from 41_2022
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
# Plot ACF
plot_acf(time_series_1['41_2022':]['Quantity in Kg'], ax=ax[0])
ax[0].set_title('Autocorrelation Function (ACF)')
# Plot PACF
plot_pacf(time_series_1['41_2022':]['Quantity in Kg'], ax=ax[1])
ax[1].set_title('Partial Autocorrelation Function (PACF)')
plt.tight_layout()
plt.savefig('/workspaces/-round3---scmission2024---vanguard-s-code-/Pictures/Weekly/ ACF_PACF_at_2022_41.png')

# Perform seasonal decomposition
decomposition = seasonal_decompose(time_series_1['41_2022':]['Quantity in Kg'], model='additive', period=52)  # Assuming weekly data with yearly seasonality
# Plot the decomposed components
decomposition.plot()
plt.tight_layout()
plt.savefig('/workspaces/-round3---scmission2024---vanguard-s-code-/Pictures/Weekly/ Decomposition_at_2022_41.png')

#Since the data exhibit a clear trend, it needs to be differenced
diff = time_series_1.copy().diff()
diff = diff.dropna()


plt.figure(figsize=(12, 8))
plt.plot(diff.index, diff['Quantity in Kg'], marker='o', linestyle='-')
plt.title('Differenced time series from 2021 to 2024')
plt.xlabel('Date')
plt.ylabel('Quantity in Kg')
plt.grid(True)
plt.savefig('/workspaces/-round3---scmission2024---vanguard-s-code-/Pictures/Weekly/Diff_time_series_plot.png')

#Creating the ACF of the differenced time series
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
# Plot ACF
plot_acf(diff['2022-10-09':]['Quantity in Kg'], ax=ax[0])
ax[0].set_title('Autocorrelation Function (ACF)')
# Plot PACF
plot_pacf(diff['2022-10-09':]['Quantity in Kg'], ax=ax[1])
ax[1].set_title('Partial Autocorrelation Function (PACF)')
plt.tight_layout()
plt.savefig('/workspaces/-round3---scmission2024---vanguard-s-code-/Pictures/Weekly/ACF_PACF_differenced_at_2022_10_09.png')

# Perform the ADF test
result = adfuller(diff['2022-10-09':]['Quantity in Kg'])

# Extract and print the results
adf_statistic = result[0]
p_value = result[1]
used_lag = result[2]
n_obs = result[3]
critical_values = result[4]

print(f'ADF Statistic: {adf_statistic}')
print(f'p-value: {p_value}')
print(f'Number of lags used: {used_lag}')
print(f'Number of observations used: {n_obs}')
print('Critical Values:')
for key, value in critical_values.items():
    print(f'   {key}: {value}')

# Interpret the p-value
if p_value < 0.05:
    print("Reject the null hypothesis (H0), the data is stationary.")
else:
    print("Fail to reject the null hypothesis (H0), the data is not stationary.")


