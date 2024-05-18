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
from statsmodels.tsa.seasonal import seasonal_decompose

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


#Detecting outliers 
plt.figure(figsize=(12, 8))
plt.boxplot(time_series_1['Quantity in Kg'])
plt.title('Boxplot of Time Series Data')
plt.savefig('/workspaces/-round3---scmission2024---vanguard-s-code-/Pictures/Boxplot_of_Time_Series_Data.png')

#Creating the ACF and PACF
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
# Plot ACF
plot_acf(time_series_1['2022-10-01':]['Quantity in Kg'], ax=ax[0])
ax[0].set_title('Autocorrelation Function (ACF)')
# Plot PACF
plot_pacf(time_series_1['Quantity in Kg'], ax=ax[1])
ax[1].set_title('Partial Autocorrelation Function (PACF)')
plt.tight_layout()
plt.savefig('/workspaces/-round3---scmission2024---vanguard-s-code-/Pictures/ACF_PACF.png')

#Since the data exhibit a clear trend, it needs to be differenced
diff = time_series_1.copy().diff()
diff = diff.dropna()
plt.figure(figsize=(12, 8))

plt.plot(diff.index, diff['Quantity in Kg'], marker='o', linestyle='-')
plt.title('Differenced time series from 2021 to 2024')
plt.xlabel('Date')
plt.ylabel('Quantity in Kg')
plt.grid(True)
plt.savefig('/workspaces/-round3---scmission2024---vanguard-s-code-/Pictures/Diff_time_series_plot.png')

#Creating the ACF of the differenced time series
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
# Plot ACF
plot_acf(diff['Quantity in Kg'], ax=ax[0])
ax[0].set_title('Autocorrelation Function (ACF)')
# Plot PACF
plot_pacf(diff['Quantity in Kg'], ax=ax[1])
ax[1].set_title('Partial Autocorrelation Function (PACF)')
plt.tight_layout()
plt.savefig('/workspaces/-round3---scmission2024---vanguard-s-code-/Pictures/ACF_PACF_differenced.png')


# Perform seasonal decomposition
decomposition = seasonal_decompose(time_series_1['Quantity in Kg'], model='additive', period=12)  # Assuming weekly data with yearly seasonality
# Plot the decomposed components
decomposition.plot()
plt.tight_layout()
plt.savefig('/workspaces/-round3---scmission2024---vanguard-s-code-/Pictures/Monthly/ Decomposition_at_2021_08_01.png')