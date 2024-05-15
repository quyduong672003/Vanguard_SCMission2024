import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.model import Holt
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