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



# Load the Excel file
file_path = '/workspaces/hello/Data/Cleaned data for sales order, product master.xlsx'
excel_data = pd.read_excel('/workspaces/hello/Data/Cleaned data for sales order, product master.xlsx',sheet_name=None)

# Load the sheets into dataframes
product_master_df = pd.read_excel(file_path, sheet_name='Cleaned product master')
sales_order_df = pd.read_excel(file_path, sheet_name='Cleaned sales order')

# Filter sales data for Thailand
thailand_sales_df = sales_order_df[sales_order_df['Sales in Country'] == 'Thailand']

# Define the product IDs for consumer daily liquid
consumer_daily_liquid_ids = ['310421', '310422', '310423', '310424', '310425', '310426']

# Filter sales data for the specified product IDs
filtered_sales_df = thailand_sales_df[thailand_sales_df['Product ID'].astype(str).isin(consumer_daily_liquid_ids)]

# Aggregate the data by month
filtered_sales_df['Date'] = pd.to_datetime(filtered_sales_df['Date'])
filtered_sales_df['Month'] = filtered_sales_df['Date'].dt.to_period('M')
monthly_sales = filtered_sales_df.groupby('Month')['Quantity in Kg'].sum()

# Convert PeriodIndex to DateTimeIndex for plotting
monthly_sales.index = monthly_sales.index.to_timestamp()

# Plot the time series of monthly sales
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales.index, monthly_sales, marker='o', linestyle='-')
plt.title('Monthly Sales of Consumer Daily Liquid Products in Thailand')
plt.xlabel('Month')
plt.ylabel('Sales Quantity (Kg)')
plt.grid(True)
plt.savefig('/workspaces/hello/Time series analysis/extra1')

# Plot the ACF
plt.figure(figsize=(12, 6))
sm.graphics.tsa.plot_acf(monthly_sales)
plt.title('Autocorrelation Function (ACF) of Monthly Sales in Thailand')
plt.xlabel('Lag (months)')
plt.ylabel('Autocorrelation')
plt.savefig('/workspaces/hello/Time series analysis/extra2')
