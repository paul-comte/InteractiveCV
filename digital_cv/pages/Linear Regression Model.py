import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from pandas_datareader import data as pdr
from statsmodels.stats.stattools import jarque_bera
from statsmodels.compat import lzip

# Streamlit page configuration
st.set_page_config(page_title="Linear Regression Model", layout="wide", initial_sidebar_state="expanded")

# Application title
st.title("Linear Regression Model for S&P500")

# Application description
st.markdown("""
This application allows you to create a linear regression model to predict the monthly evolution of the S&P500 using various macroeconomic and stock market variables.
""")

# Load data
@st.cache_data
def load_data():
    index = ['^GSPC', '^IXIC']  # S&P 500 and NASDAQ
    start_date = '2010-01-01'
    end_date = '2023-01-01'

    # Download stock market data
    market_data = yf.download(index, start=start_date, end=end_date, interval='1mo').reset_index()
    market_data["Date"] = market_data["Date"].dt.strftime('%Y/%m')
    market_data = market_data[["Date", "Close"]]
    market_data.columns = [v[0] + "_" + v[1].replace("^", "") if v[1] != "" else v[0] for v in market_data.columns]

    # Download macroeconomic data
    unemployment_rate = pdr.get_data_fred('UNRATE', start=start_date, end=end_date).reset_index()
    unemployment_rate["Date"] = unemployment_rate["DATE"].dt.strftime('%Y/%m')
    unemployment_rate.drop("DATE", axis="columns", inplace=True)

    interest_rate = pdr.get_data_fred('DFF', start=start_date, end=end_date, freq="1mo").resample('M').last().reset_index()
    interest_rate["Date"] = interest_rate["DATE"].dt.strftime('%Y/%m')
    interest_rate.drop("DATE", axis="columns", inplace=True)

    # Merge data
    data = market_data[["Date", 'Close_GSPC', 'Close_IXIC']].merge(unemployment_rate, on="Date", how="left").merge(interest_rate, on="Date", how="left")
    data = data.drop("Date", axis=1)

    # Differentiate data
    data["Close_GSPC"] = data["Close_GSPC"].diff()
    data["Close_IXIC"] = data["Close_IXIC"].diff()
    data["UNRATE"] = data["UNRATE"].diff()
    data["DFF"] = data["DFF"].diff()
    data = data.dropna()

    return data

data = load_data()

# Data preprocessing
preprocess = StandardScaler()
preprocessed_data = preprocess.fit_transform(data)
preprocessed_data = pd.DataFrame(preprocessed_data, columns=["Close_GSPC", "Close_IXIC", "UNRATE", "DFF"])

# Calculate moving averages
target = "Close_GSPC"
preprocessed_data['MA_3'] = preprocessed_data[target].rolling(window=6).mean()
preprocessed_data['MA_6'] = preprocessed_data[target].rolling(window=3).mean()
preprocessed_data['MA_12'] = preprocessed_data[target].rolling(window=12).mean()
preprocessed_data['MA_24'] = preprocessed_data[target].rolling(window=24).mean()
preprocessed_data['N_MA_3'] = preprocessed_data["Close_IXIC"].rolling(window=6).mean()
preprocessed_data['N_MA_6'] = preprocessed_data["Close_IXIC"].rolling(window=3).mean()

preprocessed_data['Target'] = preprocessed_data[target]
preprocessed_data = preprocessed_data.drop(target, axis=1)
preprocessed_data.dropna(inplace=True)

# Feature selection
st.sidebar.header("Feature Selection")
features = st.sidebar.multiselect(
    "Select input features",
    options=preprocessed_data.columns.drop('Target'),
    default=["Close_IXIC", "MA_3", "MA_6"]
)

# Model parameters
st.sidebar.header("Model Parameters")

# Prepare data for the model
X = preprocessed_data[features]
y = preprocessed_data['Target']

# Split data into training and test sets
train_size = len(preprocessed_data) - 36
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# Add a constant for the regression model
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

# Train the model
model = sm.OLS(y_train, X_train).fit()

# Display model results
st.header("Model Results")
st.write(model.summary())

# Predictions and model evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"Mean Squared Error (MSE): {mse}")
st.write(f"Mean Absolute Error (MAE): {mae}")
st.write(f"R-squared (R²): {r2}")

# Explanation of metrics
st.markdown("""
- **Mean Squared Error (MSE)**: Measures the average of the squares of the errors. It is more sensitive to outliers compared to MAE. Lower values indicate better model performance.
- **Mean Absolute Error (MAE)**: Measures the average of the absolute errors. It provides a linear score, meaning all errors are weighted equally. Lower values indicate better model performance.
- **R-squared (R²)**: Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables. Values range from 0 to 1, with higher values indicating better model fit.
""")

# Visualization of predictions
st.header("Prediction Plot")
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(preprocessed_data.index[train_size:], y_test.values, label='Actual Prices', color='blue')
ax.plot(preprocessed_data.index[train_size:], y_pred, label='Predicted Prices', color='red')
ax.set_title('Actual vs Predicted Prices')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig)

# Visualization of residuals
st.header("Residuals Histogram")
fig, ax = plt.subplots(figsize=(14, 7))
ax.hist(model.resid, bins=20, color='red', alpha=0.7)
ax.set_title('Histogram of Residuals')
ax.set_xlabel('Residuals')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# Explanation of residuals
st.markdown("""
- **Residuals**: The difference between the observed and predicted values. A good model should have residuals that are randomly distributed and centered around zero.
- **Histogram of Residuals**: Shows the distribution of residuals. Ideally, it should resemble a normal distribution, indicating that the model's errors are random and not biased.
""")

# Jarque-Bera test for normality of residuals
name = ["Jarque-Bera", "Chi^2 two-tail prob.", "Skew", "Kurtosis"]
test = jarque_bera(model.resid)
st.write(lzip(name, test))

# Explanation of Jarque-Bera test
st.markdown("""
- **Jarque-Bera Test**: Tests the null hypothesis that the residuals are normally distributed. A high Jarque-Bera statistic and a low p-value indicate that the residuals are not normally distributed.
- **Skew**: Measures the asymmetry of the residuals' distribution. A value close to zero indicates symmetry.
- **Kurtosis**: Measures the "tailedness" of the residuals' distribution. A value close to three indicates a normal distribution.
""")

# User tips
st.sidebar.header("Tips")
st.sidebar.markdown("""
- Use a combination of features to improve model performance.
- Experiment with different moving averages to capture trends.
- Monitor residuals to ensure model assumptions are met.
""")
