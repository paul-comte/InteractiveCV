import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report, accuracy_score

# Streamlit page configuration
st.set_page_config(page_title="Logistic Regression Model", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for styling
st.markdown("""
    <style>
    .title {
        background-color: orange;
        padding: 10px;
        border-radius: 5px;
        color: white;
        text-align: center;
    }
    .prediction-box {
        background-color: lightgray;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .prediction-result {
        font-size: 1.5em;
        font-weight: bold;
    }
    .up {
        color: green;
    }
    .down {
        color: red;
    }
    </style>
    """, unsafe_allow_html=True)

# Application title with custom styling
st.markdown('<div class="title"><h1>Logistic Regression Model for Stock Price Movement</h1></div>', unsafe_allow_html=True)

# Application description
st.markdown("""
This application allows you to create a logistic regression model to predict the movement (up or down) of a target stock based on the price movements of other stocks.
""")

# Load data
@st.cache_data
def load_data(companies, start_date, end_date):
    data = yf.download(companies, start=start_date, end=end_date)['Adj Close']
    returns = data.diff().dropna().reset_index().drop("Date", axis=1)
    return returns

# List of companies
companies = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'BRK-B', 'JPM', 'JNJ', 'V',
             'WMT', 'PG', 'UNH', 'NVDA', 'HD', 'DIS', 'PYPL', 'MA', 'VZ', 'ADBE']

# User input for stock selection
st.sidebar.header("Stock Selection")
selected_stocks = st.sidebar.multiselect(
    "Select stocks to include in the model",
    options=companies,
    default=['AAPL', 'MSFT', 'AMZN', 'META']
)

# User input for target stock
all_stocks = companies + ["Other (specify below)"]
target_stock_option = st.sidebar.selectbox("Select the target stock", options=all_stocks)

if target_stock_option == "Other (specify below)":
    target_stock = st.sidebar.text_input("Enter the target stock ticker:", value="").strip().upper()
    if target_stock:
        selected_stocks = [stock for stock in selected_stocks if stock != target_stock]
else:
    target_stock = target_stock_option

# User input for date range
st.sidebar.header("Date Range")
start_date = st.sidebar.date_input("Start date", value=pd.to_datetime('2018-01-01'))
end_date = st.sidebar.date_input("End date", value=pd.to_datetime('2024-01-01'))

# Load data based on user selection
data = load_data(selected_stocks + [target_stock], start_date, end_date)

# Prepare the data
data['TARGET'] = (data[target_stock] > 0).astype(int)
data = data.drop(columns=[target_stock])
data["AVG_day"] = data.mean(axis=1)

# Check if necessary stocks for AVG_day_GAFAM are selected
gafam_stocks = ['AAPL', 'MSFT', 'AMZN', 'META']
if all(stock in selected_stocks for stock in gafam_stocks):
    data["AVG_day_GAFAM"] = data[gafam_stocks].mean(axis=1)
else:
    st.warning("Selected stocks do not include all GAFAM stocks (AAPL, MSFT, AMZN, META). AVG_day_GAFAM will not be calculated.")

data = data.dropna()

# Split the data
preprocessor = StandardScaler()
train, test = train_test_split(data, test_size=0.2, shuffle=True, random_state=321, stratify=data["TARGET"])

X_train = train.drop(columns=["TARGET"])
X_train = preprocessor.fit_transform(X_train)
X_train = pd.DataFrame(X_train, columns=train.drop(columns=["TARGET"]).columns)
y_train = train["TARGET"]

X_test = test.drop(columns=["TARGET"])
X_test = preprocessor.transform(X_test)
X_test = pd.DataFrame(X_test, columns=test.drop(columns=["TARGET"]).columns)
y_test = test["TARGET"]

# User input for model parameters
st.sidebar.header("Model Parameters")
C = st.sidebar.number_input("Regularization Strength (C)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)

# Train the model with Ridge regularization
model = LogisticRegression(fit_intercept=False, penalty="l2", C=C, solver="lbfgs", max_iter=1000)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

# Display results
st.header("Model Results")
st.write(f"Accuracy: {accuracy:.4f}")
st.write(f"ROC AUC Score: {roc_auc:.4f}")
st.write(f"Cross-Validation Accuracy: {cv_scores.mean():.4f}")

# Confusion Matrix
st.header("Confusion Matrix")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
st.pyplot(fig)

# ROC Curve
st.header("ROC Curve")
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic')
ax.legend(loc="lower right")
st.pyplot(fig)

# Classification Report
st.header("Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)

# Feature Importance
st.header("Feature Importance")
coef_logistic = pd.DataFrame({"Variables": train.drop(columns=["TARGET"]).columns, "Coef": model.coef_[0]})
st.dataframe(coef_logistic.sort_values("Coef", ascending=False))

# User tips and explanations
st.sidebar.header("Tips and Explanations")
st.sidebar.markdown("""
- **Accuracy**: The proportion of true results among the total number of cases examined.
- **ROC AUC Score**: Measures the ability of the model to distinguish between classes.
- **Confusion Matrix**: A table used to describe the performance of a classification model.
- **Feature Importance**: Shows the impact of each feature on the model's predictions.
- **Regularization**: Helps to prevent overfitting by adding a penalty for large coefficients.
- **Cross-Validation**: Evaluates the model's performance on different subsets of the data.
""")

# Additional metrics and visualizations
st.header("Additional Metrics")
st.markdown("""
- **Precision**: The ratio of correctly predicted positive observations to the total predicted positives.
- **Recall**: The ratio of correctly predicted positive observations to all observations in the actual class.
- **F1 Score**: The weighted average of Precision and Recall.
""")

# Precision-Recall Curve
st.header("Precision-Recall Curve")
from sklearn.metrics import precision_recall_curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(recall, precision, color='purple', lw=2)
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve')
st.pyplot(fig)

# Predict tomorrow's stock movement
st.markdown('<div class="prediction-box"><h2>Tomorrow\'s Stock Movement Prediction</h2>', unsafe_allow_html=True)

# Get the latest data for prediction
latest_data = yf.download(selected_stocks + [target_stock], period="1d")['Adj Close'].diff().dropna()
latest_data = latest_data.T

# Ensure the latest data has the same columns as the training data
latest_data = latest_data.reindex(columns=X_train.columns, fill_value=0)

# Prepare the latest data for prediction
latest_data["AVG_day"] = latest_data.mean(axis=1)
if all(stock in selected_stocks for stock in gafam_stocks):
    latest_data["AVG_day_GAFAM"] = latest_data[gafam_stocks].mean(axis=1)

latest_data = latest_data.drop(columns=[target_stock], errors='ignore')
latest_data = preprocessor.transform(latest_data)
latest_data = pd.DataFrame(latest_data, columns=X_train.columns)

# Make prediction
tomorrow_pred_prob = model.predict_proba(latest_data)[:, 1][0]
tomorrow_pred = "Up" if tomorrow_pred_prob >= 0.5 else "Down"

# Display prediction
prediction_class = "up" if tomorrow_pred == "Up" else "down"
st.markdown(f'<div class="prediction-result {prediction_class}">Predicted movement for tomorrow: <span>{tomorrow_pred}</span></div>', unsafe_allow_html=True)
st.markdown(f'<div class="prediction-result">Probability of moving up: <span>{tomorrow_pred_prob:.2%}</span></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
