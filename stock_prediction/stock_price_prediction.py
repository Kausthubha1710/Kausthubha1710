import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import yfinance as yf
import matplotlib.pyplot as plt

# Step 1: Fetch historical stock data
def fetch_stock_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

# Step 2: Prepare the data
def prepare_data(data):
    data['Date'] = data.index
    data.reset_index(drop=True, inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    data['Ordinal_Date'] = data['Date'].apply(lambda x: x.toordinal())
    return data

# Step 3: Train the model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Step 4: Make predictions
def make_predictions(model, X_test):
    predictions = model.predict(X_test)
    return predictions

# Step 5: Visualize the results
def visualize_results(data, predictions):
    plt.figure(figsize=(14,7))
    plt.plot(data['Date'], data['Close'], label='Actual Stock Price')
    plt.plot(data['Date'][-len(predictions):], predictions, label='Predicted Stock Price')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Prediction')
    plt.legend()
    plt.show()

def main():
    # Fetch historical data for a particular stock
    symbol = 'AAPL'  # Example: Apple Inc.
    start_date = '2020-01-01'
    end_date = '2022-01-01'
    stock_data = fetch_stock_data(symbol, start_date, end_date)
    
    # Prepare the data
    prepared_data = prepare_data(stock_data)
    
    # Define features and target variable
    X = prepared_data[['Ordinal_Date']]
    y = prepared_data['Close']
    
    # Train the model
    model, X_test, y_test = train_model(X, y)
    
    # Make predictions
    predictions = make_predictions(model, X_test)
    
    # Visualize the results
    visualize_results(prepared_data, predictions)

if __name__ == "__main__":
    main()
