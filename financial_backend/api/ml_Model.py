import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from .models import StockData, PredictionData  
from sklearn.metrics import mean_absolute_error
import base64
import matplotlib.pyplot as plt
from decimal import Decimal
from django.utils import timezone
import logging
import os

logger = logging.getLogger(__name__)

def get_data(symbol, days):
    """
    Fetch the stock data from the PostgreSQL for a specific symbol and for specific days.

    Args:
        symbol (str): The stock symbol to fetch data for.
        days (int): The number of days of historical data to fetch.

    Returns:
        pd.DataFrame: A DataFrame containing the stock data for the specified days.
    """
    stock_data_for_specific_days = StockData.objects.filter(symbol=symbol).order_by('-timestamp')[:days]
    df = pd.DataFrame(list(stock_data_for_specific_days.values('timestamp', 'close_price')))
    df['timestamp'] = pd.to_datetime(df['timestamp'])  
    df = df.sort_values(by='timestamp')
    return df

def train_model(data):
    """
    Train a Linear model for the stock data.

    Args:
        data (pd.DataFrame): DataFrame containing historical stock data.

    Returns:
        LinearRegression: The trained LinearRegression model.
    
    Raises:
        ValueError: If the provided data is insufficient for training.
    """
    if len(data) < 2:
        raise ValueError("Not enough data to train the model")

    X = np.arange(len(data)).reshape(-1, 1)
    Y = data['close_price'].values

    model = LinearRegression()
    model.fit(X, Y)

    return model

def predict_prices(symbol, days):
    """
    Predict stock prices for the next given number of days using historical data.

    Args:
        symbol (str): The stock symbol to predict prices for.
        days (int): The number of future days to predict prices for.

    Returns:
        np.ndarray: Array of predicted prices for the specified number of days.
    """
    try:
        df = get_data(symbol=symbol, days=730)  # Getting the previous 2 years of data

        model = train_model(df)

        future_X = np.arange(len(df), len(df) + days).reshape(-1, 1)  
        predictions = model.predict(future_X)  

        for i in range(days):
            predicted_price = round(predictions[i], 2)
            
            stock_data = StockData.objects.filter(symbol=symbol).order_by('-timestamp').first()

            PredictionData.objects.create(
                stock_data=stock_data,
                predicted_price=Decimal(predicted_price),
                prediction_date=timezone.now() + pd.Timedelta(days=i+1),
                model_used='Linear Regression'
            )

        return predictions

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return []

def calculate_prediction_metrics(actual_prices, predicted_prices):
    """
    Find the Mean Absolute Error for the model's predictions.

    Args:
        actual_prices (list): List of actual stock prices.
        predicted_prices (list): List of predicted stock prices.

    Returns:
        dict: A dictionary containing the Mean Absolute Error and total predictions count.
    """
    mae = mean_absolute_error(actual_prices, predicted_prices)

    return {
        'MAE': mae,
        'Total Predictions': len(predicted_prices)
    }

def generate_stock_price_plot(actual_prices, predicted_prices, dates, symbol):
    """
    Generate a plot comparing actual and predicted stock prices and save it as an image.

    Args:
        actual_prices (list): List of actual stock prices.
        predicted_prices (list): List of predicted stock prices.
        dates (list): List of corresponding dates for the prices.
        symbol (str): The stock symbol for the plot title.

    Returns:
        str: The path to the saved image file, or None if an error occurred.
    """
    try:
        dates = [date for date in dates]

        actual_prices_float = [float(price) for price in actual_prices]
        predicted_prices_float = [float(price) for price in predicted_prices]

        plt.figure(figsize=(12, 6))
        plt.plot(dates, actual_prices_float, label='Actual Prices', color='blue', marker='o', markersize=4, linewidth=2)
        plt.plot(dates, predicted_prices_float, label='Predicted Prices', color='orange', marker='x', markersize=4, linewidth=2)

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.title(f'Actual vs Predicted Prices for {symbol}', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Price', fontsize=14)
        plt.xticks(rotation=45)  
        plt.legend(fontsize=12, loc='upper left', shadow=True, fancybox=True, framealpha=0.8)

        plt.ylim(0, max(max(actual_prices_float), max(predicted_prices_float)) * 1.1)
        plt.tight_layout()

        image_folder = os.path.join(os.getcwd(), 'images')
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        image_path = os.path.join(image_folder, f'stock_price_comparison_{symbol}.png')

        plt.savefig(image_path, dpi=300)
        plt.close()

        logger.info(f"Image saved at {os.path.abspath(image_path)}")  

        return image_path

    except Exception as e:
        logger.error(f"Error generating stock price plot: {e}")
        return None

def generate_visualization(symbol, days):
    """
    Fetch stock data, make predictions, generate and return a stock price plot image.

    Args:
        symbol (str): The stock symbol to visualize.
        days (int): The number of future days to predict.

    Returns:
        str: Base64 encoded string of the image, or an error message if failed.
    """
    try:
        df = get_data(symbol, days)

        predicted_prices = predict_prices(symbol, days)

        if not predicted_prices:
            return "Prediction failed"

        actual_prices = df['close_price'].values
        dates = df['timestamp'].dt.strftime('%Y-%m-%d').values

        image_filename = generate_stock_price_plot(actual_prices, predicted_prices, dates, symbol)

        with open(image_filename, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode('utf-8')

        os.remove(image_filename)

        return encoded_string

    except Exception as e:
        logger.error(f"Error generating visualization: {e}")
        return "Error generating visualization"