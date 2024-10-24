import os
import requests
from celery import shared_task
from .models import StockData
from django.conf import settings
import datetime
from dotenv import load_dotenv

load_dotenv()

# API key from .env
API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

@shared_task
def fetch_stock_data(symbol):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}'
    response = requests.get(url)
    data = response.json()

    # Parse data
    if 'Time Series (Daily)' in data:
        for date_str, stats in data['Time Series (Daily)'].items():
            date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
            StockData.objects.update_or_create(
                symbol=symbol,
                date=date,
                defaults={
                    'openPrice': stats['1. open'],
                    'highPrice': stats['2. high'],
                    'lowPrice': stats['3. low'],
                    'closePrice': stats['4. close'],
                    'volume': stats['5. volume'],
                }
            )
