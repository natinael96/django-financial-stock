import pandas as pd
import numpy as np
from django.http import HttpResponse, JsonResponse
from .tasks import fetch_stock_data
from .models import StockData, PredictionData
from datetime import datetime, timedelta
from django.utils import timezone
from .backtest import backtest_strategy, calculate_metrics
from .ml_Model import predict_prices, calculate_prediction_metrics, generate_stock_price_plot
from .utils import generate_pdf_report
import logging
import os

logger = logging.getLogger(__name__)

def home_view(request):
    return HttpResponse("Welcome to the Blockhouse Stock Fetcher App!")


def fetch_data_view(request):
    """
    Fetch stock data for a given symbol and store it in the database.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        JsonResponse: A JSON response indicating success or error.
    """
    symbol = request.GET.get('symbol', 'AMZN')  
    if not symbol:
        return JsonResponse({'status': 'error', 'message': 'Stock symbol is required.'}, status=400)
    try:
        fetch_stock_data(symbol=symbol) 
        return JsonResponse({'status': 'success', 'message': 'Stock Data fetched successfully.'})
    except Exception as error:
        return JsonResponse({'status': 'error', 'message': str(error)}, status=500)


def backtest_view(request):
    """
    Perform backtesting for a given stock symbol using specified moving averages.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        JsonResponse: A JSON response containing the backtest results or an error message.
    """
    symbol = request.GET.get('symbol', 'AMZN')  
    if not symbol:
        return JsonResponse({'status': 'error', 'message': 'Stock symbol is required.'}, status=400)

    initial_investment = float(request.GET.get('initial_investment', 10000))
    short_window = int(request.GET.get('short_window', 50))
    long_window = int(request.GET.get('long_window', 200))
    
    stock_data = StockData.objects.filter(symbol=symbol).order_by('timestamp')
    if not stock_data.exists():
        stock_data_df, error = fetch_stock_data(symbol=symbol, return_as_dataframe=True)
        if error:
            return JsonResponse({'status': 'error', 'message': str(error)}, status=500)
    else:
        stock_data_df = pd.DataFrame(list(stock_data.values('timestamp', 'close_price', 'open_price', 'high_price', 'low_price', 'volume')))

    result = backtest_strategy(symbol=symbol, initial_investment=initial_investment, 
                               short_window=short_window, long_window=long_window)
    return JsonResponse(result)


def predict_view(request):
    """
    Predict future stock prices for a given symbol.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        JsonResponse: A JSON response containing predicted prices or an error message.
    """
    symbol = request.GET.get('symbol', 'AMZN')  
    days = int(request.GET.get('days', 30))  

    stock_data = StockData.objects.filter(symbol=symbol)
    if not stock_data.exists():
        fetch_stock_data(symbol)

    latest_prediction = PredictionData.objects.filter(stock_data__symbol=symbol).order_by('-prediction_date').first()

    if latest_prediction:
        last_prediction_date = latest_prediction.prediction_date.date()
        today = timezone.now().date()

        if last_prediction_date >= today + timedelta(days=days):
            prediction_records = PredictionData.objects.filter(
                stock_data__symbol=symbol,
                prediction_date__gte=today
            ).order_by('prediction_date')[:days]
        else:
            predict_prices(symbol, days)
            prediction_records = PredictionData.objects.filter(stock_data__symbol=symbol).order_by('prediction_date')[:days]
    else:
        predict_prices(symbol, days)
        prediction_records = PredictionData.objects.filter(stock_data__symbol=symbol).order_by('prediction_date')[:days]

    prediction_list = [
        {
            'predicted_price': record.predicted_price,
            'prediction_date': record.prediction_date
        }
        for record in prediction_records
    ]

    return JsonResponse({
        'symbol': symbol,
        'Predictions': prediction_list
    })


def report_view(request):
    """
    Generate a report for a given stock symbol, including backtest and prediction metrics.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        JsonResponse or HttpResponse: A JSON response with report data or a PDF attachment.
    """
    symbol = request.GET.get('symbol', 'AAPL')
    format = request.GET.get('format', 'json')
    days = int(request.GET.get('days', 30))
    initial_investment = 10000

    try:
        stock_data = StockData.objects.filter(symbol=symbol)
        if not stock_data.exists():
            fetch_stock_data(symbol)

        backtest_results = backtest_strategy(symbol=symbol, initial_investment=initial_investment)
        actual_prices = backtest_results.get('actual_prices', [])
        dates = backtest_results.get('dates', [])

        if not actual_prices or not dates:
            return JsonResponse({'status': 'error', 'message': 'Backtest data is incomplete.'}, status=500)

        predict_prices(symbol, days)
        predictions = PredictionData.objects.filter(stock_data__symbol=symbol).order_by('prediction_date')[:days]
        predicted_prices = [p.predicted_price for p in predictions]

        if len(predicted_prices) != len(actual_prices):
            actual_prices = actual_prices[-len(predicted_prices):]
            dates = dates[-len(predicted_prices):]

        backtest_metrics = calculate_metrics(backtest_results)
        prediction_metrics = calculate_prediction_metrics(actual_prices, predicted_prices)

        report_data = {
            'backtest_metrics': backtest_metrics,
            'prediction_metrics': prediction_metrics,
        }

        if format == 'json':
            return JsonResponse(report_data)
        else:
            image_folder = os.path.join(os.getcwd(), 'images')
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)

            image_path = os.path.join(image_folder, f'stock_price_comparison_{symbol}.png')
            generate_stock_price_plot(actual_prices=actual_prices, predicted_prices=predicted_prices, dates=dates, symbol=symbol)

            if os.path.exists(image_path):
                reports_folder = os.path.join(os.getcwd(), 'reports')
                if not os.path.exists(reports_folder):
                    os.makedirs(reports_folder)
                
                pdf_filename = generate_pdf_report(report_data, symbol, image_path)
                pdf_path = os.path.join(reports_folder, pdf_filename)

                if pdf_filename:
                    with open(pdf_path, 'rb') as pdf_file:
                        response = HttpResponse(pdf_file.read(), content_type='application/pdf')
                        response['Content-Disposition'] = f'attachment; filename="{os.path.basename(pdf_filename)}"'
                        return response
                else:
                    logger.error("PDF file could not be generated.")
                    return JsonResponse({'status': 'error', 'message': 'Failed to generate PDF.'}, status=500)
            else:
                logger.error(f"Image not found at path: {image_path}")
                return JsonResponse({'status': 'error', 'message': 'Failed to generate image.'}, status=500)

    except Exception as e:
        logger.error(f"Error generating report for symbol {symbol}: {e}")
        return JsonResponse({'status': 'error', 'message': 'Error generating report.'}, status=500)


def custom_404_view(request, exception=None):
    response_data = {
        'error': 'Page Not Found',
        'message': 'The page you are looking for does not exist.'
    }
    return JsonResponse(response_data, status=404)