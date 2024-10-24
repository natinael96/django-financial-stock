from django.db import models
from django.utils import timezone

class StockData(models.Model):
    """
    Model representing historical stock data for a specific symbol.
    """
    symbol = models.CharField(max_length=10)
    timestamp = models.DateTimeField(default=timezone.now)
    open_price = models.DecimalField(max_digits=10, decimal_places=2)
    close_price = models.DecimalField(max_digits=10, decimal_places=2)
    high_price = models.DecimalField(max_digits=10, decimal_places=2)
    low_price = models.DecimalField(max_digits=10, decimal_places=2)
    volume = models.BigIntegerField()

    class Meta:
        unique_together = ('symbol', 'timestamp')
        indexes = [
            models.Index(fields=['symbol', 'timestamp']),
        ]
    
    def __str__(self):
        return f"{self.symbol} - {self.timestamp}"


class PredictionData(models.Model):
    """
    Model representing predicted stock prices for future dates.
    """
    stock_data = models.ForeignKey(StockData, on_delete=models.CASCADE, related_name='predictions')
    predicted_price = models.DecimalField(max_digits=14, decimal_places=4)
    prediction_date = models.DateTimeField()
    model_used = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction for {self.stock_data.symbol} on {self.prediction_date} by {self.model_used}"

    class Meta:
        unique_together = ('stock_data', 'prediction_date')
        indexes = [
            models.Index(fields=['prediction_date']),
        ]
