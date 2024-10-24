import pandas as pd
from .models import StockData
import os
import logging
import io
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

logger = logging.getLogger(__name__)

def get_stock_data(symbol):
    """
    Retrieve stock data for a given symbol from the database.

    Args:
        symbol (str): The stock symbol to retrieve data for.

    Returns:
        pd.DataFrame: A DataFrame containing the stock data.

    Raises:
        ValueError: If no data is found for the symbol.
    """
    stock_records = StockData.objects.filter(symbol=symbol).order_by('timestamp')
    if not stock_records.exists():
        raise ValueError(f"No data found for symbol: {symbol}")

    data = {
        'timestamp': [record.timestamp for record in stock_records],
        'close_price': [record.close_price for record in stock_records],
        'open_price': [record.open_price for record in stock_records],
        'high_price': [record.high_price for record in stock_records],
        'low_price': [record.low_price for record in stock_records],
        'volume': [record.volume for record in stock_records],
    }
    df = pd.DataFrame(data)
    return df

def generate_pdf_report(report_data, symbol, image_path):
    """
    Generate a PDF report containing backtest and prediction metrics for a given stock symbol.

    Args:
        report_data (dict): The data to include in the report.
        symbol (str): The stock symbol for the report.
        image_path (str): The path to the image to include in the report.

    Returns:
        str: The filename of the generated PDF report, or None if failed.
    """
    try:
        if not isinstance(report_data, dict):
            raise ValueError(f"Expected 'report_data' to be a dictionary but got {type(report_data)}")

        reports_folder = os.path.join(os.getcwd(), 'reports')
        if not os.path.exists(reports_folder):
            os.makedirs(reports_folder)

        pdf_filename = os.path.join(reports_folder, f'{symbol}_stock_report.pdf')
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            'Title',
            parent=styles['Title'],
            fontSize=24,
            textColor=colors.HexColor("#3498db"),
            spaceAfter=20,
            alignment=1  # Center alignment
        )

        content = []

        title = f"Stock Report for {symbol}"
        content.append(Paragraph(title, title_style))

        content.append(Spacer(1, 12))
        content.append(Paragraph("Backtest Metrics:", styles['Heading2']))
        backtest_data = [[key, str(value)] for key, value in report_data.get('backtest_metrics', {}).items()]
        backtest_table = Table(backtest_data)
        backtest_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#ecf0f1")),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        content.append(backtest_table)
        content.append(Spacer(1, 20))

        content.append(Paragraph("Prediction Metrics:", styles['Heading2']))
        prediction_data = [[key, str(value)] for key, value in report_data.get('prediction_metrics', {}).items()]
        prediction_table = Table(prediction_data)
        prediction_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#ecf0f1")),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        content.append(prediction_table)
        content.append(Spacer(1, 20))

        if os.path.exists(image_path):
            img = Image(image_path, 6 * inch, 3 * inch)
            img.hAlign = 'CENTER'
            content.append(img)
        else:
            logger.error(f"Image not found at path: {image_path}")

        content.append(PageBreak())

        doc.build(content)

        with open(pdf_filename, "wb") as f:
            f.write(buffer.getvalue())

        return pdf_filename

    except Exception as e:
        logger.error(f"Error generating PDF report for {symbol}: {e}")
        return None
