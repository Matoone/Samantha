import yfinance as yf
import pandas as pd

class YFinanceWrapper:
    def __init__(self):
        pass  # Pas besoin d'initialisation spéciale pour yfinance

    async def get_stock_history(self, symbol: str, period: str) -> pd.DataFrame:
        """
        Gets historical stock data for a given symbol.
        """
        try:
            stock = yf.Ticker(symbol)
            history = stock.history(period=period)
            if history.empty:
                raise Exception(f"No data found for symbol: {symbol}")
            return history
        except Exception as e:
            raise Exception(f"Error fetching stock data: {e}")

    async def get_stock_info(self, symbol: str) -> dict:
        """
        Gets detailed information about a stock.
        """
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            return info
        except Exception as e:
            raise Exception(f"Error fetching stock info: {e}")