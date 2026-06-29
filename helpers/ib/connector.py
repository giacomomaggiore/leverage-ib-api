"""IB Gateway connector for downloading historical data via ib_insync.

Requires IB Gateway (or TWS) running and accepting API connections:
  - IB Gateway paper: port 4002
  - IB Gateway live:  port 4001
  - TWS paper:        port 7497

Usage:
    from helpers.ib.connector import IBConnector
    ib = IBConnector()          # defaults to IB Gateway paper (4002)
    ib.connect()
    df = ib.download_historical('SPY', duration='5 Y')
    ib.download_historical_csv('SPY', 'data/SPY.csv')
    ib.disconnect()
"""
from typing import Optional
import pandas as pd

try:
    import nest_asyncio
    nest_asyncio.apply()
    from ib_insync import IB, Stock, util
except ImportError:
    IB = None
    Stock = None
    util = None


class IBConnector:
    def __init__(self, host: str = '127.0.0.1', port: int = 4002, clientId: int = 1):
        self.host = host
        self.port = port
        self.clientId = clientId
        self.ib: Optional[IB] = None

    def connect(self, timeout: int = 5) -> bool:
        if IB is None:
            raise ImportError('ib_insync is not installed')
        self.ib = IB()
        self.ib.connect(self.host, self.port, clientId=self.clientId, timeout=timeout)
        return self.ib.isConnected()

    def disconnect(self) -> None:
        if self.ib is not None and self.ib.isConnected():
            try:
                self.ib.disconnect()
            finally:
                self.ib = None

    def download_historical(
        self,
        ticker: str,
        duration: str = '5 Y',
        barSize: str = '1 day',
        whatToShow: str = 'TRADES',
        useRTH: bool = True,
    ) -> pd.DataFrame:
        if self.ib is None:
            raise RuntimeError('Not connected. Call connect() first.')
        contract = Stock(ticker, 'SMART', 'USD')
        bars = self.ib.reqHistoricalData(
            contract, endDateTime='', durationStr=duration,
            barSizeSetting=barSize, whatToShow=whatToShow,
            useRTH=useRTH, formatDate=1,
        )
        df = util.df(bars)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df

    def download_historical_csv(self, ticker: str, filename: str, **kwargs) -> str:
        df = self.download_historical(ticker, **kwargs)
        df.to_csv(filename)
        return filename
