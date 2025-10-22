import pandas as pd

TIMEFRAME_MAP = {
    'm1': '1T','m5':'5T','m15':'15T','m30':'30T',
    'h1':'1H','h4':'4H','d1':'1D'
}

class DataManager:
    """
    Gère les données brutes (supposées M1) et les versions resamplées cache.
    """
    def __init__(self, raw_by_symbol: dict[str, pd.DataFrame]):
        # raw_by_symbol: DataFrames index DatetimeIndex, colonnes open/high/low/close (M1)
        self.raw = raw_by_symbol
        self.cache: dict[tuple[str,str], pd.DataFrame] = {}

    def get(self, symbol: str, timeframe: str) -> pd.DataFrame:
        timeframe = timeframe.lower()
        if timeframe == 'm1':
            return self.raw[symbol]
        key = (symbol, timeframe)
        if key in self.cache:
            return self.cache[key]
        rule = TIMEFRAME_MAP.get(timeframe)
        if rule is None:
            raise ValueError(f"Timeframe inconnu: {timeframe}")
        df = self.raw[symbol].resample(rule).agg({
            'open':'first','high':'max','low':'min','close':'last'
        }).dropna()
        self.cache[key] = df
        return df