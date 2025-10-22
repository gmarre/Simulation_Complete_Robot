from typing import Dict, List
import pandas as pd

class DataFeed:
    """Synchronise et fournit les barres multi-symboles sur une timeline unifiée."""
    def __init__(self, data_by_symbol: Dict[str, pd.DataFrame]):
        # On suppose que chaque DataFrame a un index datetime et contient au moins 'close'
        self.data_by_symbol = data_by_symbol
        # Timeline unifiée (union)
        self.timeline: List[pd.Timestamp] = sorted(set().union(*[df.index for df in data_by_symbol.values()]))

    def iter(self):
        last_row_per_symbol = {}
        for t in self.timeline:
            slice_dict = {}
            for sym, df in self.data_by_symbol.items():
                if t in df.index:
                    row = df.loc[t]
                    last_row_per_symbol[sym] = row
                # On fournit la dernière ligne disponible (forward fill) si besoin
                if sym in last_row_per_symbol:
                    slice_dict[sym] = last_row_per_symbol[sym]
            yield t, slice_dict
