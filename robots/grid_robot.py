from typing import List, Dict, Any
import pandas as pd
from engine.strategy_base import StrategyBase
from engine.broker import Order

class GridRobot(StrategyBase):
    """Robot grille simplifié: place des ordres de sens alternés quand l'écart au prix de référence dépasse un pas.

    Paramètres:
        symbol: symbole tradé
        grid_step: pourcentage ou fraction du prix (ex: 0.002 = 0.2%)
        max_levels: nombre max de positions ouvertes
        base_lots: taille d'une position
        direction_bias: 'long', 'short' ou None (alternance)
    """
    def __init__(self, robot_id: str, symbol: str, grid_step: float = 0.002, max_levels: int = 5, base_lots: float = 0.1, direction_bias: str = None):
        super().__init__(robot_id, symbol)
        self.symbol = symbol
        self.grid_step = grid_step
        self.max_levels = max_levels
        self.base_lots = base_lots
        self.direction_bias = direction_bias  # peut forcer toujours BUY ou SELL
        self.reference_price = None
        self.last_direction = 'BUY'  # pour alterner si pas de bias

    def on_bar(self, time: pd.Timestamp, data_slice: Dict[str, Any]):
        if self.symbol not in data_slice:
            return []
        price = data_slice[self.symbol]['close']
        if self.reference_price is None:
            self.reference_price = price
            return []
        distance = (price - self.reference_price) / self.reference_price
        orders: List[Order] = []
        # Si le prix s'est éloigné d'un multiple du step depuis la ref -> placer un ordre
        if abs(distance) >= self.grid_step:
            # déterminer direction
            if self.direction_bias:
                side = 'BUY' if self.direction_bias.lower() == 'long' else 'SELL'
            else:
                # alterner
                side = 'SELL' if self.last_direction == 'BUY' else 'BUY'
            orders.append(Order(robot_id=self.robot_id, symbol=self.symbol, side=side, lots=self.base_lots))
            self.last_direction = side
            # réinitialiser référence
            self.reference_price = price
        return orders
