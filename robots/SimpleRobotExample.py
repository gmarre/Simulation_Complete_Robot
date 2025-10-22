from typing import Dict, Any, Optional, List
from engine.strategy_base import StrategyBase
from engine.broker import Order

class DailyTimeWindowRobot(StrategyBase):
    """
    Ouvre une position chaque jour à open_hour:open_minute (par défaut 10:00)
    et la ferme à close_hour:close_minute (par défaut 12:00).
    - Une seule position simultanée par robot.
    - Aucune logique de signal : purement temporel.
    """
    def __init__(
        self,
        robot_id: str,
        symbol: str,
        timeframe: str = 'm1',
        side: str = "BUY",
        lots: float = 0.1,
        open_hour: int = 10,
        open_minute: int = 0,
        close_hour: int = 12,
        close_minute: int = 0,
        debug: bool = False
    ):
        super().__init__(robot_id, symbol, timeframe=timeframe)
        self.side = side.upper()  # 'BUY' ou 'SELL'
        self.lots = lots
        self.open_hour = open_hour
        self.open_minute = open_minute
        self.close_hour = close_hour
        self.close_minute = close_minute
        self.debug = debug
        self._open_position_id: Optional[int] = None
        self._last_open_trade_date: Optional[str] = None  # YYYY-MM-DD

    def _log(self, msg: str):
        if self.debug:
            print(f"[{self.robot_id}] {msg}")

    def _has_open_position(self) -> bool:
        if not self._broker:
            return False
        return any(p.id == self._open_position_id for p in self._broker.positions)

    def on_bar(self, time, data_slice: Dict[str, Any]):
        if self.symbol not in data_slice:
            return []
        orders: List[Order] = []

        # Heure / minute de la barre
        h = time.hour
        m = time.minute
        current_date = time.date().isoformat()

        # 1) Tentative d'ouverture à l'heure définie
        if (h == self.open_hour and m == self.open_minute
            and not self._has_open_position()
            and self._last_open_trade_date != current_date):
            # Crée l'ordre
            orders.append(Order(
                robot_id=self.robot_id,
                symbol=self.symbol,
                side=self.side,
                lots=self.lots
            ))
            self._last_open_trade_date = current_date
            self._log(f"Ouvrir position {self.side} @ {time}")

        # 2) Tentative de fermeture à l'heure définie (si position ouverte)
        if (h == self.close_hour and m == self.close_minute
            and self._has_open_position()
            and self._open_position_id is not None
            and self._broker):
            # Ferme au prix de clôture de la barre (approx) : on utilise close
            bar = data_slice[self.symbol]
            price_close = float(bar['close'])
            self._broker.close_position(self._open_position_id, price_close, time=time, reason="time_exit")
            self._log(f"Fermeture position id={self._open_position_id} @ {time}")
            self._open_position_id = None

        return orders

    def on_position_opened(self, position_id: int, time):
        # Enregistre la position gérée
        self._open_position_id = position_id

    def on_position_closed(self, position_id: int, time, reason: str):
        if position_id == self._open_position_id:
            self._open_position_id = None