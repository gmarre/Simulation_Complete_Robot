from typing import Any, Dict, List, Optional

class StrategyBase:
    """
    Base minimale pour les stratégies.
    """
    def __init__(self, robot_id: str, symbol: str, timeframe: str = 'm1'):
        self.robot_id = robot_id
        self.symbol = symbol
        self.timeframe = timeframe.lower()
        self._broker = None

    def set_environment(self, broker):
        self._broker = broker

    def on_bar(self, time, data_slice: Dict[str, Any]) -> List[Any]:
        return []

    def on_position_opened(self, position_id: int, time):
        pass

    def on_position_closed(self, position_id: int, time, reason: str):
        pass

    def on_finish(self):
        pass
