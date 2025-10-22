from dataclasses import dataclass
from typing import List, Dict, Optional
import logging

@dataclass
class Order:
    robot_id: str
    symbol: str
    side: str          # 'BUY' ou 'SELL'
    lots: float
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None

@dataclass
class Position:
    id: int
    robot_id: str
    symbol: str
    side: str          # 'LONG' / 'SHORT'
    lots: float
    entry_price: float
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    open_time: Optional[object] = None  # pd.Timestamp

class Broker:
    def __init__(self, starting_balance: float = 10000.0, contract_size: float = 100000,
                 commission_per_lot: float = 0.0, leverage: int = 30):
        self.starting_balance = starting_balance
        self.balance = starting_balance
        self.contract_size = contract_size
        self.commission_per_lot = commission_per_lot
        self.leverage = leverage
        self.positions: List[Position] = []
        self.closed_trades: List[Dict] = []
        self.trade_events: List[Dict] = []
        self._next_pos_id = 1

    def execute(self, order: Order, price: float, time=None) -> int:
        direction = 'LONG' if order.side.upper() == 'BUY' else 'SHORT'
        pos = Position(
            id=self._next_pos_id,
            robot_id=order.robot_id,
            symbol=order.symbol,
            side=direction,
            lots=order.lots,
            entry_price=price,
            take_profit=order.take_profit,
            stop_loss=order.stop_loss,
            open_time=time
        )
        self._next_pos_id += 1
        if self.commission_per_lot:
            self.balance -= self.commission_per_lot * order.lots
        self.positions.append(pos)
        self.trade_events.append({
            "time": time, "event": "open",
            "position_id": pos.id, "robot_id": pos.robot_id,
            "symbol": pos.symbol, "side": pos.side,
            "price": pos.entry_price, "lots": pos.lots
        })
        logging.debug(f"[BROKER] OPEN pos_id={pos.id} robot={pos.robot_id} {pos.side} {pos.symbol} "
                      f"lots={pos.lots} price={pos.entry_price} tp={pos.take_profit} sl={pos.stop_loss} t={time}")
        return pos.id

    def close_position(self, position_id: int, price: float, time=None, reason: str = "close"):
        pos = next((p for p in self.positions if p.id == position_id), None)
        if not pos:
            return
        pnl = self._pnl_unrealized_single(pos, price)
        self.balance += pnl
        self.closed_trades.append({
            "position_id": pos.id, "robot_id": pos.robot_id,
            "symbol": pos.symbol, "side": pos.side,
            "lots": pos.lots, "entry_price": pos.entry_price,
            "exit_price": price, "pnl": pnl,
            "reason": reason, "open_time": pos.open_time, "close_time": time
        })
        self.trade_events.append({
            "time": time, "event": "close",
            "position_id": pos.id, "robot_id": pos.robot_id,
            "symbol": pos.symbol, "side": pos.side,
            "price": price, "lots": pos.lots, "reason": reason
        })
        logging.debug(f"[BROKER] CLOSE pos_id={pos.id} robot={pos.robot_id} {pos.side} {pos.symbol} "
                      f"lots={pos.lots} exit={price} pnl={pnl:.5f} reason={reason} t={time}")
        self.positions = [p for p in self.positions if p.id != position_id]

    def update_take_profit(self, position_id: int, new_tp: Optional[float]):
        pos = next((p for p in self.positions if p.id == position_id), None)
        if pos:
            pos.take_profit = new_tp

    def _pnl_unrealized_single(self, pos: Position, price: float):
        """
        PnL en devise du compte (approx) :
        diff (en prix) * lots * contract_size
        (Hypothèse: compte dans la devise de cotation, pas de conversion FX).
        """
        diff = (price - pos.entry_price) if pos.side == 'LONG' else (pos.entry_price - price)
        return diff * pos.lots * self.contract_size

    def unrealized_pnl(self, last_prices: Dict[str, float]):
        return sum(self._pnl_unrealized_single(p, last_prices[p.symbol]) for p in self.positions if p.symbol in last_prices)

    def equity(self, last_prices: Dict[str, float]):
        """
        Equity = balance réalisé + PnL latent.
        """
        return self.balance + self.unrealized_pnl(last_prices)

    def margin_used(self, last_prices: Dict[str, float]) -> float:
        if self.leverage <= 0:
            return 0.0
        margin = 0.0
        for p in self.positions:
            price = last_prices.get(p.symbol)
            if price is None:
                continue
            notional = price * self.contract_size * p.lots
            margin += notional / self.leverage
        return margin

    def lots_open(self):
        return sum(p.lots for p in self.positions)

    def lots_by_robot(self) -> Dict[str, float]:
        agg: Dict[str, float] = {}
        for p in self.positions:
            agg[p.robot_id] = agg.get(p.robot_id, 0.0) + p.lots
        return agg

    def find_positions_by_robot_side(self, robot_id: str, side: str) -> List[Position]:
        return [p for p in self.positions if p.robot_id == robot_id and p.side == side]

    def has_positions_direction(self, robot_id: str, side: str) -> bool:
        return any(p.robot_id == robot_id and p.side == side for p in self.positions)
