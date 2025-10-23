import logging, time as _time
from typing import List, Dict
import pandas as pd
from engine.broker import Broker, Order
from engine.strategy_base import StrategyBase
from engine.data_manager import DataManager

class Simulator:
    """
    Boucle sur les timestamps M1 (granularité fine) et déclenche chaque stratégie
    seulement à la clôture de sa barre (timeframe propre).
    """
    def __init__(self, raw_data_by_symbol: Dict[str, pd.DataFrame], strategies: List[StrategyBase], broker: Broker):
        self.broker = broker
        self.strategies = strategies
        self.records = []
        # Data manager
        self.data_mgr = DataManager(raw_data_by_symbol)
        # Base timeline = union M1 de tous les symboles (suppose M1 données)
        base_indexes = []
        for df in raw_data_by_symbol.values():
            base_indexes.append(df.index)
        if not base_indexes:
            raise ValueError("Aucune donnée fournie.")
        # Intersection/time union
        self.m1_index = sorted(set().union(*base_indexes))
        # Pre-cache timeframe data per strategy
        self._tf_cache = {}
        for strat in self.strategies:
            df_tf = self.data_mgr.get(strat.symbol, strat.timeframe)
            self._tf_cache[strat.robot_id] = df_tf
            try: strat.set_environment(self.broker)
            except AttributeError: pass

    def run(self):
        start_t = _time.time()
        total = len(self.m1_index)
        for i, t in enumerate(self.m1_index):
            # Build last_prices from M1 rows if available
            last_prices: Dict[str,float] = {}
            data_slice_symbol_rows: Dict[str, pd.Series] = {}

            for sym, raw_df in self.data_mgr.raw.items():
                if t in raw_df.index:
                    row = raw_df.loc[t]
                    last_prices[sym] = float(row['close'])
                    data_slice_symbol_rows[sym] = row

            # Trigger strategies that have a bar closing at t
            for strat in self.strategies:
                tf_df = self._tf_cache[strat.robot_id]
                # Only proceed if this timestamp is a bar close for that TF
                if t not in tf_df.index:
                    continue
                # Build data_slice with only this symbol's TF bar row
                row_tf = tf_df.loc[t]
                data_slice = { strat.symbol: row_tf }
                orders = []
                try:
                    orders = strat.on_bar(t, data_slice) or []
                except Exception as e:
                    logging.error(f"[SIM] Exception stratégie {strat.robot_id} @ {t}: {e}")
                # Execution price: use TF close (row_tf['close'])
                exec_price = float(row_tf['close'])
                for order in orders:
                    # Vérifier si le robot a stocké un prix spécifique:
                    if hasattr(strat, 'pending_entry_price') and strat.pending_entry_price is not None:
                        execution_price = strat.pending_entry_price
                        strat.pending_entry_price = None  # Reset après utilisation
                    else:
                        execution_price = exec_price  # Prix par défaut (close)
                    
                    pos_id = self.broker.execute(order, execution_price, t)
                    if hasattr(strat, "on_position_opened"):
                        strat.on_position_opened(pos_id, t)

            # TP/SL intrabar: utilise high/low M1
            to_close = []
            for p in list(self.broker.positions):
                # Pour précision, utiliser la barre M1 si dispo
                m1_df = self.data_mgr.raw.get(p.symbol)
                if m1_df is None or t not in m1_df.index:
                    continue
                bar_m1 = m1_df.loc[t]
                high = bar_m1['high']
                low = bar_m1['low']
                if p.take_profit is not None:
                    if p.side == 'LONG' and high >= p.take_profit:
                        to_close.append((p.id, p.take_profit, "tp"))
                    elif p.side == 'SHORT' and low <= p.take_profit:
                        to_close.append((p.id, p.take_profit, "tp"))

            for pid, px, reason in to_close:
                pos = next((pp for pp in self.broker.positions if pp.id == pid), None)
                rid = pos.robot_id if pos else None
                self.broker.close_position(pid, px, time=t, reason=reason)
                if rid:
                    strat = next((s for s in self.strategies if s.robot_id == rid), None)
                    if strat and hasattr(strat, "on_position_closed"):
                        strat.on_position_closed(pid, t, reason)

            # Metrics (equity every minute)
            equity = self.broker.equity(last_prices) if last_prices else self.broker.balance
            unreal = self.broker.unrealized_pnl(last_prices) if last_prices else 0.0
            row_metric = {
                'time': t,
                'balance': self.broker.balance,
                'equity': equity,
                'unrealized_pnl': unreal,
                'open_positions': len(self.broker.positions),
                'lots_open': self.broker.lots_open(),
                'margin_used': self.broker.margin_used(last_prices) if last_prices else 0.0
            }
            for rid, lots in self.broker.lots_by_robot().items():
                row_metric[f'lots_{rid}'] = lots
            for p in self.broker.positions:
                if p.symbol in last_prices:
                    key = f'unrealized_pnl_{p.robot_id}'
                    row_metric[key] = row_metric.get(key, 0.0) + self.broker._pnl_unrealized_single(p, last_prices[p.symbol])
            self.records.append(row_metric)

            if (i+1) % 5000 == 0:
                elapsed = _time.time() - start_t
                logging.info(f"[SIM] Progress {i+1}/{total} {(i+1)/total:.1%} elapsed={elapsed:.1f}s open={len(self.broker.positions)}")

        for strat in self.strategies:
            if hasattr(strat, 'on_finish'):
                try: strat.on_finish()
                except Exception: pass

        elapsed = _time.time() - start_t
        logging.info(f"[SIM] Terminé. Barres M1={total} durée={elapsed:.2f}s positions_finales={len(self.broker.positions)}")
        return pd.DataFrame(self.records)
