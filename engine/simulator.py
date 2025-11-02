import logging
import time as _time
from typing import List, Dict
import pandas as pd
from engine.broker import Broker
from engine.strategy_base import StrategyBase
from engine.data_manager import DataManager

class Simulator:
    """
    Boucle sur les timestamps M1 et déclenche chaque stratégie
    seulement à la clôture de sa barre (timeframe propre).
    """
    def __init__(self, raw_data_by_symbol: Dict[str, pd.DataFrame], strategies: List[StrategyBase], broker: Broker):
        self.broker = broker
        self.strategies = strategies
        self.records = []
        
        # Data manager
        self.data_mgr = DataManager(raw_data_by_symbol)
        
        # Base timeline = union M1 de tous les symboles
        base_indexes = []
        for df in raw_data_by_symbol.values():
            base_indexes.append(df.index)
        if not base_indexes:
            raise ValueError("Aucune donnée fournie.")
        
        self.m1_index = sorted(set().union(*base_indexes))
        
        # Liste de tous les symboles disponibles pour conversion
        self.available_symbols = list(raw_data_by_symbol.keys())
        
        # Pre-cache timeframe data per strategy
        self._tf_cache = {}
        for strat in self.strategies:
            df_tf = self.data_mgr.get(strat.symbol, strat.timeframe)
            self._tf_cache[strat.robot_id] = df_tf
            
            # Set environment for strategy
            try:
                strat.set_environment(self.broker)
            except AttributeError:
                pass

    def run(self) -> pd.DataFrame:
        """
        Exécute la simulation et retourne un DataFrame avec les métriques à chaque tick.
        """
        start_t = _time.time()
        total = len(self.m1_index)
        
        logging.info(f"🚀 Démarrage simulation: {total} ticks M1 à traiter")
        
        for i, t in enumerate(self.m1_index):
            # ========== 1. MISE À JOUR DES PRIX (OHLC COMPLET) ==========
            current_prices = {}
            for sym in self.available_symbols:
                if sym in self.data_mgr.raw and t in self.data_mgr.raw[sym].index:
                    row = self.data_mgr.raw[sym].loc[t]
                    current_prices[sym] = {
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'bid': float(row['close']),  # Simplifié: bid = close
                        'ask': float(row['close'])   # Simplifié: ask = close
                    }
            
            # ========== 2. VÉRIFICATION TP/SL (avec high/low) ==========
            self._check_tp_sl(t, current_prices)
            
            # ========== 3. EXÉCUTION DES STRATÉGIES ==========
            for strat in self.strategies:
                tf_df = self._tf_cache[strat.robot_id]
                
                # Trigger uniquement à la clôture de la barre du timeframe
                if t not in tf_df.index:
                    continue
                
                row_tf = tf_df.loc[t]
                data_slice = {strat.symbol: row_tf}
                
                orders = []
                try:
                    orders = strat.on_bar(t, data_slice) or []
                except Exception as e:
                    logging.error(f"[SIM] Exception stratégie {strat.robot_id} @ {t}: {e}")
                
                # Exécution des ordres
                exec_price = float(row_tf['close'])
                
                for order in orders:
                    # Prix d'exécution personnalisé si disponible
                    if hasattr(strat, 'pending_entry_price') and strat.pending_entry_price is not None:
                        execution_price = strat.pending_entry_price
                        strat.pending_entry_price = None
                    else:
                        execution_price = exec_price
                    
                    pos_id = self.broker.execute(order, execution_price, t)
                    
                    # Notification au robot
                    if hasattr(strat, "on_position_opened"):
                        try:
                            strat.on_position_opened(pos_id, t)
                        except Exception as e:
                            logging.error(f"[SIM] Erreur on_position_opened {strat.robot_id}: {e}")
            
            # ========== 4. ENREGISTREMENT MÉTRIQUES ==========
            equity = self.broker.equity(current_prices) if current_prices else self.broker.balance
            unreal = self.broker.unrealized_pnl(current_prices) if current_prices else 0.0
            
            row_metric = {
                'time': t,
                'balance': self.broker.balance,
                'equity': equity,
                'unrealized_pnl': unreal,
                'open_positions': len(self.broker.positions),
                'lots_open': self.broker.lots_open(),
                'margin_used': self.broker.margin_used(current_prices) if current_prices else 0.0
            }
            
            # Lots par robot
            for rid, lots in self.broker.lots_by_robot().items():
                row_metric[f'lots_{rid}'] = lots
            
            self.records.append(row_metric)
            
            # Log de progression
            if (i + 1) % 5000 == 0:
                elapsed = _time.time() - start_t
                logging.info(
                    f"[SIM] Progress {i+1}/{total} ({(i+1)/total*100:.1f}%) | "
                    f"Temps={elapsed:.1f}s | Positions={len(self.broker.positions)}"
                )
        
        # ========== 5. FINALISATION ==========
        for strat in self.strategies:
            if hasattr(strat, 'on_finish'):
                try:
                    strat.on_finish()
                except Exception as e:
                    logging.error(f"[SIM] Erreur on_finish {strat.robot_id}: {e}")
        
        elapsed = _time.time() - start_t
        logging.info(
            f"[SIM] ✅ Simulation terminée | "
            f"Barres M1={total} | Durée={elapsed:.2f}s | "
            f"Positions finales={len(self.broker.positions)} | "
            f"Métriques enregistrées={len(self.records)}"
        )
        
        # ========== 6. RETOUR DU DATAFRAME ==========
        results_df = pd.DataFrame(self.records)
        
        if results_df.empty:
            logging.warning("⚠️ Aucune métrique enregistrée!")
            return pd.DataFrame()
        
        logging.info(f"📊 DataFrame créé: {len(results_df)} lignes × {len(results_df.columns)} colonnes")
        return results_df
        # ============================================

    def _check_tp_sl(self, time, current_prices: Dict[str, Dict[str, float]]):
        """
        Vérifie les TP/SL pour toutes les positions.
        FIX: Utilise HIGH pour TP LONG et LOW pour TP SHORT (réaliste intra-barre)
        """
        positions_to_close = []
        
        for pos in self.broker.positions:
            symbol_prices = current_prices.get(pos.symbol)
            if symbol_prices is None:
                continue
            
            # Prix de clôture pour fermeture effective
            close_price = symbol_prices['close']
            
            # Prix réels d'atteinte TP/SL (high/low)
            high_price = symbol_prices['high']
            low_price = symbol_prices['low']
            
            tp_hit = False
            sl_hit = False
            
            # Normalisation du side
            is_long = pos.side in ('LONG', 'BUY')
            is_short = pos.side in ('SHORT', 'SELL')
            
            # ========== VÉRIFICATION TP (avec high/low) ==========
            if pos.take_profit is not None:
                if is_long:
                    # LONG: TP atteint si HIGH >= TP
                    tp_hit = high_price >= pos.take_profit
                    if tp_hit:
                        logging.info(
                            f"[SIM] [{time}] 🎯 TP HIT LONG | Pos {pos.id} | "
                            f"TP={pos.take_profit:.5f} | High={high_price:.5f} | "
                            f"Fermeture à close={close_price:.5f}"
                        )
                
                elif is_short:
                    # SHORT: TP atteint si LOW <= TP
                    tp_hit = low_price <= pos.take_profit
                    if tp_hit:
                        logging.info(
                            f"[SIM] [{time}] 🎯 TP HIT SHORT | Pos {pos.id} | "
                            f"TP={pos.take_profit:.5f} | Low={low_price:.5f} | "
                            f"Fermeture à close={close_price:.5f}"
                        )
            
            # ========== VÉRIFICATION SL (avec high/low) ==========
            if pos.stop_loss is not None:
                if is_long:
                    # LONG: SL atteint si LOW <= SL
                    sl_hit = low_price <= pos.stop_loss
                    if sl_hit:
                        logging.info(
                            f"[SIM] [{time}] 🛑 SL HIT LONG | Pos {pos.id} | "
                            f"SL={pos.stop_loss:.5f} | Low={low_price:.5f} | "
                            f"Fermeture à close={close_price:.5f}"
                        )
                
                elif is_short:
                    # SHORT: SL atteint si HIGH >= SL
                    sl_hit = high_price >= pos.stop_loss
                    if sl_hit:
                        logging.info(
                            f"[SIM] [{time}] 🛑 SL HIT SHORT | Pos {pos.id} | "
                            f"SL={pos.stop_loss:.5f} | High={high_price:.5f} | "
                            f"Fermeture à close={close_price:.5f}"
                        )
            
            # Fermeture au prix de clôture (approximation réaliste)
            if tp_hit:
                positions_to_close.append((pos.id, close_price, 'tp'))
            elif sl_hit:
                positions_to_close.append((pos.id, close_price, 'sl'))

        # ========== FERMETURE DES POSITIONS ==========
        for pos_id, price, reason in positions_to_close:
            self.broker.close_position(
                pos_id,
                price,
                reason=reason,
                time=time,
                current_prices=current_prices
            )
            
            # Notification au robot
            for strat in self.strategies:
                if hasattr(strat, 'on_position_closed'):
                    try:
                        strat.on_position_closed(pos_id, time, reason)
                    except Exception as e:
                        logging.error(f"[SIM] Erreur on_position_closed {strat.robot_id}: {e}")
