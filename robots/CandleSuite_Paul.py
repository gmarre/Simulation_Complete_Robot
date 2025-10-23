"""CandleSuite_Paul - Grid Mean Reversion / Trend Following

Stratégie configurable via paramètre 'inversion':

**Mode Mean Reversion (inversion=True, par défaut):**
- LONG: après suite ROUGE + cassure SOUS plus bas (achat sur survente)
- SHORT: après suite VERTE + cassure AU-DESSUS plus haut (vente sur surachat)
→ Paris sur REBOND (retour à la moyenne)

**Mode Trend Following (inversion=False):**
- LONG: après suite VERTE + cassure AU-DESSUS plus haut (achat momentum)
- SHORT: après suite ROUGE + cassure SOUS plus bas (vente momentum)
→ Paris sur CONTINUATION (suivre la tendance)

Grid: moyennage dans la direction adverse (pas de SL, philosophie grid pure).
TP commun: profit initial réparti sur toutes les couches.

Gestion des lots dynamique:
 - inp_lot_for_10k définit la taille de position pour 10 000€ de capital
 - Calcul automatique: lots = (balance / 10000) * inp_lot_for_10k
"""

from typing import List, Dict, Optional
import math
import logging
try:
    from engine.strategy_base import StrategyBase
    from engine.broker import Order
except ModuleNotFoundError:
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    from engine.strategy_base import StrategyBase
    from engine.broker import Order
    
logging.basicConfig(level=logging.CRITICAL) 

class CandleSuitePaul(StrategyBase):
    def __init__(self, robot_id: str, symbol: str, timeframe: str = 'm1',
                 inp_suite: int = 3, inp_xtrem_research: int = 200,
                 atr_period: int = 200, inp_tp: float = 1.5, 
                 inp_lot_for_10k: float = 0.1,
                 inp_distance_between_orders: float = 1.0, inp_grid_recov_factor: float = 2.0,
                 close_on_common_tp: bool = True, max_grid_levels: int = 0, 
                 inversion: bool = True,
                 debug: bool = False):
        super().__init__(robot_id, symbol, timeframe=timeframe)
        self.inp_suite = inp_suite
        self.inp_xtrem_research = inp_xtrem_research
        self.atr_period = atr_period
        self.inp_tp = inp_tp
        self.inp_lot_for_10k = inp_lot_for_10k
        self.inp_distance_between_orders = inp_distance_between_orders
        self.inp_grid_recov_factor = inp_grid_recov_factor
        self.close_on_common_tp = close_on_common_tp
        self.max_grid_levels = max_grid_levels
        self.inversion = inversion
        self.debug = debug

        mode = "MEAN REVERSION (contre-tendance)" if self.inversion else "TREND FOLLOWING (continuation)"
        logging.info(f"[{self.robot_id}] 🎯 Mode stratégie: {mode}")

        self._buffer_max_size = max(self.atr_period, self.inp_xtrem_research) + self.inp_suite + 10
        self.buffer: List[Dict] = []
        
        # État LONG
        self.long_positions: List[Dict] = []
        self.long_atr_first: Optional[float] = None
        # État SHORT
        self.short_positions: List[Dict] = []
        self.short_atr_first: Optional[float] = None

        # ========== FLAGS POUR ATTENDRE PROCHAINE BOUGIE ==========
        self.entry_long_pending = False
        self.entry_short_pending = False
        self.grid_long_pending = False
        self.grid_short_pending = False
        
        # ========== STOCKAGE DU PRIX D'ENTRÉE VOULU ==========
        # On stocke le prix OPEN de la bougie suivante pour exécution
        self.pending_entry_price: Optional[float] = None
        # ======================================================

        self._warmup_logged = False

    def _log(self, msg: str):
        if self.debug:
            print(f"[{self.robot_id}] {msg}")

    def _get_current_balance(self) -> float:
        if not self._broker:
            return 10000.0
        return self._broker.balance

    def _calc_base_lots(self) -> float:
        balance = self._get_current_balance()
        lots = (balance / 10000.0) * self.inp_lot_for_10k
        lots = math.ceil(lots * 100) / 100
        if lots < 0.01:
            lots = 0.01
        if self.debug:
            logging.debug(f"[{self.robot_id}] Balance={balance:.2f} => base_lots={lots:.2f}")
        return lots

    def _calc_atr(self) -> Optional[float]:
        if len(self.buffer) < 2:
            return None
        trs = []
        for i in range(1, len(self.buffer)):
            h = self.buffer[i]['high']
            l = self.buffer[i]['low']
            pc = self.buffer[i-1]['close']
            tr = max(h - l, abs(h - pc), abs(l - pc))
            trs.append(tr)
        if len(trs) < self.atr_period:
            window = trs
        else:
            window = trs[-self.atr_period:]
        if not window:
            return None
        return sum(window) / len(window)

    def _check_entry_long(self) -> bool:
        """
        Logique LONG adaptative selon le mode:
        
        Mode Mean Reversion (inversion=True):
          - Suite ROUGE + cassure SOUS plus bas => ACHAT (rebond attendu)
        
        Mode Trend Following (inversion=False):
          - Suite VERTE + cassure AU-DESSUS plus haut => ACHAT (continuation haussière)
        """
        if len(self.buffer) < max(self.inp_suite, self.inp_xtrem_research) + 1:
            return False
        
        current_time = self.buffer[-1]['time']
        lastN = self.buffer[-self.inp_suite:]
        
        # ========== DÉTECTION SELON MODE ==========
        if self.inversion:
            # Mean Reversion: suite ROUGE (baisse)
            suite_ok = all(bar['close'] < bar['open'] for bar in lastN)
            suite_type = "ROUGE (baissière)"
            
            if suite_ok:
                logging.info(f"[{self.robot_id}] [{current_time}] ✅ Suite {suite_type}: {self.inp_suite} bougies consécutives")
            else:
                logging.info(f"[{self.robot_id}] [{current_time}] ❌ Suite {suite_type} NON remplie")
                return False
            
            # Cassure sous le plus bas
            recent = self.buffer[-(self.inp_xtrem_research + 2):-2]
            curr_close = self.buffer[-1]['close']
            low_min = min(b['low'] for b in recent)
            is_breakout = curr_close < low_min

            # Afficher la période de recherche
            start_date = recent[0]['time']
            end_date = recent[-1]['time']
            nb_periods = len(recent)
            logging.info(f"[{self.robot_id}] [{current_time}] 🔍 Recherche cassure BASSE: {start_date} → {end_date} ({nb_periods} périodes) | low_min={low_min:.5f}")
            
            # Afficher la période de recherche en date de la cassure
            if is_breakout:
                logging.info(f"[{self.robot_id}] [{current_time}] ✅ LONG Mean Reversion: close={curr_close:.5f} < low_min={low_min:.5f} → ACHAT SUR SURVENTE")
            else:
                logging.info(f"[{self.robot_id}] [{current_time}] ❌ Cassure basse NON remplie: {curr_close:.5f} >= {low_min:.5f} : {start_date} → {end_date} ({nb_periods} périodes)")
            
            return is_breakout
        
        else:
            # Trend Following: suite VERTE (hausse)
            suite_ok = all(bar['close'] > bar['open'] for bar in lastN)
            suite_type = "VERTE (haussière)"
            
            if suite_ok:
                logging.info(f"[{self.robot_id}] [{current_time}] ✅ Suite {suite_type}: {self.inp_suite} bougies consécutives")
            else:
                logging.info(f"[{self.robot_id}] [{current_time}] ❌ Suite {suite_type} NON remplie")
                return False
            
            # Cassure au-dessus du plus haut
            recent = self.buffer[-(self.inp_xtrem_research + 2):-2]
            curr_close = self.buffer[-1]['close']
            high_max = max(b['high'] for b in recent)
            is_breakout = curr_close > high_max

            # Afficher la période de recherche
            start_date = recent[0]['time']
            end_date = recent[-1]['time']
            nb_periods = len(recent)
            logging.info(f"[{self.robot_id}] [{current_time}] 🔍 Recherche cassure HAUTE: {start_date} → {end_date} ({nb_periods} périodes) | high_max={high_max:.5f}")

            if is_breakout:
                logging.info(f"[{self.robot_id}] [{current_time}] ✅ LONG Trend Following: close={curr_close:.5f} > high_max={high_max:.5f} → ACHAT MOMENTUM")
            else:
                logging.info(f"[{self.robot_id}] [{current_time}] ❌ Cassure haute NON remplie: {curr_close:.5f} <= {high_max:.5f} : {start_date} → {end_date} ({nb_periods} périodes)")
            
            return is_breakout
        # ==========================================

    def _check_entry_short(self) -> bool:
        """
        Logique SHORT adaptative selon le mode:
        
        Mode Mean Reversion (inversion=True):
          - Suite VERTE + cassure AU-DESSUS plus haut => VENTE (rebond attendu)
        
        Mode Trend Following (inversion=False):
          - Suite ROUGE + cassure SOUS plus bas => VENTE (continuation baissière)
        """
        if len(self.buffer) < max(self.inp_suite, self.inp_xtrem_research) + 1:
            return False
        
        current_time = self.buffer[-1]['time']
        lastN = self.buffer[-self.inp_suite:]
        
        # ========== DÉTECTION SELON MODE ==========
        if self.inversion:
            # Mean Reversion: suite VERTE (hausse)
            suite_ok = all(bar['close'] > bar['open'] for bar in lastN)
            suite_type = "VERTE (haussière)"
            
            if suite_ok:
                logging.info(f"[{self.robot_id}] [{current_time}] ✅ Suite {suite_type}: {self.inp_suite} bougies consécutives")
            else:
                logging.info(f"[{self.robot_id}] [{current_time}] ❌ Suite {suite_type} NON remplie")
                return False
            
            # Cassure au-dessus du plus haut
            recent = self.buffer[-(self.inp_xtrem_research + 2):-2]
            curr_close = self.buffer[-1]['close']
            high_max = max(b['high'] for b in recent)
            is_breakout = curr_close > high_max

            # Afficher la période de recherche
            start_date = recent[0]['time']
            end_date = recent[-1]['time']
            nb_periods = len(recent)
            logging.info(f"[{self.robot_id}] [{current_time}] 🔍 Recherche cassure HAUTE: {start_date} → {end_date} ({nb_periods} périodes) | high_max={high_max:.5f}")

            if is_breakout:
                logging.info(f"[{self.robot_id}] [{current_time}] ✅ SHORT Mean Reversion: close={curr_close:.5f} > high_max={high_max:.5f} → VENTE SUR SURACHAT")
            else:
                logging.info(f"[{self.robot_id}] [{current_time}] ❌ Cassure haute NON remplie: {curr_close:.5f} <= {high_max:.5f} : {start_date} → {end_date} ({nb_periods} périodes)")

            return is_breakout
        
        else:
            # Trend Following: suite ROUGE (baisse)
            suite_ok = all(bar['close'] < bar['open'] for bar in lastN)
            suite_type = "ROUGE (baissière)"
            
            if suite_ok:
                logging.info(f"[{self.robot_id}] [{current_time}] ✅ Suite {suite_type}: {self.inp_suite} bougies consécutives")
            else:
                logging.info(f"[{self.robot_id}] [{current_time}] ❌ Suite {suite_type} NON remplie")
                return False
            
            # Cassure sous le plus bas
            recent = self.buffer[-(self.inp_xtrem_research + 2):-2]
            curr_close = self.buffer[-1]['close']
            low_min = min(b['low'] for b in recent)
            is_breakout = curr_close < low_min

            # Afficher la période de recherche
            start_date = recent[0]['time']
            end_date = recent[-1]['time']
            nb_periods = len(recent)
            logging.info(f"[{self.robot_id}] [{current_time}] 🔍 Recherche cassure BASSE: {start_date} → {end_date} ({nb_periods} périodes) | low_min={low_min:.5f}")

            
            if is_breakout:
                logging.info(f"[{self.robot_id}] [{current_time}] ✅ SHORT Trend Following: close={curr_close:.5f} < low_min={low_min:.5f} → VENTE MOMENTUM")
            else:
                logging.info(f"[{self.robot_id}] [{current_time}] ❌ Cassure basse NON remplie: {curr_close:.5f} >= {low_min:.5f} : {start_date} → {end_date} ({nb_periods} périodes)")
            
            return is_breakout
        # ==========================================

    def _grid_can_add(self, side: str, price: float, atr: float) -> bool:
        if side == 'LONG':
            if not self.long_positions:
                return False
            last_entry = self.long_positions[-1]['entry']
            adverse = (last_entry - price) / atr if atr and atr > 0 else 0
            if adverse >= self.inp_distance_between_orders:
                if self.max_grid_levels > 0 and len(self.long_positions) >= self.max_grid_levels:
                    return False
                return True
        else:
            if not self.short_positions:
                return False
            last_entry = self.short_positions[-1]['entry']
            adverse = (price - last_entry) / atr if atr and atr > 0 else 0
            if adverse >= self.inp_distance_between_orders:
                if self.max_grid_levels > 0 and len(self.short_positions) >= self.max_grid_levels:
                    return False
                return True
        return False

    def _calc_next_lots(self, side: str) -> float:
        base_lots = self._calc_base_lots()
        
        if side == 'LONG':
            n = len(self.long_positions)
        else:
            n = len(self.short_positions)
        
        lots = base_lots * (self.inp_grid_recov_factor ** n)
        lots = round(lots, 2)
        
        if lots < 0.01:
            lots = 0.01
        
        return lots

    def _apply_common_tp(self, side: str):
        if not self._broker or not self.close_on_common_tp:
            return
        
        if side == 'LONG':
            if not self.long_positions or self.long_atr_first is None:
                return
            first_lots = self.long_positions[0]['lots']
            target_profit = first_lots * self.long_atr_first * self.inp_tp
            total_lots = sum(p['lots'] for p in self.long_positions)
            numerator = target_profit + sum(p['lots'] * p['entry'] for p in self.long_positions)
            tp = numerator / total_lots
            for p in self.long_positions:
                self._broker.update_take_profit(p['id'], tp)
            logging.debug(f"[{self.robot_id}] Recalc TP commun LONG -> {tp:.5f} niveaux={len(self.long_positions)}")
        else:
            if not self.short_positions or self.short_atr_first is None:
                return
            first_lots = self.short_positions[0]['lots']
            target_profit = first_lots * self.short_atr_first * self.inp_tp
            total_lots = sum(p['lots'] for p in self.short_positions)
            numerator = sum(p['lots'] * p['entry'] for p in self.short_positions) - target_profit
            tp = numerator / total_lots
            for p in self.short_positions:
                self._broker.update_take_profit(p['id'], tp)
            logging.debug(f"[{self.robot_id}] Recalc TP commun SHORT -> {tp:.5f} niveaux={len(self.short_positions)}")

    def _set_individual_tp(self, pos_id: int, side: str, entry: float, atr: float):
        if not self._broker:
            return
        if atr is None:
            return
        if side == 'LONG':
            tp = entry + atr * self.inp_tp
        else:
            tp = entry - atr * self.inp_tp
        self._broker.update_take_profit(pos_id, tp)

    def on_bar(self, time, data_slice: Dict[str, any]):
        if self.symbol not in data_slice:
            return []
        row = data_slice[self.symbol]
        bar = {
            'time': time,
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close'])
        }
        self.buffer.append(bar)
        
        if len(self.buffer) > self._buffer_max_size:
            self.buffer.pop(0)
        
        atr = self._calc_atr()
        
        warmup_required = max(self.atr_period, self.inp_xtrem_research) + 1
        
        if len(self.buffer) < warmup_required:
            if not self._warmup_logged:
                atr_str = f"{atr:.5f}" if atr is not None else "N/A"
                mode = "MEAN REVERSION" if self.inversion else "TREND FOLLOWING"
                logging.info(f"[{self.robot_id}] [{time}] ⏳ WARM-UP ({mode}): {len(self.buffer)}/{warmup_required} barres (ATR={atr_str})")
                self._warmup_logged = True
            return []
        
        if self._warmup_logged and len(self.buffer) == warmup_required:
            mode = "MEAN REVERSION (contre-tendance)" if self.inversion else "TREND FOLLOWING (continuation)"
            logging.info(f"[{self.robot_id}] [{time}] ✅ WARM-UP TERMINÉ | Mode: {mode}")
            self._warmup_logged = False
        
        if atr is None or atr <= 0:
            logging.warning(f"[{self.robot_id}] [{time}] ⚠️ ATR invalide ({atr}), skip")
            return []
        
        orders: List[Order] = []

        # ========== GESTION LONG ==========
        # 1. Exécuter entrée LONG en attente
        if self.entry_long_pending:
            entry_price = bar['open']  # ← PRIX = OPEN de la bougie d'exécution
            mode_desc = "mean reversion (survente)" if self.inversion else "trend following (momentum)"
            logging.info(f"[{self.robot_id}] [{time}] ✅ ENTRÉE LONG EXÉCUTÉE (prochaine bougie) | {mode_desc} | Prix={entry_price:.5f}")
            lots = self._calc_next_lots('LONG')
            
            # ========== STOCKER LE PRIX POUR LE SIMULATEUR ==========
            self.pending_entry_price = entry_price
            # ========================================================
            
            orders.append(Order(self.robot_id, self.symbol, 'BUY', lots, take_profit=None))
            self.entry_long_pending = False
        
        # 2. Exécuter grid LONG en attente
        elif self.grid_long_pending:
            entry_price = bar['open']  # ← PRIX = OPEN de la bougie d'exécution
            logging.info(f"[{self.robot_id}] [{time}] ✅ GRID LONG EXÉCUTÉ (prochaine bougie) | Prix={entry_price:.5f}")
            lots = self._calc_next_lots('LONG')
            
            # ========== STOCKER LE PRIX POUR LE SIMULATEUR ==========
            self.pending_entry_price = entry_price
            # ========================================================
            
            orders.append(Order(self.robot_id, self.symbol, 'BUY', lots, take_profit=None))
            self.grid_long_pending = False
        
        # 3. Vérifier nouveau signal d'entrée LONG
        elif not self.long_positions and self._check_entry_long():
            mode_desc = "mean reversion (survente)" if self.inversion else "trend following (momentum)"
            current_close = bar['close']
            logging.info(f"[{self.robot_id}] [{time}] 🔔 SIGNAL LONG DÉTECTÉ ({mode_desc}) | Prix={current_close:.5f} | ATR={atr:.5f} | ATTENTE PROCHAINE BOUGIE")
            self.entry_long_pending = True
        
        # 4. Vérifier nouveau signal de grid LONG
        elif self.long_positions and self._grid_can_add('LONG', bar['close'], atr):
            last_entry = self.long_positions[-1]['entry']
            adverse_dist = (last_entry - bar['close']) / atr
            logging.info(f"[{self.robot_id}] [{time}] 🔔 GRID ACHETEUR DÉTECTÉ | Niveau {len(self.long_positions)+1} | Distance={adverse_dist:.2f} ATR | ATTENTE PROCHAINE BOUGIE")
            self.grid_long_pending = True

        # ========== GESTION SHORT ==========
        # 1. Exécuter entrée SHORT en attente
        if self.entry_short_pending:
            entry_price = bar['open']  # ← PRIX = OPEN de la bougie d'exécution
            mode_desc = "mean reversion (surachat)" if self.inversion else "trend following (momentum)"
            logging.info(f"[{self.robot_id}] [{time}] ✅ ENTRÉE SHORT EXÉCUTÉE (prochaine bougie) | {mode_desc} | Prix={entry_price:.5f}")
            lots = self._calc_next_lots('SHORT')
            
            # ========== STOCKER LE PRIX POUR LE SIMULATEUR ==========
            self.pending_entry_price = entry_price
            # ========================================================
            
            orders.append(Order(self.robot_id, self.symbol, 'SELL', lots, take_profit=None))
            self.entry_short_pending = False
        
        # 2. Exécuter grid SHORT en attente
        elif self.grid_short_pending:
            entry_price = bar['open']  # ← PRIX = OPEN de la bougie d'exécution
            logging.info(f"[{self.robot_id}] [{time}] ✅ GRID SHORT EXÉCUTÉ (prochaine bougie) | Prix={entry_price:.5f}")
            lots = self._calc_next_lots('SHORT')
            
            # ========== STOCKER LE PRIX POUR LE SIMULATEUR ==========
            self.pending_entry_price = entry_price
            # ========================================================
            
            orders.append(Order(self.robot_id, self.symbol, 'SELL', lots, take_profit=None))
            self.grid_short_pending = False
        
        # 3. Vérifier nouveau signal d'entrée SHORT
        elif not self.short_positions and self._check_entry_short():
            mode_desc = "mean reversion (surachat)" if self.inversion else "trend following (momentum)"
            current_close = bar['close']
            logging.info(f"[{self.robot_id}] [{time}] 🔔 SIGNAL SHORT DÉTECTÉ ({mode_desc}) | Prix={current_close:.5f} | ATR={atr:.5f} | ATTENTE PROCHAINE BOUGIE")
            self.entry_short_pending = True
        
        # 4. Vérifier nouveau signal de grid SHORT
        elif self.short_positions and self._grid_can_add('SHORT', bar['close'], atr):
            last_entry = self.short_positions[-1]['entry']
            adverse_dist = (bar['close'] - last_entry) / atr
            logging.info(f"[{self.robot_id}] [{time}] 🔔 GRID VENDEUR DÉTECTÉ | Niveau {len(self.short_positions)+1} | Distance={adverse_dist:.2f} ATR | ATTENTE PROCHAINE BOUGIE")
            self.grid_short_pending = True

        self._current_atr_for_assign = atr
        return orders

    def on_position_opened(self, position_id: int, time):
        """Callback appelé quand une position est ouverte."""
        if not self._broker:
            return
        
        # ========== RÉCUPÉRATION POSITION DEPUIS LISTE ==========
        # broker.positions est une LISTE d'objets Position
        pos_obj = None
        for p in self._broker.positions:
            if p.id == position_id:
                pos_obj = p
                break
    
        if not pos_obj:
            logging.warning(f"[{self.robot_id}] [{time}] ⚠️ Position {position_id} introuvable")
            return
    
        # Convertir l'objet Position en dict pour uniformité
        pos = {
            'id': pos_obj.id,
            'side': pos_obj.side,
            'price': pos_obj.entry_price,
            'lots': pos_obj.lots
        }
        # ========================================================
    
        atr = getattr(self, '_current_atr_for_assign', None)
    
        # ========== GESTION LONG ==========
        if pos['side'] == 'LONG':
            self.long_positions.append({
                'id': pos['id'],
                'entry': pos['price'],
                'lots': pos['lots']
            })
            
            # Stocker ATR de la première position
            if len(self.long_positions) == 1:
                if self.long_atr_first is None and atr is not None:
                    self.long_atr_first = atr
                    if self.debug:
                        logging.debug(f"[{self.robot_id}] LONG: ATR référence = {atr:.5f}")
        
            # Recalculer TP commun ou individuel
            if self.close_on_common_tp:
                self._apply_common_tp('LONG')
            else:
                if atr:
                    self._set_individual_tp(pos['id'], 'LONG', pos['price'], atr)
    
        # ========== GESTION SHORT ==========
        else:
            self.short_positions.append({
                'id': pos['id'],
                'entry': pos['price'],
                'lots': pos['lots']
            })
            
            # Stocker ATR de la première position
            if len(self.short_positions) == 1:
                if self.short_atr_first is None and atr is not None:
                    self.short_atr_first = atr
                    if self.debug:
                        logging.debug(f"[{self.robot_id}] SHORT: ATR référence = {atr:.5f}")
        
            # Recalculer TP commun ou individuel
            if self.close_on_common_tp:
                self._apply_common_tp('SHORT')
            else:
                if atr:
                    self._set_individual_tp(pos['id'], 'SHORT', pos['price'], atr)

    def on_position_closed(self, position_id: int, time, reason: str):
        logging.info(f"[{self.robot_id}] [{time}] 🔒 Position fermée id={position_id} reason={reason}")
        self.long_positions = [p for p in self.long_positions if p['id'] != position_id]
        self.short_positions = [p for p in self.short_positions if p['id'] != position_id]
        
        if not self.long_positions:
            self.long_atr_first = None
            self.entry_long_pending = False   # ← Reset flag si plus de positions
            self.grid_long_pending = False    # ← Reset flag si plus de positions
        
        if not self.short_positions:
            self.short_atr_first = None
            self.entry_short_pending = False  # ← Reset flag si plus de positions
            self.grid_short_pending = False   # ← Reset flag si plus de positions

    def get_warmup_periods(self) -> int:
        return max(self.atr_period, self.inp_xtrem_research) + self.inp_suite + 10

