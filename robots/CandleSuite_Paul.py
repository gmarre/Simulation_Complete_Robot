"""CandleSuite_Paul - Grid Mean Reversion

Stratégie de retour à la moyenne (range/counter-trend):
- Entre à contre-tendance après un mouvement extrême (anticipation de rebond).
- Grid ajouté si continuation adverse (moyenner à la baisse/hausse).

Entrées initiales déclenchées par:
 - Suite de bougies de MÊME COULEUR => mouvement unidirectionnel fort
 - Cassure d'un extrême => sur-extension
 ⚠️ LOGIQUE MEAN REVERSION:
   - LONG: après suite ROUGE + cassure SOUS plus bas (achat sur survente)
   - SHORT: après suite VERTE + cassure AU-DESSUS plus haut (vente sur surachat)

Grid: moyennage à contre-tendance (pas de SL, philosophie grid pure).
TP commun: profit initial réparti sur toutes les couches.

Gestion des lots dynamique:
 - inp_lot_for_10k définit la taille de position pour 10 000€ de capital
 - Calcul automatique: lots = (balance / 10000) * inp_lot_for_10k
 - Exemple: inp_lot_for_10k=0.1, balance=5000 => lots=0.05
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
                 inp_lot_for_10k: float = 0.1,  # ← Nouveau paramètre
                 inp_distance_between_orders: float = 1.0, inp_grid_recov_factor: float = 2.0,
                 close_on_common_tp: bool = True, max_grid_levels: int = 0, debug: bool = False):
        super().__init__(robot_id, symbol, timeframe=timeframe)
        self.inp_suite = inp_suite
        self.inp_xtrem_research = inp_xtrem_research
        self.atr_period = atr_period
        self.inp_tp = inp_tp
        self.inp_lot_for_10k = inp_lot_for_10k  # Référence pour 10k€
        self.inp_distance_between_orders = inp_distance_between_orders
        self.inp_grid_recov_factor = inp_grid_recov_factor
        self.close_on_common_tp = close_on_common_tp
        self.max_grid_levels = max_grid_levels
        self.debug = debug

        # Buffer optimisé (rotation automatique)
        self._buffer_max_size = max(self.atr_period, self.inp_xtrem_research) + self.inp_suite + 10
        self.buffer: List[Dict] = []
        
        # Etat LONG
        self.long_positions: List[Dict] = []
        self.long_atr_first: Optional[float] = None
        # Etat SHORT
        self.short_positions: List[Dict] = []
        self.short_atr_first: Optional[float] = None

        self._warmup_logged = False  # Flag pour logger 1 seule fois

    def _log(self, msg: str):
        if self.debug:
            print(f"[{self.robot_id}] {msg}")

    def _get_current_balance(self) -> float:
        """Récupère le balance actuel depuis le broker."""
        if not self._broker:
            return 10000.0  # fallback si pas de broker
        return self._broker.balance

    def _calc_base_lots(self) -> float:
        """
        Calcule la taille de position de base proportionnelle au capital.
        Formule: lots = (balance / 10000) * inp_lot_for_10k
        
        Exemples:
        - balance=5000, inp_lot_for_10k=0.1 => 0.05 lot
        - balance=10000, inp_lot_for_10k=0.1 => 0.1 lot
        - balance=20000, inp_lot_for_10k=0.1 => 0.2 lot
        """
        balance = self._get_current_balance()
        lots = (balance / 10000.0) * self.inp_lot_for_10k
        
        # Arrondi à 2 décimales (standard forex)
        lots = round(lots, 2)
        
        # Minimum 0.01 lot (micro-lot)
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
        """LONG: achat sur survente après suite rouge + nouveau plus bas cassé."""
        if len(self.buffer) < max(self.inp_suite, self.inp_xtrem_research) + 1:
            return False
        
        current_time = self.buffer[-1]['time']
        
        # Suite rouge = close < open
        lastN = self.buffer[-self.inp_suite:]
        suite_ok = all(bar['close'] < bar['open'] for bar in lastN)
        
        if suite_ok:
            logging.info(f"[{self.robot_id}] [{current_time}] ✅ Condition SUITE ROUGE remplie: {self.inp_suite} bougies baissières consécutives")
        else:
            logging.info(f"[{self.robot_id}] [{current_time}] ❌ Condition SUITE ROUGE NON remplie: bougies mixtes détectées")
            return False
        
        # Cassure sous le plus bas des N précédentes (exclure barre actuelle)
        recent = self.buffer[-(self.inp_xtrem_research + 1):-1]
        curr_close = self.buffer[-1]['close']
        low_min = min(b['low'] for b in recent)
        
        is_breakout = curr_close < low_min
        
        if is_breakout:
            low_bar = min(recent, key=lambda b: b['low'])
            time_diff = current_time - low_bar['time']
            logging.info(f"[{self.robot_id}] [{current_time}] ✅ Condition CASSURE EXTRÊME remplie: close={curr_close:.5f} < low_min={low_min:.5f} (barre extrême à {low_bar['time']}, delta={time_diff})")
        else:
            logging.info(f"[{self.robot_id}] [{current_time}] ❌ Cassure extrême NON remplie: close={curr_close:.5f} >= low_min={low_min:.5f}")
        
        return is_breakout

    def _check_entry_short(self) -> bool:
        """SHORT: vente sur surachat après suite verte + nouveau plus haut cassé."""
        if len(self.buffer) < max(self.inp_suite, self.inp_xtrem_research) + 1:
            return False
        
        current_time = self.buffer[-1]['time']
        
        # Suite verte = close > open
        lastN = self.buffer[-self.inp_suite:]
        suite_ok = all(bar['close'] > bar['open'] for bar in lastN)
        
        if suite_ok:
            logging.info(f"[{self.robot_id}] [{current_time}] ✅ Condition SUITE VERTE remplie: {self.inp_suite} bougies haussières consécutives")
        else:
            logging.info(f"[{self.robot_id}] [{current_time}] ❌ Condition SUITE VERTE NON remplie: bougies mixtes détectées")
            return False
        
        recent = self.buffer[-(self.inp_xtrem_research + 1):-1]
        curr_close = self.buffer[-1]['close']
        high_max = max(b['high'] for b in recent)
        
        is_breakout = curr_close > high_max
        
        if is_breakout:
            high_bar = max(recent, key=lambda b: b['high'])
            time_diff = current_time - high_bar['time']
            logging.info(f"[{self.robot_id}] [{current_time}] ✅ Condition CASSURE EXTRÊME remplie: close={curr_close:.5f} > high_max={high_max:.5f} (barre extrême à {high_bar['time']}, delta={time_diff})")
        else:
            logging.info(f"[{self.robot_id}] [{current_time}] ❌ Cassure extrême NON remplie: close={curr_close:.5f} <= high_max={high_max:.5f}")
        
        return is_breakout

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
        """
        Calcule la taille du prochain ordre (avec facteur multiplicatif grid).
        - Niveau 0 (premier): base_lots (dynamique selon balance)
        - Niveau n: base_lots * (inp_grid_recov_factor ** n)
        """
        base_lots = self._calc_base_lots()  # ← Dynamique selon balance actuel
        
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
            # Profit cible = premier lot * ATR_first * inp_tp
            first_lots = self.long_positions[0]['lots']
            target_profit = first_lots * self.long_atr_first * self.inp_tp
            total_lots = sum(p['lots'] for p in self.long_positions)
            numerator = target_profit + sum(p['lots'] * p['entry'] for p in self.long_positions)
            tp = numerator / total_lots
            for p in self.long_positions:
                self._broker.update_take_profit(p['id'], tp)
            logging.debug(f"[{self.robot_id}] Recalc TP commun LONG -> {tp:.5f} niveaux={len(self.long_positions)} targetProfit={target_profit:.5f}")
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
            logging.debug(f"[{self.robot_id}] Recalc TP commun SHORT -> {tp:.5f} niveaux={len(self.short_positions)} targetProfit={target_profit:.5f}")

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
        
        # Rotation buffer
        if len(self.buffer) > self._buffer_max_size:
            self.buffer.pop(0)
        
        atr = self._calc_atr()
        
        # ========== LOGS WARM-UP EXPLICITES ==========
        warmup_required = max(self.atr_period, self.inp_xtrem_research) + 1
        
        if len(self.buffer) < warmup_required:
            if not self._warmup_logged:
                # ❌ INCORRECT (cause l'erreur)
                # logging.info(f"... (ATR={atr:.5f if atr else 'N/A'})")
                
                # ✅ CORRECT
                atr_str = f"{atr:.5f}" if atr is not None else "N/A"
                logging.info(f"[{self.robot_id}] [{time}] ⏳ WARM-UP en cours: {len(self.buffer)}/{warmup_required} barres chargées (ATR={atr_str})")
                self._warmup_logged = True
            return []
        
        # Log une fois quand warm-up terminé
        if self._warmup_logged and len(self.buffer) == warmup_required:
            logging.info(f"[{self.robot_id}] [{time}] ✅ WARM-UP TERMINÉ: {warmup_required} barres disponibles, stratégie ACTIVE")
            self._warmup_logged = False  # Reset pour ne pas re-logger
        
        # Skip si ATR invalide (ne devrait plus arriver après warm-up)
        if atr is None or atr <= 0:
            logging.warning(f"[{self.robot_id}] [{time}] ⚠️ ATR invalide ({atr}), skip barre")
            return []
        # ========== FIN LOGS WARM-UP ==========
        
        orders: List[Order] = []
        price = bar['close']

        # LONG side (mean reversion: achat sur survente)
        if not self.long_positions and self._check_entry_long():
            logging.info(f"[{self.robot_id}] [{time}] 🚀 SIGNAL LONG INITIAL (mean reversion) | Prix={price:.5f} | ATR={atr:.5f}")
            lots = self._calc_next_lots('LONG')
            orders.append(Order(self.robot_id, self.symbol, 'BUY', lots, take_profit=None))
        elif self.long_positions and self._grid_can_add('LONG', price, atr):
            last_entry = self.long_positions[-1]['entry']
            adverse_dist = (last_entry - price) / atr
            logging.info(f"[{self.robot_id}] [{time}] 📊 GRID ACHETEUR | Ajout niveau {len(self.long_positions)+1}/{self.max_grid_levels or '∞'} | Prix={price:.5f} | Distance={adverse_dist:.2f} ATR | Dernier niveau={last_entry:.5f}")
            lots = self._calc_next_lots('LONG')
            orders.append(Order(self.robot_id, self.symbol, 'BUY', lots, take_profit=None))
        elif self.long_positions:
            # Grid actif mais condition d'ajout non remplie
            last_entry = self.long_positions[-1]['entry']
            adverse_dist = (last_entry - price) / atr if atr > 0 else 0
            if adverse_dist > 0:  # Marché adverse mais pas assez
                logging.debug(f"[{self.robot_id}] [{time}] 📊 GRID ACHETEUR actif | Distance={adverse_dist:.2f}/{self.inp_distance_between_orders} ATR | Attente baisse supplémentaire")

        # SHORT side (mean reversion: vente sur surachat)
        if not self.short_positions and self._check_entry_short():
            logging.info(f"[{self.robot_id}] [{time}] 🚀 SIGNAL SHORT INITIAL (mean reversion) | Prix={price:.5f} | ATR={atr:.5f}")
            lots = self._calc_next_lots('SHORT')
            orders.append(Order(self.robot_id, self.symbol, 'SELL', lots, take_profit=None))
        elif self.short_positions and self._grid_can_add('SHORT', price, atr):
            last_entry = self.short_positions[-1]['entry']
            adverse_dist = (price - last_entry) / atr
            logging.info(f"[{self.robot_id}] [{time}] 📊 GRID VENDEUR | Ajout niveau {len(self.short_positions)+1}/{self.max_grid_levels or '∞'} | Prix={price:.5f} | Distance={adverse_dist:.2f} ATR | Dernier niveau={last_entry:.5f}")
            lots = self._calc_next_lots('SHORT')
            orders.append(Order(self.robot_id, self.symbol, 'SELL', lots, take_profit=None))
        elif self.short_positions:
            # Grid actif mais condition d'ajout non remplie
            last_entry = self.short_positions[-1]['entry']
            adverse_dist = (price - last_entry) / atr if atr > 0 else 0
            if adverse_dist > 0:  # Marché adverse mais pas assez
                logging.debug(f"[{self.robot_id}] [{time}] 📊 GRID VENDEUR actif | Distance={adverse_dist:.2f}/{self.inp_distance_between_orders} ATR | Attente hausse supplémentaire")

        self._current_atr_for_assign = atr
        return orders

    def on_position_opened(self, position_id: int, time):
        if not self._broker:
            return
        pos = next((p for p in self._broker.positions if p.id == position_id), None)
        if not pos:
            return
        atr = getattr(self, '_current_atr_for_assign', None)
        
        if pos.side == 'LONG':
            self.long_positions.append({'id': pos.id, 'entry': pos.entry_price, 'lots': pos.lots})
            if len(self.long_positions) == 1:
                if self.long_atr_first is None and atr is not None:
                    self.long_atr_first = atr
            if self.close_on_common_tp:
                self._apply_common_tp('LONG')
            else:
                if atr:
                    self._set_individual_tp(pos.id, 'LONG', pos.entry_price, atr)
        else:
            self.short_positions.append({'id': pos.id, 'entry': pos.entry_price, 'lots': pos.lots})
            if len(self.short_positions) == 1:
                if self.short_atr_first is None and atr is not None:
                    self.short_atr_first = atr
            if self.close_on_common_tp:
                self._apply_common_tp('SHORT')
            else:
                if atr:
                    self._set_individual_tp(pos.id, 'SHORT', pos.entry_price, atr)

    def on_position_closed(self, position_id: int, time, reason: str):
        logging.info(f"[{self.robot_id}] [{time}] 🔒 Position fermée id={position_id} reason={reason}")
        self.long_positions = [p for p in self.long_positions if p['id'] != position_id]
        self.short_positions = [p for p in self.short_positions if p['id'] != position_id]
        if not self.long_positions:
            self.long_atr_first = None
        if not self.short_positions:
            self.short_atr_first = None

    def get_warmup_periods(self) -> int:
        """
        Retourne le nombre de périodes (timeframe du robot) nécessaires avant trading.
        """
        return max(self.atr_period, self.inp_xtrem_research) + self.inp_suite + 10

