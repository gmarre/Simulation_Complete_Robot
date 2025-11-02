from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
import logging

@dataclass
class Order:
    robot_id: str
    symbol: str
    side: str  # 'BUY' ou 'SELL'
    lots: float
    take_profit: Optional[float] = None

@dataclass
class Position:
    id: int
    robot_id: str
    symbol: str
    side: str
    entry_price: float
    lots: float
    open_time: datetime
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None

class CurrencyConverter:
    """
    Gère la conversion des PnL vers EUR en utilisant les taux de change disponibles.
    """
    def __init__(self, account_currency: str = 'EUR'):
        self.account_currency = account_currency
        self.conversion_cache = {}
        
    def parse_symbol(self, symbol: str) -> tuple:
        """
        Extrait la devise de base et la devise de cotation d'un symbole.
        Ex: 'EURGBP' → ('EUR', 'GBP')
        """
        if len(symbol) >= 6:
            base = symbol[:3].upper()
            quote = symbol[3:6].upper()
            return base, quote
        raise ValueError(f"Symbole invalide: {symbol}")
    
    def get_pnl_currency(self, symbol: str) -> str:
        """
        Détermine la devise dans laquelle le PnL est calculé.
        Forex: toujours la devise de cotation (2ème devise).
        """
        _, quote = self.parse_symbol(symbol)
        return quote
    
    def convert_to_account_currency(
        self, 
        pnl_amount: float, 
        symbol: str, 
        current_prices: Dict[str, float],
        time: datetime = None
    ) -> tuple:
        """
        Convertit un PnL de la devise du symbole vers EUR.
        
        Args:
            pnl_amount: Montant du PnL dans la devise de cotation
            symbol: Symbole tradé (ex: 'EURGBP')
            current_prices: Dict des prix actuels {symbol: price}
            time: Timestamp pour logs
        
        Returns:
            (pnl_eur, conversion_rate, conversion_path)
        """
        base, quote = self.parse_symbol(symbol)
        
        # Cas 1: PnL déjà en EUR
        if quote == self.account_currency:
            return pnl_amount, 1.0, f"{quote}→{self.account_currency} (direct)"
        
        # Cas 2: Symbole contient EUR en devise de base (EURGBP, EURUSD)
        if base == self.account_currency:
            # EURGBP: PnL en GBP, taux EUR/GBP disponible
            # Conversion: 1 GBP = (1 / EUR/GBP) EUR
            eur_quote_rate = current_prices.get(symbol)
            if eur_quote_rate:
                conversion_rate = 1 / eur_quote_rate
                pnl_eur = pnl_amount * conversion_rate
                return pnl_eur, conversion_rate, f"{quote}→{self.account_currency} (via {symbol})"
        
        # Cas 3: Aucun EUR dans le symbole (GBPNZD, GBPUSD, etc.)
        # → Conversion en 2 étapes
        symbol_rate = current_prices.get(symbol)
        if symbol_rate:
            pnl_in_base = pnl_amount / symbol_rate  # Quote → Base
            
            # Chercher un taux EUR/BASE
            eur_base_symbol = f"{self.account_currency}{base}"  # EURGBP
            
            if eur_base_symbol in current_prices:
                eur_base_rate = current_prices[eur_base_symbol]
                conversion_rate = 1 / eur_base_rate
                pnl_eur = pnl_in_base * conversion_rate
                return pnl_eur, conversion_rate, f"{quote}→{base}→{self.account_currency} (via {symbol} + {eur_base_symbol})"
        
        # Cas 4: Fallback avec taux fixes
        fallback_rates = {
            'USD': 0.92,
            'GBP': 1.15,
            'JPY': 0.0062,
            'CHF': 1.05,
            'NZD': 0.55,
            'AUD': 0.60,
            'CAD': 0.68,
        }
        
        if quote in fallback_rates:
            rate = fallback_rates[quote]
            pnl_eur = pnl_amount * rate
            logging.warning(f"[CONVERTER] Utilisation taux fixe pour {quote}→EUR: {rate:.5f}")
            return pnl_eur, rate, f"{quote}→{self.account_currency} (FALLBACK FIXE)"
        
        # Cas 5: Impossible de convertir
        logging.error(f"[CONVERTER] Impossible de convertir {quote}→{self.account_currency} pour {symbol}")
        return pnl_amount, 1.0, f"{quote}→{self.account_currency} (ERREUR - PAS DE CONVERSION)"


class Broker:
    def __init__(self, starting_balance: float = 10000.0, leverage: int = 100, account_currency: str = 'EUR'):
        self.initial_balance = starting_balance
        self.balance = starting_balance
        self.leverage = leverage
        self.account_currency = account_currency
        self.positions: List[Position] = []
        self.closed_trades: List[Dict] = []
        self.trade_events: List[Dict] = []
        self.next_position_id = 1
        
        # Conversion de devises
        self.converter = CurrencyConverter(account_currency)
        
        # Spread config (en pips)
        self.spread_config = {
            'EURGBP': 0.6,
            'EURUSD': 0.8,
            'GBPUSD': 1.2,
            'GBPNZD': 2.0,
            'USDJPY': 0.5,
            'EURJPY': 0.8,
            'GBPJPY': 1.5,
        }

    def execute(self, order: Order, price: float, time=None) -> int:
        """Exécute un ordre (entrée en position)"""
        pos = Position(
            id=self.next_position_id,
            robot_id=order.robot_id,
            symbol=order.symbol,
            side=order.side,
            entry_price=price,
            lots=order.lots,
            open_time=time or datetime.now(),
            take_profit=order.take_profit,
            stop_loss=None
        )
        
        self.positions.append(pos)
        self.next_position_id += 1
        
        # Log trade event
        self.trade_events.append({
            'time': time or datetime.now(),
            'robot_id': order.robot_id,
            'symbol': order.symbol,
            'action': 'OPEN',
            'side': order.side,
            'price': price,
            'lots': order.lots,
            'position_id': pos.id
        })
        
        logging.debug(f"[BROKER] Position {pos.id} ouverte: {order.side} {order.lots} lots {order.symbol} @ {price:.5f}")
        
        return pos.id

    def close_position(
        self, 
        position_id: int, 
        price: float, 
        reason: str = 'manual', 
        time=None,
        current_prices: Dict[str, float] = None
    ):
        """
        Ferme une position avec conversion automatique vers EUR.
        
        Args:
            current_prices: Dict contenant TOUS les prix nécessaires pour conversion
        """
        pos = next((p for p in self.positions if p.id == position_id), None)
        if not pos:
            return None
        
        if current_prices is None:
            current_prices = {}
        
        # Calcul PnL brut (devise de cotation)
        spread_pips = self.spread_config.get(pos.symbol, 1.0)
        spread_price = spread_pips * 0.00001
        
        # ========== NORMALISATION DU SIDE (CORRIGÉ) ==========
        is_long = pos.side in ('LONG', 'BUY')
        is_short = pos.side in ('SHORT', 'SELL')
        # ====================================================
        
        if is_long:
            exit_price = price  # Vend au Bid
            pnl_quote = (exit_price - pos.entry_price) * pos.lots * 100000
        elif is_short:
            exit_price = price + spread_price  # Achète à l'Ask
            pnl_quote = (pos.entry_price - exit_price) * pos.lots * 100000
        else:
            logging.error(f"[BROKER] Side invalide pour position {pos.id}: {pos.side}")
            pnl_quote = 0.0
            exit_price = price
        
        # Conversion vers EUR
        pnl_eur, conversion_rate, conversion_path = self.converter.convert_to_account_currency(
            pnl_quote, 
            pos.symbol, 
            current_prices,
            time
        )
        
        self.balance += pnl_eur
        
        # Enregistrement trade
        _, quote_currency = self.converter.parse_symbol(pos.symbol)
        
        trade_record = {
            'position_id': pos.id,
            'robot_id': pos.robot_id,
            'symbol': pos.symbol,
            'side': pos.side,
            'lots': pos.lots,
            'entry_price': pos.entry_price,
            'exit_price': exit_price,
            'pnl_eur': pnl_eur,
            'pnl_quote_currency': pnl_quote,
            'quote_currency': quote_currency,
            'conversion_rate': conversion_rate,
            'conversion_path': conversion_path,
            'reason': reason,
            'open_time': pos.open_time,
            'close_time': time or datetime.now()
        }
        
        self.closed_trades.append(trade_record)
        
        # Log trade event
        self.trade_events.append({
            'time': time or datetime.now(),
            'robot_id': pos.robot_id,
            'symbol': pos.symbol,
            'action': 'CLOSE',
            'side': pos.side,
            'price': exit_price,
            'lots': pos.lots,
            'pnl': pnl_eur,
            'position_id': pos.id,
            'reason': reason
        })
        
        self.positions.remove(pos)
        
        # Log détaillé
        logging.info(
            f"[BROKER] Position {pos.id} fermée: {reason} @ {exit_price:.5f} | "
            f"PnL={pnl_eur:.2f}€ (brut={pnl_quote:.2f}{quote_currency}) | "
            f"{conversion_path} (rate={conversion_rate:.5f})"
        )
        
        return trade_record

    def update_take_profit(self, position_id: int, new_tp: Optional[float]):
        """Met à jour le TP d'une position"""
        pos = next((p for p in self.positions if p.id == position_id), None)
        if pos:
            pos.take_profit = new_tp

    def get_balance(self) -> float:
        """Retourne la balance en EUR"""
        return self.balance

    def equity(self, current_prices: Dict[str, float]) -> float:
        """
        Calcul equity = balance + PnL flottant de toutes positions ouvertes (en EUR).
        """
        equity_value = self.balance
        
        for pos in self.positions:
            current_price = current_prices.get(pos.symbol, pos.entry_price)
            
            # PnL flottant en devise de cotation
            spread_pips = self.spread_config.get(pos.symbol, 1.0)
            spread_price = spread_pips * 0.00001
            
            if pos.side == 'LONG':
                unrealized_quote = (current_price - pos.entry_price) * pos.lots * 100000
            else:
                unrealized_quote = (pos.entry_price - (current_price + spread_price)) * pos.lots * 100000
            
            # Conversion vers EUR
            unrealized_eur, _, _ = self.converter.convert_to_account_currency(
                unrealized_quote,
                pos.symbol,
                current_prices
            )
            
            equity_value += unrealized_eur
        
        return equity_value

    def unrealized_pnl(self, current_prices: Dict[str, float]) -> float:
        """Retourne le PnL non réalisé total en EUR"""
        return self.equity(current_prices) - self.balance

    def margin_used(self, current_prices: Dict[str, float]) -> float:
        """Calcule la marge utilisée"""
        if self.leverage <= 0:
            return 0.0
        
        margin = 0.0
        for pos in self.positions:
            price = current_prices.get(pos.symbol, pos.entry_price)
            notional = price * 100000 * pos.lots  # contract_size = 100,000
            margin += notional / self.leverage
        
        return margin

    def lots_open(self) -> float:
        """Retourne le total de lots ouverts"""
        return sum(p.lots for p in self.positions)

    def lots_by_robot(self) -> Dict[str, float]:
        """Retourne les lots par robot"""
        agg: Dict[str, float] = {}
        for p in self.positions:
            agg[p.robot_id] = agg.get(p.robot_id, 0.0) + p.lots
        return agg

    def find_positions_by_robot_side(self, robot_id: str, side: str) -> List[Position]:
        """Trouve les positions d'un robot pour un côté donné"""
        return [p for p in self.positions if p.robot_id == robot_id and p.side == side]

    def has_positions_direction(self, robot_id: str, side: str) -> bool:
        """Vérifie si un robot a des positions dans une direction"""
        return any(p.robot_id == robot_id and p.side == side for p in self.positions)

    def open_position(self, order: Order, entry_price: float, time):
        """Ouvre une nouvelle position"""
        normalized_side = order.side.upper()
        
        if normalized_side not in ['BUY', 'SELL']:
            raise ValueError(f"Côté de commande invalide: {order.side}")
        
        pos = Position(
            id=self.next_position_id,
            robot_id=order.robot_id,
            symbol=order.symbol,
            side=normalized_side,
            entry_price=entry_price,
            lots=order.lots,
            open_time=time
        )
        
        # ========== DEBUG: VÉRIFIER LE SIDE ==========
        logging.info(
            f"[BROKER] Position {pos.id} créée: "
            f"order.side={order.side} → pos.side={pos.side} "
            f"(normalized={normalized_side})"
        )
        # =============================================
        
        self.positions.append(pos)
        self.next_position_id += 1
        
        # Log trade event
        self.trade_events.append({
            'time': time or datetime.now(),
            'robot_id': order.robot_id,
            'symbol': order.symbol,
            'action': 'OPEN',
            'side': order.side,
            'price': entry_price,
            'lots': order.lots,
            'position_id': pos.id
        })
