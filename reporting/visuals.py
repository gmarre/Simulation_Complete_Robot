import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, List
import logging
from collections import defaultdict
from matplotlib.patches import Rectangle  # déjà utile pour fallback

logging.basicConfig(level=logging.INFO)

try:
    import mplfinance as mpf
    _HAS_MPF = True
    logging.info("mplfinance disponible pour les graphiques en chandelier.")
except ImportError:
    _HAS_MPF = False
    logging.warning("mplfinance non disponible, utilisation des graphiques en ligne.")

def plot_candles(data: pd.DataFrame, title: str = 'Price', mav: Optional[List[int]] = None,
                 volume: bool = False, limit: Optional[int] = 300):
    if data.empty:
        logging.warning("Pas de données (candles).")
        return
    df = data.copy()
    if limit and len(df) > limit:
        df = df.tail(limit)
    dfc = df[['open','high','low','close']].copy()
    dfc.columns = ['Open','High','Low','Close']
    if _HAS_MPF:
        plot_kwargs = dict(type='candle', style='yahoo', title=title)
        if mav:            # seulement si mav non vide
            plot_kwargs['mav'] = mav
        if volume:         # seulement si demandé
            plot_kwargs['volume'] = True
        mpf.plot(dfc, **plot_kwargs)
    else:
        plt.figure(figsize=(11,4))
        plt.plot(dfc.index, dfc['Close'], label='Close', color='black')
        plt.title(title + " (line fallback)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

def plot_equity(results: pd.DataFrame, title: str = 'Equity / Balance',
                robots: Optional[List[str]] = None,
                fixed_scale: bool = True,
                scale_pct: float = 0.20,
                zoom_threshold_frac: float = 0.005):
    """
    fixed_scale: si True => Y principal de 0 à start_balance*(1+scale_pct)
    zoom_threshold_frac: si range Equity < start_balance * threshold -> affiche un sous-graphe zoom
    """
    if results.empty or 'balance' not in results.columns or 'equity' not in results.columns:
        print("Pas de résultats ou colonnes manquantes pour equity.")
        return

    start_balance = float(results['balance'].iloc[0])
    eq_min = results['equity'].min()
    eq_max = results['equity'].max()
    eq_range = eq_max - eq_min

    if fixed_scale:
        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(11,6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) if eq_range < start_balance * zoom_threshold_frac else gridspec.GridSpec(1,1)
        ax_main = fig.add_subplot(gs[0])
    else:
        fig, ax_main = plt.subplots(figsize=(11,5))

    ax_main.plot(results.index, results['balance'], label='Balance', color='tab:blue', linewidth=1.4)
    ax_main.plot(results.index, results['equity'], label='Equity', color='tab:orange', linewidth=1.0)

    if robots:
        for rid in robots:
            col = f'unrealized_pnl_{rid}'
            if col in results.columns:
                ax_main.plot(results.index,
                             results['balance'] + results[col],
                             '--', linewidth=0.8,
                             label=f'Balance+Unrlzd {rid}')

    ax_main.set_title(title)
    ax_main.grid(alpha=0.3)
    ax_main.legend(loc='upper left')
    ax_main.set_ylabel("Account Value")

    if fixed_scale:
        ax_main.set_ylim(0, start_balance * (1.0 + scale_pct))

    # Sous-graphe zoom si variations trop petites
    if fixed_scale and eq_range < start_balance * zoom_threshold_frac:
        ax_zoom = fig.add_subplot(gs[1])
        pct = (results['equity'] - start_balance) / start_balance * 100.0
        ax_zoom.plot(results.index, pct, color='tab:orange', linewidth=0.9, label='% Equity vs Start')
        ax_zoom.axhline(0, color='grey', linewidth=0.7)
        ax_zoom.set_ylabel("% Δ")
        ax_zoom.set_xlabel("Time")
        ax_zoom.grid(alpha=0.3)
        ax_zoom.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

def plot_unrealized(results: pd.DataFrame, robots: Optional[List[str]] = None):
    if results.empty:
        return
    fig, ax = plt.subplots(figsize=(11,4))
    ax.plot(results.index, results['unrealized_pnl'], label='Unrealized Total', color='purple')
    if robots:
        for rid in robots:
            col = f'unrealized_pnl_{rid}'
            if col in results.columns:
                ax.plot(results.index, results[col], label=f'Unreal {rid}', linewidth=1)
    ax.legend()
    ax.set_title("Unrealized PnL")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_lots(results: pd.DataFrame, robots: Optional[List[str]] = None):
    if results.empty:
        return
    fig, ax = plt.subplots(figsize=(11,4))
    ax.plot(results.index, results['lots_open'], label='Lots Total', color='black')
    if robots:
        for rid in robots:
            col = f'lots_{rid}'
            if col in results.columns:
                ax.plot(results.index, results[col], label=rid)
    ax.legend()
    ax.set_title("Lots ouverts")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_margin(results: pd.DataFrame):
    """
    Affiche:
    1. Marge utilisée en EUR (valeur absolue)
    2. Margin Level en % (Equity / Margin Used × 100) - ÉCHELLE LOG
    """
    if results.empty or 'margin_used' not in results.columns:
        logging.warning("Pas de données margin_used pour tracer.")
        return
    
    if 'equity' not in results.columns:
        logging.error("Colonne 'equity' manquante, impossible de calculer Margin Level")
        return
    

    
    sample = results[results['margin_used'] > 0].head(5)
    if not sample.empty:
       
        for idx, row in sample.iterrows():
            ml = (row['equity'] / row['margin_used']) * 100
            

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # ========== GRAPHIQUE 1: Marge Utilisée (EUR) ==========
    ax1.plot(results.index, results['margin_used'], 
             label='Margin Used (EUR)', color='brown', linewidth=1.5)
    ax1.set_ylabel('Margin Used (EUR)', fontsize=11)
    ax1.set_title("Margin Used - Absolute Value", fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.3)
    
    # ========== GRAPHIQUE 2: Margin Level LOG SCALE ==========
    mask_valid = results['margin_used'] > 1e-9
    margin_level = pd.Series(index=results.index, dtype=float)
    
    if mask_valid.any():
        margin_level[mask_valid] = (results.loc[mask_valid, 'equity'] / 
                                     results.loc[mask_valid, 'margin_used']) * 100
        
        valid_ml = margin_level[mask_valid]
       
        
        # ========== CALCUL CAP_VALUE POUR Y-AXIS ==========
        max_ml = valid_ml.max()
        cap_value = min(max_ml * 1.2, 100000)  # 120% du max, ou 100,000% max
        # ==================================================
        
    else:
        logging.warning("⚠️ Aucune donnée valide pour Margin Level")
        return
    
    margin_level[~mask_valid] = float('nan')
    
    
    # Plot sans clip initial (échelle log gère les grandes valeurs)
    ax2.plot(results.index, margin_level, 
             label='Margin Level', color='darkblue', linewidth=2, marker='.', markersize=2)
    
    # ========== ÉCHELLE LOG ==========
    ax2.set_yscale('log')
    ax2.set_ylim(10, cap_value)  # ← cap_value maintenant défini
    # =================================
    
    # Zones critiques
    ax2.axhline(y=100, color='orange', linestyle='--', linewidth=1.5, 
                label='⚠️ Margin Call (100%)', alpha=0.8)
    ax2.axhline(y=50, color='red', linestyle='--', linewidth=1.5, 
                label='🔴 Stop Out (50%)', alpha=0.8)
    ax2.axhline(y=200, color='green', linestyle=':', linewidth=1, 
                label='✅ Zone Saine (200%)', alpha=0.6)
    ax2.axhline(y=1000, color='blue', linestyle=':', linewidth=0.8, 
                label='🟢 Très Safe (1000%)', alpha=0.5)
    
    ax2.set_ylabel('Margin Level (%) - Échelle LOG', fontsize=11)
    ax2.set_xlabel('Time', fontsize=11)
    ax2.set_title("Margin Level - Risk Indicator (Logarithmic Scale)", fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, which='both', alpha=0.3)
    ax2.grid(True, which='minor', alpha=0.1, linestyle=':')
    
    # Annotations
    ax2.text(0.02, 0.95, f'🟢 Min: {valid_ml.min():.0f}%\n🔵 Max: {valid_ml.max():.0f}%\n📊 Mean: {valid_ml.mean():.0f}%', 
             transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    

def plot_price_with_trades(data: pd.DataFrame, trade_events, title="Price + Trades"):
    if data.empty:
        print("Données vides, impossible de tracer.")
        return
    dfp = data.copy()
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(dfp.index, dfp['close'], color='black', linewidth=1, label='Close')

    # FIX: Utiliser 'action' au lieu de 'event'
    opens_long = [(e['time'], e['price']) for e in trade_events if e['action'].upper() == 'OPEN' and e['side'] == 'LONG']
    opens_short = [(e['time'], e['price']) for e in trade_events if e['action'].upper() == 'OPEN' and e['side'] == 'SHORT']
    closes_long = [(e['time'], e['price']) for e in trade_events if e['action'].upper() == 'CLOSE' and e['side'] == 'LONG']
    closes_short = [(e['time'], e['price']) for e in trade_events if e['action'].upper() == 'CLOSE' and e['side'] == 'SHORT']

    def sc(points, marker, color, label):
        if points:
            times, prices = zip(*points)
            ax.scatter(times, prices, marker=marker, color=color, s=100, label=label, zorder=5, edgecolors='black', linewidths=1)

    sc(opens_long, '^', 'green', 'Buy')
    sc(opens_short, 'v', 'red', 'Sell')
    sc(closes_long, 'x', 'darkgreen', 'Buy Close')
    sc(closes_short, 'x', 'darkred', 'Sell Close')

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys())
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

def plot_candles_with_trades(
    data: pd.DataFrame,
    trade_events,
    title: str = "Price + Trades",
    limit: Optional[int] = None,
    mav: Optional[List[int]] = None,
    volume: bool = False,
    align_to_nearest: bool = True,
    robot_coloring: bool = True,
    annotate: bool = False,
    max_annot: int = 200
):
    """
    Combine chandeliers + évènements de trade.
    FIX: utilise addplot (mplfinance) pour aligner correctement timestamps.
    """
    if data is None or data.empty:
        logging.warning("Données vides, impossible de tracer les chandeliers.")
        return
    df = data.copy()
    if limit and limit > 0 and len(df) > limit:
        df = df.iloc[-limit:]

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Préparation OHLC
    ohlc = df[['open','high','low','close']].copy()
    ohlc.columns = ['Open','High','Low','Close']

    # Mapping couleurs robots
    robots = sorted({e.get('robot_id','') for e in trade_events if e.get('robot_id')})
    robot_colors = {}
    if robot_coloring and robots:
        cmap = plt.get_cmap('tab10')
        for i, rid in enumerate(robots):
            robot_colors[rid] = cmap(i % 10)
    
    # FIX: Utiliser 'action' au lieu de 'event'
    opens_buy = []
    opens_sell = []
    closes_buy = []
    closes_sell = []
    
    for e in trade_events:
        t = pd.to_datetime(e['time'])
        if align_to_nearest and t not in ohlc.index:
            idx = ohlc.index.get_indexer([t], method='nearest')[0]
            if idx >= 0:
                t = ohlc.index[idx]
        
        if t not in ohlc.index:
            continue
        
        price = e.get('price', ohlc.loc[t, 'Close'])
        side = e.get('side', '').upper()
        action = e.get('action', '').upper()  # ← FIX: 'action' au lieu de 'event'
        robot_id = e.get('robot_id', '')
        
        color = robot_colors.get(robot_id, 'black') if robot_coloring else 'black'
        
        if action == 'OPEN':
            if side == 'LONG' or side == 'BUY':
                opens_buy.append((t, price, color, robot_id))
            elif side == 'SHORT' or side == 'SELL':
                opens_sell.append((t, price, color, robot_id))
        elif action == 'CLOSE':
            if side == 'LONG' or side == 'BUY':
                closes_buy.append((t, price, color, robot_id))
            elif side == 'SHORT' or side == 'SELL':
                closes_sell.append((t, price, color, robot_id))
    
    # ========== CRÉATION DES SÉRIES POUR MARKERS ==========
    # Créer une série par trade avec NaN partout sauf au timestamp du trade
    all_addplots = []
    
    # Buy Opens (triangles verts ▲)
    for t, price, color, robot_id in opens_buy:
        series = pd.Series([float('nan')] * len(ohlc), index=ohlc.index)
        series.loc[t] = price
        all_addplots.append(mpf.make_addplot(
            series, 
            type='scatter', 
            markersize=120, 
            marker='^', 
            color=color,
            secondary_y=False, 
            panel=0
        ))
    
    # Sell Opens (triangles rouges ▼)
    for t, price, color, robot_id in opens_sell:
        series = pd.Series([float('nan')] * len(ohlc), index=ohlc.index)
        series.loc[t] = price
        all_addplots.append(mpf.make_addplot(
            series, 
            type='scatter', 
            markersize=120, 
            marker='v', 
            color=color,
            secondary_y=False, 
            panel=0
        ))
    
    # Buy Closes (croix vertes ✖)
    for t, price, color, robot_id in closes_buy:
        series = pd.Series([float('nan')] * len(ohlc), index=ohlc.index)
        series.loc[t] = price
        all_addplots.append(mpf.make_addplot(
            series, 
            type='scatter', 
            markersize=100, 
            marker='x', 
            color=color,
            secondary_y=False, 
            panel=0
        ))
    
    # Sell Closes (croix rouges ✖)
    for t, price, color, robot_id in closes_sell:
        series = pd.Series([float('nan')] * len(ohlc), index=ohlc.index)
        series.loc[t] = price
        all_addplots.append(mpf.make_addplot(
            series, 
            type='scatter', 
            markersize=100, 
            marker='x', 
            color=color,
            secondary_y=False, 
            panel=0
        ))
    
    # ========== LOGS DEBUG ==========
    logging.info(f"📍 Markers créés:")
    logging.info(f"   - Buy Opens: {len(opens_buy)}")
    logging.info(f"   - Sell Opens: {len(opens_sell)}")
    logging.info(f"   - Buy Closes: {len(closes_buy)}")
    logging.info(f"   - Sell Closes: {len(closes_sell)}")
    logging.info(f"   - Total addplots: {len(all_addplots)}")
    # ===============================
    
    # Plot
    kwargs = {
        'type': 'candle',
        'style': 'charles',
        'title': title,
        'ylabel': 'Price',
        'volume': volume,
        'addplot': all_addplots if all_addplots else None,
        'warn_too_much_data': 10000
    }
    
    if mav:
        kwargs['mav'] = mav
    
    mpf.plot(ohlc, **kwargs)
