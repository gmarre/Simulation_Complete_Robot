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
        print("Pas de données pour tracer trades.")
        return
    dfp = data.copy()
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(dfp.index, dfp['close'], color='black', linewidth=1, label='Close')

    opens_long = [(e['time'], e['price']) for e in trade_events if e['event']=='open' and e['side']=='LONG']
    opens_short = [(e['time'], e['price']) for e in trade_events if e['event']=='open' and e['side']=='SHORT']
    closes_long = [(e['time'], e['price']) for e in trade_events if e['event']=='close' and e['side']=='LONG']
    closes_short = [(e['time'], e['price']) for e in trade_events if e['event']=='close' and e['side']=='SHORT']

    def sc(points, marker, color, label):
        if points:
            tms, prs = zip(*points)
            ax.scatter(tms, prs, marker=marker, color=color, edgecolor='k', s=55, linewidths=0.4, label=label, zorder=5)

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
        logging.warning("plot_candles_with_trades: DataFrame vide.")
        return
    df = data.copy()
    if limit and limit > 0 and len(df) > limit:
        df = df.tail(limit)

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            logging.error("Index non convertible en datetime.")
            return

    # Préparation OHLC
    ohlc = df[['open','high','low','close']].copy()
    ohlc.columns = ['Open','High','Low','Close']

    # Mapping couleurs robots
    robots = sorted({e.get('robot_id','') for e in trade_events if e.get('robot_id')})
    base_colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
    robot_color_map = {}
    for i, rid in enumerate(robots):
        robot_color_map[rid] = base_colors[i % len(base_colors)]

    # Helper align time
    def nearest_ts(ts):
        if ts in ohlc.index:
            return ts
        pos = ohlc.index.searchsorted(ts)
        if pos <= 0:
            return ohlc.index[0]
        if pos >= len(ohlc.index):
            return ohlc.index[-1]
        before = ohlc.index[pos-1]
        after = ohlc.index[pos]
        return before if (ts - before) <= (after - ts) else after

    # Traitement événements
    processed = []
    for e in trade_events:
        t = e.get('time')
        if t is None:
            continue
        if not isinstance(t, pd.Timestamp):
            try:
                t = pd.Timestamp(t)
            except Exception:
                continue
        if align_to_nearest:
            t_aligned = nearest_ts(t)
        else:
            if t not in ohlc.index:
                continue
            t_aligned = t
        # Filtrer hors fenêtre
        if t_aligned < ohlc.index[0] or t_aligned > ohlc.index[-1]:
            continue
        pe = dict(e)
        pe['_t'] = t_aligned
        processed.append(pe)

    if _HAS_MPF:
        # Créer des séries pour chaque type de marker alignées sur ohlc.index
        # Initialiser NaN partout
        open_long_series = pd.Series(index=ohlc.index, dtype=float)
        open_short_series = pd.Series(index=ohlc.index, dtype=float)
        close_long_series = pd.Series(index=ohlc.index, dtype=float)
        close_short_series = pd.Series(index=ohlc.index, dtype=float)

        for e in processed:
            t_aligned = e['_t']
            price = e.get('price')
            if price is None:
                continue
            if e.get('event') == 'open' and e.get('side') == 'LONG':
                open_long_series.loc[t_aligned] = price
            elif e.get('event') == 'open' and e.get('side') == 'SHORT':
                open_short_series.loc[t_aligned] = price
            elif e.get('event') == 'close' and e.get('side') == 'LONG':
                close_long_series.loc[t_aligned] = price
            elif e.get('event') == 'close' and e.get('side') == 'SHORT':
                close_short_series.loc[t_aligned] = price

        # Construire addplot
        apds = []
        if not open_long_series.isna().all():
            apds.append(mpf.make_addplot(open_long_series, type='scatter', markersize=60, marker='^', color='green', secondary_y=False))
        if not open_short_series.isna().all():
            apds.append(mpf.make_addplot(open_short_series, type='scatter', markersize=60, marker='v', color='red', secondary_y=False))
        if not close_long_series.isna().all():
            apds.append(mpf.make_addplot(close_long_series, type='scatter', markersize=50, marker='x', color='darkgreen', secondary_y=False))
        if not close_short_series.isna().all():
            apds.append(mpf.make_addplot(close_short_series, type='scatter', markersize=50, marker='x', color='darkred', secondary_y=False))

        plot_kwargs = dict(type='candle', style='yahoo', title=title, volume=volume, figsize=(13,6), returnfig=True)
        if mav:
            plot_kwargs['mav'] = mav
        if apds:
            plot_kwargs['addplot'] = apds

        fig, axlist = mpf.plot(ohlc, **plot_kwargs)
        ax_price = axlist[0]

        # Annotations (optionnel)
        if annotate:
            for idx_a, e in enumerate(processed):
                if idx_a >= max_annot:
                    break
                t_aligned = e['_t']
                price = e.get('price')
                txt = e.get('robot_id','')
                pid = e.get('position_id')
                if pid is not None:
                    txt += f"#{pid}"
                color = robot_color_map.get(e.get('robot_id'), 'blue') if robot_coloring else 'blue'
                ax_price.text(t_aligned, price, txt, fontsize=7, ha='center', va='bottom', color=color)

        # Légende manuelle (mplfinance n'ajoute pas de labels aux addplot)
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0],[0], marker='^', color='w', markerfacecolor='green', markersize=8, label='Buy Open'),
            Line2D([0],[0], marker='v', color='w', markerfacecolor='red', markersize=8, label='Sell Open'),
            Line2D([0],[0], marker='x', color='w', markerfacecolor='darkgreen', markersize=8, label='Buy Close'),
            Line2D([0],[0], marker='x', color='w', markerfacecolor='darkred', markersize=8, label='Sell Close')
        ]
        ax_price.legend(handles=legend_elements, loc='upper left')
        ax_price.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()
    else:
        # Fallback manuel (inchangé, déjà OK)
        fig, ax = plt.subplots(figsize=(13,6))
        width = 0.6
        for t in ohlc.index:
            o = ohlc.at[t,'Open']; h = ohlc.at[t,'High']; l = ohlc.at[t,'Low']; c = ohlc.at[t,'Close']
            col = 'green' if c >= o else 'red'
            ax.plot([t,t],[l,h], color=col, linewidth=1)
            top = max(o,c); bottom = min(o,c)
            height = top-bottom if top!=bottom else (abs(c)*1e-7 or 1e-7)
            rect = Rectangle((t, bottom), width, height, facecolor=col, edgecolor='black', linewidth=0.4)
            ax.add_patch(rect)

        for e in processed:
            color = robot_color_map.get(e.get('robot_id'), 'blue') if robot_coloring else (
                'green' if e.get('side')=='LONG' else 'red')
            marker = '^' if (e.get('event')=='open' and e.get('side')=='LONG') else \
                     'v' if (e.get('event')=='open' and e.get('side')=='SHORT') else 'x'
            ax.scatter(e['_t'], e.get('price'), marker=marker, color=color,
                       edgecolor='k', s=60, linewidths=0.4, zorder=5)
            if annotate:
                txt = e.get('robot_id','')
                pid = e.get('position_id')
                if pid is not None:
                    txt += f"#{pid}"
                ax.text(e['_t'], e.get('price'), txt, fontsize=7, ha='center', va='bottom', color=color)

        ax.set_title(title + " (fallback)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()
