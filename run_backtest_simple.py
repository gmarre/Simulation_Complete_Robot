import pandas as pd
import matplotlib.pyplot as plt
from engine.broker import Broker
from engine.simulator import Simulator
from reporting import visuals
from reporting.visuals import plot_candles_with_trades
import os
import argparse
import logging
import sys
from typing import List, Dict, Optional
from robots.CandleSuite_Paul import CandleSuitePaul
from robots.DailyTimeWindowRobot import DailyTimeWindowRobot

logging.disable(logging.CRITICAL)

# ========== CONFIGURATION ==========
TIMEFRAME_MAP = {
    'm1': '1min', 'm5': '5min', 'm15': '15min', 'm30': '30min',
    'h1': '1H', 'h4': '4H', 'd1': '1D'
}

PLOT_OPTIONS = ['candles_trades', 'equity', 'lots', 'margin']
# ===================================


def validate_timeframe(tf: str) -> str:
    """Valide et normalise un timeframe"""
    tf = tf.lower()
    if tf not in TIMEFRAME_MAP:
        raise ValueError(f"Timeframe inconnu: {tf}. Attendus: {list(TIMEFRAME_MAP.keys())}")
    return tf


def load_mt5_csv(symbol: str, filename: str, start: Optional[str] = None, 
                 end: Optional[str] = None) -> pd.DataFrame:
    """
    Charge un fichier CSV MT5 avec gestion robuste des formats.
    Format attendu: Date, Timestamp, Open, High, Low, Close, TickCount, Volume, Spread
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    path = os.path.join(base_dir, 'data', filename)

    
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    
    # Détection header
    with open(path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip().lower()
    
    has_header = any(col in first_line for col in ['open', 'date', 'timestamp'])
    
    if has_header:
        df = pd.read_csv(path)
    else:
        cols = ['Date', 'Timestamp', 'Open', 'High', 'Low', 'Close', 
                'TickCount', 'Volume', 'Spread']
        df = pd.read_csv(path, names=cols)
    
    # Normalisation colonnes
    df.columns = [c.lower() for c in df.columns]
    
    # Parse datetime
    if 'date' not in df.columns:
        raise ValueError(f"Colonne 'Date' manquante dans {filename}")
    
    if 'timestamp' in df.columns:
        dt_str = df['date'].astype(str) + ' ' + df['timestamp'].astype(str)
    else:
        dt_str = df['date'].astype(str)
    
    df.index = pd.to_datetime(dt_str, errors='coerce', infer_datetime_format=True)
    df = df.sort_index().dropna()
    
    # Extraire OHLC
    ohlc_cols = ['open', 'high', 'low', 'close']
    missing = [c for c in ohlc_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans {filename}: {missing}")
    
    ohlc = df[ohlc_cols].astype(float)
    
    # Filtrage
    if start:
        ohlc = ohlc.loc[start:]
    if end:
        ohlc = ohlc.loc[:end]
    
    logging.info(f"📊 {symbol}: {len(ohlc)} barres chargées ({ohlc.index[0]} → {ohlc.index[-1]})")
    return ohlc


def calculate_warmup_period(robots: List, tf_minutes: Dict[str, int]) -> pd.Timedelta:
    """
    Calcule la période de warm-up nécessaire pour tous les robots.
    Ajoute 35% de marge pour les weekends Forex.
    """
    max_warmup = pd.Timedelta(0)
    
    for robot in robots:
        if not hasattr(robot, 'get_warmup_periods'):
            continue
        
        bars_needed = robot.get_warmup_periods()
        if bars_needed == 0:
            continue
        
        tf = robot.timeframe.lower()
        minutes = bars_needed * tf_minutes.get(tf, 1)
        minutes = int(minutes * 1.35)  # Marge Forex 35%
        
        warmup = pd.Timedelta(minutes=minutes)
        if warmup > max_warmup:
            max_warmup = warmup
        
        logging.debug(f"  - {robot.robot_id} ({tf.upper()}): {bars_needed} barres → {warmup}")
    
    return max_warmup


def load_data_with_warmup(symbol: str, filename: str, start: str, end: str, 
                          warmup: pd.Timedelta) -> pd.DataFrame:
    """Charge les données avec extension pour warm-up"""
    if warmup > pd.Timedelta(0):
        warmup_start = (pd.to_datetime(start) - warmup).strftime('%Y-%m-%d')
        logging.info(f"🔄 Warm-up: chargement depuis {warmup_start} (au lieu de {start})")
    else:
        warmup_start = start
        logging.info(f"📊 Pas de warm-up requis")
    
    return load_mt5_csv(symbol, filename, start=warmup_start, end=end)


def create_robots(args) -> List:
    """
    Factory: Créer tous les robots de la simulation.
    Centralisé pour faciliter la gestion de 50+ robots.
    """
    robots = []
    
    # ========== CANDLESUITE ROBOTS ==========
    candlesuite_configs = [
        {
            'id': 'CS1_m30', 'tf': 'm30', 'suite': 8, 'xtrem': 200, 
            'atr': 200, 'tp': 2.5, 'lot': 0.1, 'dist': 3, 'factor': 2, 
            'inversion': False
        },
        {
            'id': 'CS1_m15', 'tf': 'm15', 'suite': 6, 'xtrem': 100, 
            'atr': 200, 'tp': 3, 'lot': 0.1, 'dist': 3, 'factor': 2, 
            'inversion': False
        },
        {
            'id': 'CS1_H1', 'tf': 'h1', 'suite': 3, 'xtrem': 200, 
            'atr': 200, 'tp': 1.5, 'lot': 0.1, 'dist': 5, 'factor': 2.5, 
            'inversion': False
        },
        {
            'id': 'CS1_m30', 'tf': 'm30', 'suite': 4, 'xtrem': 150, 
            'atr': 200, 'tp': 2.5, 'lot': 0.1, 'dist': 3, 'factor': 2.5, 
            'inversion': False
        },
        {
            'id': 'CS1_m15', 'tf': 'm15', 'suite': 6, 'xtrem': 200, 
            'atr': 200, 'tp': 1.5, 'lot': 0.1, 'dist': 5, 'factor': 1.5, 
            'inversion': True
        },
        # Ajoutez ici les 50 autres robots...
        # {
        #     'id': 'CS2_H1', 'tf': 'h1', 'suite': 3, 'xtrem': 50,
        #     'atr': 200, 'tp': 2.5, 'lot': 0.1, 'dist': 3, 'factor': 1.5,
        #     'inversion': True
        # },
    ]
    
    for cfg in candlesuite_configs:
        robot = CandleSuitePaul(
            robot_id=cfg['id'],
            symbol=args.symbol,
            timeframe=cfg['tf'],
            inp_suite=cfg['suite'],
            inp_xtrem_research=cfg['xtrem'],
            atr_period=cfg['atr'],
            inp_tp=cfg['tp'],
            inp_lot_for_10k=cfg['lot'],
            inp_distance_between_orders=cfg['dist'],
            inp_grid_recov_factor=cfg['factor'],
            inversion=cfg['inversion'],
            close_on_common_tp=args.common_tp,
            max_grid_levels=args.max_grid_levels,
            debug=False
        )
        robots.append(robot)
    
    # ========== TIME WINDOW ROBOTS ==========
    # Décommenter si nécessaire
    # time_window_configs = [
    #     {'id': 'TW1', 'tf': 'h1', 'side': 'BUY', 'open_h': 10, 'close_h': 12},
    # ]
    # 
    # for cfg in time_window_configs:
    #     robot = DailyTimeWindowRobot(
    #         robot_id=cfg['id'],
    #         symbol=args.symbol,
    #         timeframe=cfg['tf'],
    #         side=cfg['side'],
    #         lots=0.1,
    #         open_hour=cfg['open_h'],
    #         close_hour=cfg['close_h'],
    #         debug=False
    #     )
    #     robots.append(robot)
    
    logging.info(f"🤖 {len(robots)} robots créés: {', '.join([r.robot_id for r in robots])}")
    return robots


def run_backtest(args):
    """Exécute le backtest complet"""
    
    # ========== CRÉATION ROBOTS ==========
    robots = create_robots(args)
    
    # ========== CALCUL WARM-UP ==========
    tf_minutes = {'m1': 1, 'm5': 5, 'm15': 15, 'm30': 30, 'h1': 60, 'h4': 240, 'd1': 1440}
    warmup = calculate_warmup_period(robots, tf_minutes)
    
    if warmup > pd.Timedelta(0):
        logging.info(f"📏 Warm-up total: {warmup} ({warmup.days}j {warmup.seconds//3600}h)")
    
    # ========== CHARGEMENT DONNÉES ==========
    raw_data = load_data_with_warmup(
        args.symbol, 
        args.file, 
        args.start, 
        args.end, 
        warmup
    )
    
    # ========== CRÉATION BROKER + SIMULATOR ==========
    broker = Broker(
        starting_balance=args.balance,
        leverage=args.leverage,
        account_currency='EUR'
    )
    
    # Multi-symboles: Ajouter ici d'autres paires si nécessaire
    data_feeds = {args.symbol: raw_data}
    
    simulator = Simulator(data_feeds, robots, broker)
    
    # ========== EXÉCUTION ==========
    logging.info("🚀 Démarrage simulation...")
    results = simulator.run()
    results.set_index('time', inplace=True)
    
    # ========== SAUVEGARDE ==========
    save_results(results, broker)
    
    # ========== VISUALISATION ==========
    if not args.no_plots:
        plot_results(args, results, broker, raw_data, robots)
    
    logging.info("✅ Backtest terminé avec succès")


def save_results(results: pd.DataFrame, broker: Broker):
    """Sauvegarde tous les résultats"""
    results.to_csv('results_metrics.csv')
    logging.info(f"💾 Métriques: results_metrics.csv ({len(results)} lignes)")
    
    if broker.closed_trades:
        trades_df = pd.DataFrame(broker.closed_trades)
        trades_df.to_csv('closed_trades.csv', index=False)
        logging.info(f"💾 Trades: closed_trades.csv ({len(broker.closed_trades)} trades)")
    else:
        logging.warning("⚠️ Aucun trade fermé")
    
    # ========== DEBUG TRADE EVENTS ==========
    logging.info(f"📊 Nombre d'événements trade_events: {len(broker.trade_events)}")
    if broker.trade_events:
        logging.info(f"   Premier événement: {broker.trade_events[0]}")
        logging.info(f"   Dernier événement: {broker.trade_events[-1]}")
        
        # Sauvegarde pour inspection
        events_df = pd.DataFrame(broker.trade_events)
        events_df.to_csv('trade_events_debug.csv', index=False)
        logging.info(f"💾 Debug: trade_events_debug.csv ({len(broker.trade_events)} événements)")
    else:
        logging.error("❌ PROBLÈME: broker.trade_events est VIDE !")
        logging.error("   Les graphiques ne pourront pas afficher les trades")
    # ========================================


def plot_results(args, results: pd.DataFrame, broker: Broker, 
                 raw_data: pd.DataFrame, robots: List):
    """Génère les graphiques demandés"""
    requested = [p.strip().lower() for p in args.plots.split(',') if p.strip()]
    logging.info(f"📊 Génération graphiques: {', '.join(requested)}")
    
    # ========== VÉRIFICATION TRADE_EVENTS ==========
    if not broker.trade_events:
        logging.error("❌ ERREUR: broker.trade_events est vide!")
        logging.error("   Impossible d'afficher les trades sur le graphique")
        logging.info("   Vérifiez que le Broker enregistre correctement les événements")
    else:
        logging.info(f"✅ {len(broker.trade_events)} événements à afficher")
    # ===============================================
    
    # Utiliser le timeframe spécifié pour les plots (au lieu du plus petit)
    plot_tf = args.plot_timeframe.lower()
    logging.info(f"🎨 Timeframe plots: {plot_tf.upper()}")
    
    # Resample data pour plots selon le timeframe demandé
    if plot_tf == 'm1':
        plot_data = raw_data
    else:
        pandas_tf = TIMEFRAME_MAP[plot_tf]
        logging.info(f"📊 Resampling M1 → {plot_tf.upper()} ({pandas_tf})")
        plot_data = raw_data.resample(pandas_tf).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()
        logging.info(f"📊 {len(plot_data)} barres {plot_tf.upper()} générées")
    
    robot_ids = [r.robot_id for r in robots]
    
    # ========== GÉNÉRATION PLOTS ==========
    if 'candles_trades' in requested:
        plot_candles_with_trades(
            plot_data,
            broker.trade_events,
            title=f'{args.symbol} ({plot_tf.upper()}) - Candles + Trades',
            mav=None,
            volume=False,
            limit=None,
            align_to_nearest=True,
            robot_coloring=True,
            annotate=False
        )
    
    if 'equity' in requested:
        visuals.plot_equity(
            results, 
            robots=robot_ids, 
            title='Equity / Balance'
        )
    
    if 'lots' in requested:
        visuals.plot_lots(results, robots=robot_ids)
    
    if 'margin' in requested:
        visuals.plot_margin(results)
    
    plt.show()


def parse_args():
    """Parse arguments ligne de commande"""
    parser = argparse.ArgumentParser(
        description='Backtest Multi-Robots Multi-Symboles',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Données
    parser.add_argument('--symbol', default='EURGBP', help='Symbole principal')
    parser.add_argument('--file', default='EURGBP_mt5_bars.csv', help='Fichier CSV')
    parser.add_argument('--start', default='2024-06-03', help='Date début (YYYY-MM-DD)')
    parser.add_argument('--end', default='2024-09-01', help='Date fin (YYYY-MM-DD)')
    
    # Broker
    parser.add_argument('--balance', type=float, default=1500.0, help='Capital initial (EUR)')
    parser.add_argument('--leverage', type=int, default=500, help='Levier')
    
    # Robots (paramètres communs)
    parser.add_argument('--max-grid-levels', type=int, default=100, help='Niveaux grid max')
    parser.add_argument('--common-tp', action='store_true', default=True, help='TP commun')
    parser.add_argument('--no-common-tp', dest='common_tp', action='store_false')
    
    # Logs
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Niveau de log')
    
    # Visualisation
    parser.add_argument('--no-plots', action='store_true', help='Désactiver graphiques')
    parser.add_argument('--plots', default='candles_trades,equity,lots,margin',
                       help=f'Graphiques à afficher: {", ".join(PLOT_OPTIONS)}')
    parser.add_argument('--plot-timeframe', default='m30',
                       choices=['m1', 'm5', 'm15', 'm30', 'h1', 'h4', 'd1'],
                       help='Timeframe pour affichage graphique candles')
    
    return parser.parse_args()


def configure_logging(level: str):
    """Configure le système de logs"""
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('backtest_logs.txt', mode='w', encoding='utf-8')
    ]
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=handlers,
        force=True
    )
    
    logging.info("=" * 70)
    logging.info("🚀 DÉMARRAGE BACKTEST MULTI-ROBOTS")
    logging.info("=" * 70)


def main():
    args = parse_args()
    configure_logging(args.log_level)
    
    try:
        run_backtest(args)
    except Exception as e:
        logging.error(f"❌ ERREUR CRITIQUE: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
