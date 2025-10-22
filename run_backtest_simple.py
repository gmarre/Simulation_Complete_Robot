import pandas as pd
import matplotlib.pyplot as plt
from engine.broker import Broker
from engine.simulator import Simulator
from robots.grid_robot import GridRobot
from reporting import visuals
import os
from typing import Optional
import argparse
from robots.CandleSuite_Paul import CandleSuitePaul
import logging
import sys
from robots.SimpleRobotExample import DailyTimeWindowRobot
from reporting.visuals import plot_candles_with_trades

logging.disable(logging.CRITICAL) 

TIMEFRAME_MAP = {
    'm1': '1T',
    'm5': '5T',
    'm15': '15T',
    'm30': '30T',
    'h1': '1H',
    'h4': '4H',
    'd1': '1D'
}

def validate_timeframe(tf: str):
    tf = tf.lower()
    if tf not in TIMEFRAME_MAP:
        raise ValueError(f"Timeframe inconnu: {tf} (attendus: {list(TIMEFRAME_MAP.keys())})")
    return tf

def load_mt5_bars(symbol: str, filename: str, start: Optional[str] = None, end: Optional[str] = None, limit: Optional[int] = None) -> pd.DataFrame:
    """Charge un fichier MT5 exporté avec entête éventuelle:
    Attendu (ordre typique): Date,Timestamp,Open,High,Low,Close,TickCount,Volume,Spread
    - Date: format 'YYYY.MM.DD' ou 'YYYY-MM-DD'
    - Timestamp: 'HH:MM' ou datetime complet; si absent on peut n'utiliser que Date
    Retourne DataFrame indexé en datetime avec colonnes normalisées open/high/low/close.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        os.path.join(base_dir, 'data', filename),
        os.path.join(base_dir, '..', 'data', filename)
    ]
    path_found = None
    for p in candidate_paths:
        if os.path.isfile(p):
            path_found = p
            break
    if path_found is None:
        raise FileNotFoundError("Fichier introuvable pour symbol {}. Testés:\n{}".format(symbol, "\n".join(candidate_paths)))

    # Lecture brute: certains exports peuvent ne pas contenir d'en-tête -> on détecte.
    with open(path_found, 'r', encoding='utf-8', errors='ignore') as f:
        first_line = f.readline().strip()
    has_header = 'open' in first_line.lower() or 'date' in first_line.lower() or 'timestamp' in first_line.lower()

    if has_header:
        df = pd.read_csv(path_found)
    else:
        # Pas d'en-tête: appliquer noms attendus
        cols = ['Date','Timestamp','Open','High','Low','Close','TickCount','Volume','Spread']
        df = pd.read_csv(path_found, names=cols)

    # Normalisation colonnes
    lower_map = {c: c.lower() for c in df.columns}
    df.rename(columns=lower_map, inplace=True)

    # Reconstitution datetime: priorité à combinaison date + timestamp
    if 'date' not in df.columns:
        raise ValueError('Colonne Date manquante dans le fichier {}.'.format(path_found))
    if 'timestamp' in df.columns:
        # Concaténer
        dt_series = df['date'].astype(str) + ' ' + df['timestamp'].astype(str)
    else:
        dt_series = df['date'].astype(str)

    # Essayer différents parse
    dt = pd.to_datetime(dt_series, errors='coerce', infer_datetime_format=True)
    if dt.isna().any():
        # Tentative formats MT5 classiques
        dt = pd.to_datetime(dt_series, format='%Y.%m.%d %H:%M', errors='coerce')
    if dt.isna().any():
        dt = pd.to_datetime(dt_series, format='%Y-%m-%d %H:%M', errors='coerce')
    if dt.isna().any():
        # Dernier recours: date seule
        dt2 = pd.to_datetime(df['date'], errors='coerce')
        if dt2.isna().all():
            raise ValueError('Impossible de parser les dates pour {}.'.format(path_found))
        dt = dt2

    df.index = dt
    df = df.sort_index()
    # Conserver OHLC
    needed_cols = ['open','high','low','close']
    for c in needed_cols:
        if c not in df.columns:
            raise ValueError(f'Colonne {c} absente dans {path_found}. Colonnes: {list(df.columns)}')
    ohlc = df[needed_cols].astype(float)

    # Filtrage
    if start:
        ohlc = ohlc.loc[start:]
    if end:
        ohlc = ohlc.loc[:end]
    if limit:
        ohlc = ohlc.tail(limit)

    print(f"Chargé {len(ohlc)} barres pour {symbol} depuis {path_found}")
    return ohlc

def resample_ohlc(df: pd.DataFrame, tf_key: str) -> pd.DataFrame:
    tf_key = tf_key.lower()
    if tf_key not in TIMEFRAME_MAP or tf_key == 'm1':
        return df
    rule = TIMEFRAME_MAP[tf_key]
    r = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low':  'min',
        'close':'last'
    }).dropna()
    return r

def get_plot_df(raw_df: pd.DataFrame, tf_code: str) -> pd.DataFrame:
    """Renvoie les données resamplées pour l'affichage."""
    tf_code = tf_code.lower()
    return resample_ohlc(raw_df, tf_code)

def parse_args():
    parser = argparse.ArgumentParser(description='Backtest CandleSuitePaul / TimeWindow')
    parser.add_argument('--symbol', default='EURGBP')
    parser.add_argument('--file', default='EURGBP_mt5_bars.csv')
    parser.add_argument('--start', default='2023-09-01')
    parser.add_argument('--end', default='2024-03-01')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--plots', default='candles_trades,equity,lots,margin', help='Liste des graphiques à afficher (candles_trades, equity, lots)')
    parser.add_argument('--suite', type=int, default=5)
    parser.add_argument('--xtrem', type=int, default=200)
    parser.add_argument('--atr-period', type=int, default=200)
    parser.add_argument('--tp', type=float, default=5)
    parser.add_argument('--inp_lot_for_10k', type=float, default=0.1)
    parser.add_argument('--dist', type=float, default=1.0)
    parser.add_argument('--factor', type=float, default=2.0)
    parser.add_argument('--max-grid-levels', type=int, default=10)
    g = parser.add_mutually_exclusive_group()
    g.add_argument('--common-tp', dest='common_tp', action='store_true', default=True)
    g.add_argument('--no-common-tp', dest='common_tp', action='store_false')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--log-level', default='INFO', help='DEBUG, INFO, WARNING...')
    parser.add_argument('--log-file', default='', help='Fichier log (sinon stdout)')
    parser.add_argument('--progress-interval', type=int, default=2000, help='Barres entre logs de progression')
    parser.add_argument('--timeframe', default='m15',help='Timeframe principal (m1,m5,m15,m30,h1,h4,d1) utilisé pour agrégation simple si besoin')
    parser.add_argument('--plot-candles-tf', default='m15',help='Timeframe spécifique pour plot_candles (sinon = --timeframe)')
    parser.add_argument('--plot-trades-tf', default='m15',help='Timeframe spécifique pour plot_price_with_trades (sinon = --plot-candles-tf / --timeframe)')
    return parser.parse_args()

def configure_logging(level: str, log_file: str):
    lvl = getattr(logging, level.upper(), logging.INFO)
    handlers = []
    
    # Handler pour fichier spécifique si demandé
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(lvl)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
        handlers.append(file_handler)
    
    # Handler pour console (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(lvl)
    console_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
    handlers.append(console_handler)
    
    # Handler pour fichier backtest_logs.txt (tous les logs INFO+)
    default_log_handler = logging.FileHandler('backtest_logs.txt', mode='w', encoding='utf-8')
    default_log_handler.setLevel(logging.INFO)  # Capture INFO et supérieur
    default_log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
    handlers.append(default_log_handler)
    
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=handlers,
        force=True  # Force la reconfiguration si déjà initialisé
    )
    
    logging.info("📝 Logs sauvegardés dans: backtest_logs.txt")

def calculate_warmup_timedelta(robot, tf: str) -> pd.Timedelta:
    """
    Calcule le Timedelta pour le warm-up basé sur le timeframe.
    AMÉLIORATION: Calcul plus réaliste avec marge adaptative.
    """
    if hasattr(robot, 'get_warmup_periods'):
        warmup_bars = robot.get_warmup_periods()
    else:
        return pd.Timedelta(0)
    
    if warmup_bars == 0:
        return pd.Timedelta(0)
    
    tf_minutes = {
        'm1': 1, 'm5': 5, 'm15': 15, 'm30': 30,
        'h1': 60, 'h4': 240, 'd1': 1440
    }
    
    minutes_per_bar = tf_minutes.get(tf.lower(), 1)
    total_minutes = warmup_bars * minutes_per_bar
    
    # ========== MARGE ADAPTATIVE FOREX ==========
    # Forex fermé ~48h/168h = 28.6% du temps
    # On ajoute 35% de marge pour être safe (weekends + gaps)
    total_minutes = int(total_minutes * 1.35)
    # ============================================
    
    return pd.Timedelta(minutes=total_minutes)

def main():
    args = parse_args()
    configure_logging(args.log_level, args.log_file)
    logging.info("=== DÉMARRAGE BACKTEST DailyTimeWindowRobot ===")

    # ========== CRÉER LES ROBOTS D'ABORD ==========
    robotPaul1 = CandleSuitePaul(
        robot_id='CS1_m15',
        symbol=args.symbol,
        timeframe='m15',
        inp_suite=args.suite,
        inp_xtrem_research=args.xtrem,
        atr_period=args.atr_period,
        inp_tp=args.tp,
        inp_lot_for_10k=args.inp_lot_for_10k,
        inp_distance_between_orders=args.dist,
        inp_grid_recov_factor=args.factor,
        close_on_common_tp=args.common_tp,
        max_grid_levels=args.max_grid_levels,
        debug=False
    )
    robotPaul2 = CandleSuitePaul(
        robot_id='CS2_H1',
        symbol=args.symbol,
        timeframe='H1',
        inp_suite=3,
        inp_xtrem_research=50,
        atr_period=args.atr_period,
        inp_tp=2.5,
        inp_lot_for_10k=args.inp_lot_for_10k,
        inp_distance_between_orders=3,
        inp_grid_recov_factor=1.5,
        close_on_common_tp=args.common_tp,
        max_grid_levels=args.max_grid_levels,
        debug=False
    )
    robotPaul3 = CandleSuitePaul(
        robot_id='CS3_m30',
        symbol=args.symbol,
        timeframe='m30',
        inp_suite=2,
        inp_xtrem_research=300,
        atr_period=args.atr_period,
        inp_tp=1.5,
        inp_lot_for_10k=args.inp_lot_for_10k,
        inp_distance_between_orders=1,
        inp_grid_recov_factor=2,
        close_on_common_tp=args.common_tp,
        max_grid_levels=args.max_grid_levels,
        debug=False
    )
    robotPaul4 = CandleSuitePaul(
        robot_id='CS4_m5',
        symbol=args.symbol,
        timeframe='m5',
        inp_suite=8,
        inp_xtrem_research=50,
        atr_period=args.atr_period,
        inp_tp=5,
        inp_lot_for_10k=0.4,
        inp_distance_between_orders=1,
        inp_grid_recov_factor=1.6,
        close_on_common_tp=args.common_tp,
        max_grid_levels=args.max_grid_levels,
        debug=False
    )
    robotPaul5 = CandleSuitePaul(
        robot_id='CS5_m15',
        symbol=args.symbol,
        timeframe='m15',
        inp_suite=8,
        inp_xtrem_research=100,
        atr_period=args.atr_period,
        inp_tp=4,
        inp_lot_for_10k=args.inp_lot_for_10k,
        inp_distance_between_orders=2,
        inp_grid_recov_factor=1.6,
        close_on_common_tp=args.common_tp,
        max_grid_levels=args.max_grid_levels,
        debug=False
    )
    
    robot1 = DailyTimeWindowRobot(
        robot_id="TIME10_12_H1",
        symbol=args.symbol,
        timeframe='h1',
        side="BUY",
        lots=0.1,
        open_hour=10,
        open_minute=0,
        close_hour=12,
        close_minute=0,
        debug=False
    )

    robots = [robotPaul1, robotPaul2, robotPaul3, robotPaul4, robotPaul5]  # On n'utilise que CandleSuite_Paul

    # ========== CALCUL WARM-UP AUTOMATIQUE ==========
    max_warmup = pd.Timedelta(0)
    warmup_details = []
    
    for r in robots:
        warmup = calculate_warmup_timedelta(r, r.timeframe)
        warmup_details.append(f"{r.robot_id} ({r.timeframe.upper()}): {warmup}")
        if warmup > max_warmup:
            max_warmup = warmup
    
    logging.info(f"🔍 Warm-up requis par robot: {', '.join(warmup_details)}")
    
    # ========== CHARGEMENT DONNÉES AVEC WARM-UP ==========
    if args.start and max_warmup > pd.Timedelta(0):
        start_date = pd.to_datetime(args.start)
        warmup_start = start_date - max_warmup
        
        logging.info(f"🔄 Warm-up automatique activé:")
        logging.info(f"   - Simulation demandée: {args.start} → {args.end}")
        logging.info(f"   - Chargement étendu: {warmup_start.strftime('%Y-%m-%d')} → {args.end}")
        logging.info(f"   - Période warm-up: {max_warmup} ({max_warmup.days} jours + {max_warmup.seconds//3600}h)")
        
        raw = load_mt5_bars(
            args.symbol, 
            args.file, 
            start=warmup_start.strftime('%Y-%m-%d'),  # ← DATE AJUSTÉE
            end=args.end
        )
        
        # Vérification données disponibles
        if not raw.empty:
            actual_start = raw.index[0]
            logging.info(f"✅ Données chargées depuis {actual_start} (première barre disponible)")
            
            # ========== VALIDATION WARM-UP POUR CHAQUE ROBOT ==========
            start_date_idx = raw.index.get_indexer([start_date], method='nearest')[0]
            bars_before_start = start_date_idx  # Nombre de barres M1 avant start
            
            tf_mult = {'m1':1, 'm5':5, 'm15':15, 'm30':30, 'h1':60, 'h4':240, 'd1':1440}
            
            all_warmup_ok = True
            for r in robots:
                if not hasattr(r, 'get_warmup_periods'):
                    continue
                
                # Warm-up requis pour CE robot (en barres de SON timeframe)
                warmup_bars_tf = r.get_warmup_periods()
                # Conversion en barres M1 (approximation)
                warmup_bars_m1 = warmup_bars_tf * tf_mult.get(r.timeframe.lower(), 1)
                
                if bars_before_start >= warmup_bars_m1:
                    logging.info(f"✅ {r.robot_id} warm-up OK: {bars_before_start} barres M1 (requis: {warmup_bars_m1})")
                else:
                    logging.warning(f"⚠️ {r.robot_id} warm-up PARTIEL: {bars_before_start}/{warmup_bars_m1} barres M1")
                    all_warmup_ok = False
            
            if all_warmup_ok:
                logging.info("✅ Tous les robots ont un warm-up complet")
            else:
                logging.warning("⚠️ Certains robots ont un warm-up partiel, résultats peuvent être biaisés")
            
            # Calcul barres disponibles avant start demandé
            bars_before_start = len(raw.loc[:start_date])
            tf_mult = {'m1':1, 'm5':5, 'm15':15, 'm30':30, 'h1':60, 'h4':240, 'd1':1440}
            expected_m15_warmup = robots[0].get_warmup_periods() if hasattr(robots[0], 'get_warmup_periods') else 0  # ← CORRECTION ICI
            expected_m1_warmup = expected_m15_warmup * tf_mult.get(robots[0].timeframe.lower(), 1)  # ← ET ICI
            
            if bars_before_start >= expected_m1_warmup:
                logging.info(f"✅ Warm-up complet: {bars_before_start} barres M1 disponibles (requis: {expected_m1_warmup})")
            else:
                logging.warning(f"⚠️ Warm-up partiel: {bars_before_start}/{expected_m1_warmup} barres M1 disponibles")
    else:
        if not args.start:
            logging.info(f"📊 Pas de --start spécifié, chargement normal sans warm-up")
        else:
            logging.info(f"📊 Aucun warm-up requis (robots sans indicateurs)")
        
        raw = load_mt5_bars(args.symbol, args.file, start=args.start, end=args.end, limit=args.limit)
    
    # (Optionnel) agrégation simple
    try:
        simulate_tf = validate_timeframe(args.timeframe)
    except ValueError as e:
        logging.error(e)
        return

    data = resample_ohlc(raw, simulate_tf)
    logging.info(f"Timeframe choisi: {simulate_tf}  Barres après agrégation: {len(data)} (source M1: {len(raw)})")
    
    # Afficher range effectif de simulation (après warm-up)
    if args.start:
        sim_start_idx = data.index.get_indexer([pd.to_datetime(args.start)], method='nearest')[0]
        actual_sim_data = data.iloc[sim_start_idx:]
        logging.info(f"📊 Période simulation effective: {actual_sim_data.index[0]} → {actual_sim_data.index[-1]} ({len(actual_sim_data)} barres)")
    else:
        logging.info(f"📊 Période simulation: {data.index[0]} → {data.index[-1]} ({len(data)} barres)")

    broker = Broker(starting_balance=1500.0, leverage=500)
    sim = Simulator({args.symbol: raw}, robots, broker)
    results = sim.run()
    results.set_index('time', inplace=True)

    logging.info("=== FIN BACKTEST ===")

    # ========== SAUVEGARDE RÉSULTATS ==========
    logging.info("💾 Sauvegarde des résultats...")
    
    # Métriques
    results.to_csv('results_metrics.csv')
    logging.info("✅ Métriques sauvegardées: results_metrics.csv")
    
    # Trades fermés
    if broker.closed_trades:
        pd.DataFrame(broker.closed_trades).to_csv('closed_trades.csv', index=False)
        logging.info(f"✅ Trades fermés sauvegardés: closed_trades.csv ({len(broker.closed_trades)} trades)")
    else:
        logging.info("⚠️ Aucun trade fermé à sauvegarder")
    
    # Logs déjà sauvegardés automatiquement dans backtest_logs.txt
    logging.info("✅ Logs complets disponibles: backtest_logs.txt")

    if not args.no_plots:
        selected = [p.strip().lower() for p in args.plots.split(',') if p.strip()]
        robots_ids = [r.robot_id for r in robots]
        main_symbol = args.symbol

        # ========== AUTO-DÉTECTION TF LE PLUS PETIT ==========
        tf_order = ['m1', 'm5', 'm15', 'm30', 'h1', 'h4', 'd1']
        robot_timeframes = [r.timeframe.lower() for r in robots]
        
        # Trouver le TF le plus petit utilisé
        smallest_tf = None
        for tf in tf_order:
            if tf in robot_timeframes:
                smallest_tf = tf
                break
        
        if smallest_tf is None:
            smallest_tf = 'm15'  # Fallback
        
        logging.info(f"🎨 TF plots automatique: {smallest_tf.upper()} (le plus petit des robots: {', '.join([r.upper() for r in set(robot_timeframes)])})")
        
        # Override si explicitement demandé
        candles_tf = args.plot_candles_tf.lower() if args.plot_candles_tf else smallest_tf
        trades_tf = args.plot_trades_tf.lower() if args.plot_trades_tf else smallest_tf
        # =======================================================

        try:
            candles_tf = validate_timeframe(candles_tf)
            trades_tf = validate_timeframe(trades_tf)
        except ValueError as e:
            logging.error(f"Plot timeframe invalide: {e}")
            return

        candles_df = get_plot_df(raw, candles_tf)
        trades_df = get_plot_df(raw, trades_tf)

        if 'candles' in selected:
            visuals.plot_candles(candles_df, title=f'{main_symbol} ({candles_tf.upper()}) Price', limit=None)
        if 'equity' in selected:
            visuals.plot_equity(results, robots=robots_ids, title='Equity / Balance')
        if 'lots' in selected:
            visuals.plot_lots(results, robots=robots_ids)
        if 'unrealized' in selected:
            visuals.plot_unrealized(results, robots=robots_ids)
        if 'margin' in selected:
            visuals.plot_margin(results)
        if 'trades' in selected:
            visuals.plot_price_with_trades(
                trades_df,
                broker.trade_events,
                title=f'{main_symbol} ({trades_tf.upper()}) Price + Trades',
                limit=None,
                align_to_nearest=True,
                show_gaps=True
            )
        if 'candles_trades' in selected or 'candle_position' in selected:
            combined_tf = trades_tf
            combined_df = trades_df
            plot_candles_with_trades(
                combined_df,
                broker.trade_events,
                title=f'{main_symbol} ({combined_tf.upper()}) Candles + Trades',
                mav=None,
                volume=False,
                limit=None,
                align_to_nearest=True,
                robot_coloring=True,
                annotate=False
            )

if __name__ == '__main__':
    main()
