"""Configuration centrale des chemins"""
import os
from pathlib import Path

# Racine du projet
PROJECT_ROOT = Path(__file__).parent

# Dossier data (en dehors du projet)
DATA_DIR = PROJECT_ROOT.parent / "data"

# Vérifier que le dossier existe
if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Le dossier data n'existe pas: {DATA_DIR}\n"
        f"Créez-le ou ajustez DATA_DIR dans config.py"
    )

# Chemins des fichiers de données
DATA_FILES = {
    'EURGBP': DATA_DIR / 'EURGBP_mt5_bars.csv',
    'AUDCAD': DATA_DIR / 'AUDCAD_mt5_bars.csv',
    'USDCAD': DATA_DIR / 'USDCAD_mt5_bars.csv',
    'EURAUD': DATA_DIR / 'EURAUD_mt5_bars.csv',
    'GBPNZD': DATA_DIR / 'GBPNZD_mt5_bars.csv',
    'NZDCAD': DATA_DIR / 'NZDCAD_mt5_bars.csv',
}

# Configuration broker
BROKER_CONFIG = {
    'starting_balance': 1500.0,
    'leverage': 500,
    'spread': 0.0001,
}

# Configuration backtest
BACKTEST_CONFIG = {
    'start_date': '2024-01-01',
    'end_date': '2024-02-01',
    'symbol': 'EURGBP',
}

print(f"✅ Configuration chargée: DATA_DIR = {DATA_DIR}")