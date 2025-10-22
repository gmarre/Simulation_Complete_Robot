# Backtest Multi-Robots (Architecture de Base)

## Structure
```
Xtrem/
  engine/
    broker.py         # Gestion des ordres, positions, PnL
    datafeed.py       # Itération synchronisée des données
    simulator.py      # Boucle principale de simulation
    strategy_base.py  # Interface de base des stratégies
  robots/
    grid_robot.py     # Implémentation d'un robot grille simple
  data/               # (Tes fichiers CSV historiques)
  run_backtest.py     # Script exemple de lancement de backtest
  requirements.txt    # Dépendances Python
  README.md
```

## Installation
```
pip install -r requirements.txt
```

## Format des données
Chaque CSV devrait avoir un index datetime (`parse_dates=["time"], index_col="time"`) et des colonnes `open, high, low, close`.

## Ajouter un robot
Créer un fichier dans `robots/` dérivant de `StrategyBase` et retournant une liste d'`Order` dans `on_bar`.

## Lancer une simulation (arguments)

Script principal: `run_backtest.py`

Arguments principaux:
```
--symbol EURGBP              # Symbole logique
--file EURGBP_mt5_bars.csv   # Nom du fichier CSV
--start 2024-01-01           # Début (inclus)
--end 2024-02-01             # Fin (incluse)
--limit 10000                # Garde seulement les N dernières barres après filtrage
--plots equity,lots,candles,unrealized,margin  # Sélection graphiques
--no-plots                   # Désactiver l'affichage
```
Exemple:
```
python run_backtest.py --symbol EURGBP --file EURGBP_mt5_bars.csv --start 2024-01-01 --end 2024-03-31 --plots equity,lots,candles
```

## Visualisation

Module `reporting/visuals.py` fournit:
1. Candles (`plot_candles`) – utilise `mplfinance` si installé (déjà listé), sinon fallback ligne du close.
2. Equity / Balance (`plot_equity`) – equity globale + equity par robot (balance + unrealized robot si dispo).
3. Lots (`plot_lots`) – total + par robot.
4. PnL latent (`plot_unrealized`).
5. Marge utilisée (`plot_margin`).

Sélection via `--plots` (liste séparée par virgules).

## Margin
Calcul simplifié: `margin = (lots * contract_size * price) / leverage` agrégé sur positions.
Leverage configurable dans `Broker(leverage=30)`.

## Extensions possibles
- Stop loss / Take profit automatiques
- Slippage & spread
- Limites de risque globales
- Export CSV des metrics & trades
- Module de métriques (Sharpe, Max Drawdown, Profit Factor)
- Auto-chargement multi-symboles + robots dynamiques

## Avertissement
Code de démonstration simplifié. Pour un usage réel, ajouter gestion d'erreurs, validation des entrées, et tests unitaires.
