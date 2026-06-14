from neuralforecast import NeuralForecast
import yfinance as yf
import logging
import importlib
from scipy.ndimage import gaussian_filter1d

import polymarket as pm

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

assets = ["GOLD", "SILVER", "BTC", "ETH", "BRENT"]
# assets = ["GOLD"]
horizon = 30
start_date = "2018-01-01"

all_predictions = {}

for asset in assets:
    print(f"Processing {asset}...")
    
    module = importlib.import_module(f"assets.{asset}")
    symbol = module.get_symbol()
    models = module.get_models(horizon)
    
    df = yf.download(symbol, start=start_date)["Close"].astype(int).reset_index()
    df.columns = ["ds", "y"]
    df.insert(0, "unique_id", "1.0")
    
    print(f"Training models for {asset}...")
    nf = NeuralForecast(models=models, freq="D")
    nf.fit(df=df, val_size=horizon)
    
    print(f"Generating predictions for {asset}...")
    Y_hat_df = nf.predict()
    Y_hat_df = Y_hat_df.assign(**{col: gaussian_filter1d(Y_hat_df[col], sigma=2)
                                  for col in Y_hat_df.select_dtypes(include='number').columns})

    # Add Polymarket-implied forecast as an extra (non-neural) model alongside the ensemble.
    print(f"Adding Polymarket-implied forecast for {asset}...")
    pm_cfg = module.get_polymarket()
    pm_events = pm.discover_events(pm_cfg["query"], pm_cfg["match"])
    pm_df = pm.forecast_path(df[["ds", "y"]], pm_events, horizon)
    Y_hat_df = Y_hat_df.merge(pm_df, on="ds", how="left")

    all_predictions[asset] = {
        'data': df,
        'predictions': Y_hat_df,
        'models': [type(model).__name__ for model in models] + [pm.POINT_COL]
    }

if __name__ == "__main__":
    print("All predictions generated.")
    from update_plot import create_plots
    create_plots(all_predictions)