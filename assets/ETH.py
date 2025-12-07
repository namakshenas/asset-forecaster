from neuralforecast.models import TSMixer, NBEATS, NHITS, MLP, TiDE

def get_models(horizon):
    return [
        TSMixer(h=horizon, n_series=1, input_size=int(horizon*3), n_block=2, ff_dim=64, dropout=0.4, revin=True,
            scaler_type="identity", max_steps=400, learning_rate=1e-3, batch_size=32, early_stop_patience_steps=10),
        NBEATS(h=horizon, input_size=336, max_steps=350, learning_rate=1e-3, scaler_type="robust",
            n_blocks=[3, 3, 2], mlp_units=[[512, 512], [512, 512], [512, 512]], 
            stack_types=["trend", "seasonality", "identity"], batch_size=32),
        NHITS(h=horizon, input_size=720, max_steps=300, learning_rate=5e-3, scaler_type="robust",
            n_freq_downsample=[16, 8, 2, 1], n_blocks=[1, 1, 1, 1], mlp_units=[[512, 512]]*4,
            interpolation_mode="linear", pooling_mode="MaxPool1d", activation="ReLU", batch_size=64),
        MLP(h=horizon, input_size=336, max_steps=300, learning_rate=5e-3, scaler_type="robust",
            num_layers=4, hidden_size=512, batch_size=64),
        TiDE(h=horizon, input_size=400, max_steps=300, learning_rate=5e-3, scaler_type="robust",
            hidden_size=512, batch_size=64),
    ]

def get_symbol():
    return "ETH-USD"