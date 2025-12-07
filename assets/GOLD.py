from neuralforecast.models import TSMixer, NBEATS, NHITS, MLP, TiDE

def get_models(horizon):
    return [
        TSMixer(h=horizon, n_series=1, input_size=504, n_block=4, ff_dim=128, dropout=0.5,
                revin=True, max_steps=350, learning_rate=5e-4, scaler_type="robust", batch_size=32),
        NBEATS(h=horizon, input_size=504, max_steps=350, learning_rate=5e-4, scaler_type="robust",
                n_blocks=[3, 3], mlp_units=[[512, 512], [512, 512]], stack_types=["trend", "seasonality"], batch_size=32),
        NHITS(h=horizon, input_size=504, max_steps=350, learning_rate=1e-3, scaler_type="robust",
                n_freq_downsample=[24, 12, 1], interpolation_mode="linear", pooling_mode="MaxPool1d", activation="LeakyReLU", batch_size=32),
        MLP(h=horizon, input_size=252, max_steps=350, learning_rate=1e-3, scaler_type="robust",
            num_layers=2, hidden_size=512, batch_size=32),
        TiDE(h=horizon, input_size=504, max_steps=350, learning_rate=5e-4, scaler_type="robust",
                hidden_size=512, decoder_output_dim=32, temporal_decoder_dim=128, dropout=0.5, batch_size=32),
    ]

def get_symbol():
    return "GC=F"