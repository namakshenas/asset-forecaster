from neuralforecast.models import TSMixer, NBEATS, NHITS, MLP, TiDE

# Brent crude oil: lower volatility and longer macro/supply cycles than crypto,
# so we widen the lookback, slow the learning rate, train longer, lower dropout,
# and shift NHITS frequency downsampling toward longer-horizon components.
def get_models(horizon):
    return [
        TSMixer(h=horizon, n_series=1, input_size=756, n_block=4, ff_dim=128, dropout=0.3,
                revin=True, max_steps=350, learning_rate=3e-4, scaler_type="robust", batch_size=64),
        NBEATS(h=horizon, input_size=756, max_steps=350, learning_rate=3e-4, scaler_type="robust",
                n_blocks=[3, 3], mlp_units=[[512, 512], [512, 512]], stack_types=["trend", "seasonality"], batch_size=64),
        NHITS(h=horizon, input_size=756, max_steps=350, learning_rate=3e-4, scaler_type="robust",
                n_freq_downsample=[60, 30, 1], interpolation_mode="linear", pooling_mode="MaxPool1d", activation="LeakyReLU", batch_size=64),
        MLP(h=horizon, input_size=756, max_steps=350, learning_rate=3e-4, scaler_type="robust",
            num_layers=2, hidden_size=512, batch_size=64),
        TiDE(h=horizon, input_size=756, max_steps=350, learning_rate=3e-4, scaler_type="robust",
                hidden_size=512, decoder_output_dim=32, temporal_decoder_dim=128, dropout=0.3, batch_size=64),
    ]

def get_symbol():
    return "BZ=F"
