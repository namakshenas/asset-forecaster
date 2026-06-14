import plotly.graph_objects as go

from polymarket import POINT_COL


def create_plots(all_predictions):
    for asset, data in all_predictions.items():
        df = data['data']
        Y_hat_df = data['predictions']
        models = data['models']

        print(f"Generating figure for {asset}...")
        fig = go.Figure()
        recent_data = df.tail(60)
        fig.add_trace(go.Scatter(x=recent_data["ds"], y=recent_data["y"],
                                name="Actual", line=dict(color="black", width=3)))

        for model in models:
            # Make the Polymarket forecaster stand out from the neural ensemble lines.
            line = dict(width=3, dash="dash", color="#1f77b4") if model == POINT_COL else dict(width=2)
            fig.add_trace(go.Scatter(x=Y_hat_df["ds"], y=Y_hat_df[model],
                                    name=model, line=line))

        fig.update_layout(title=f"{asset} Price Prediction - All Models",
                                    xaxis_title="Date", yaxis_title="Price (USD)",
                                    hovermode="x unified", template="plotly_white")

        fig.write_image(f"predictions/{asset.lower()}.png", width=1400, height=800)
        fig.show()
