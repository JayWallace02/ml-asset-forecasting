
import pandas as pd
import torch
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data import GroupNormalizer
import matplotlib.pyplot as plt



def load_model(model_path, train_data, target_variable , asset,max_encoder):
    """
    Load the Temporal Fusion Transformer (TFT) model with the given architecture and saved weights.
    """
    
    
    return_this_month = asset +"_Return_this_month"
    vol_this_month = asset +"_Vol_this_month"

    train_dataset = TimeSeriesDataSet(
        train_data,
        time_idx="time_idx",
        target=target_variable,
        group_ids=["Group_ID"],
        max_encoder_length=max_encoder,
        max_prediction_length=1,
        time_varying_known_reals=[
            "CPI_actual", "Unemployemnt Rate", "1-Year T-Bill Rate",
            "10-Year T-Bill Rate", "PMI_Actual", "CPI_Forecast_of_this_month",
            return_this_month, vol_this_month
        ],
        time_varying_unknown_reals=[target_variable],
        time_varying_known_categoricals=["Target_Month"],
        target_normalizer=GroupNormalizer(groups=["Group_ID"]),
        add_encoder_length=True,
    )
    # please amned as same parameters of loaded model
    tft = TemporalFusionTransformer.from_dataset(
        train_dataset,
        hidden_size=1,
        attention_head_size=1,
        lstm_layers=1,
        dropout=0.1,
        hidden_continuous_size=1,
        loss=QuantileLoss(quantiles=[0.05, 0.5, 0.95]),
        optimizer="Ranger",
        reduce_on_plateau_patience=10,
    )

    # Load saved weights
    tft.load_state_dict(torch.load(model_path))
    tft.eval()  # Set the model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tft.to(device)

    return tft






def plot_predictions(dates, actuals, lower_bound, median, upper_bound):
    """
    Plot the predictions (with intervals) against actual values from Nov 2017 onward.
    """
    # Convert dates to datetime if not already
    dates = pd.to_datetime(dates)
    
    # Filter for dates >= 2017-11-01
    mask = dates >= pd.to_datetime("2017-11-01")
    dates = dates[mask]
    actuals = np.array(actuals)[mask]
    lower_bound = np.array(lower_bound)[mask]
    median = np.array(median)[mask]
    upper_bound = np.array(upper_bound)[mask]

    plt.figure(figsize=(14, 7))
    plt.plot(dates, actuals, label="Actual", color="blue", linewidth=2, marker="o")
    plt.plot(dates, median, label="Predicted Median (50%)", color="orange", linewidth=2, marker="x")
    plt.fill_between(dates, lower_bound, upper_bound, color="orange", alpha=0.3, label="Prediction Interval (5%-95%)")

    plt.title("Actual vs. Predicted Values with Prediction Intervals (Post-Nov 2017)")
    plt.xlabel("Date")
    plt.ylabel("Target Value")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
def save_predictions_to_csv(dates, actuals, lower_bound, median, upper_bound, file_path):
    """
    Save the predictions and actual values to a CSV file, only from Nov 2017 onward.
    """
    # Convert dates to datetime if not already
    dates = pd.to_datetime(dates)

    # Filter for dates >= 2017-11-01
    mask = dates >= pd.to_datetime("2024-07-01")

    # Create DataFrame with filtered values
    results_df = pd.DataFrame({
        "Date": dates[mask],
        "Actual": np.array(actuals)[mask],
        "Predicted_5%": np.array(lower_bound)[mask],
        "Predicted_50%": np.array(median)[mask],
        "Predicted_95%": np.array(upper_bound)[mask]
    })

    # Save to CSV
    results_df.to_csv(file_path, index=False)
    print(f"Predictions saved to {file_path}")
    
def rolling_predict(model, full_data, target_variable, asset, max_encoder, start_month, end_month):
    full_data["Date"] = pd.to_datetime(full_data["Date"])
    full_data["Target_Month"] = full_data["Target_Month"].astype(str)
    full_data["Group_ID"] = "THIS ASSET"

    start_date = pd.to_datetime(start_month)
    end_date = pd.to_datetime(end_month)

    results = []

    current_date = start_date
    while current_date >= end_date:
        encoder_start = current_date - pd.DateOffset(months=max_encoder)
        encoder_data = full_data[(full_data["Date"] >= encoder_start) & (full_data["Date"] < current_date)].copy()

        if len(encoder_data) < max_encoder:
            print(f"Skipping {current_date.strftime('%Y-%m')} due to insufficient encoder history.")
            current_date -= pd.DateOffset(months=1)
            continue

        # Add target month row for prediction (future known values only)
        predict_row = full_data[full_data["Date"] == current_date].copy()
        if predict_row.empty:
            print(f"Skipping {current_date.strftime('%Y-%m')} because target row is missing.")
            current_date -= pd.DateOffset(months=1)
            continue

        combined = pd.concat([encoder_data, predict_row], ignore_index=True)
        combined["time_idx"] = range(len(combined))  # fresh indexing

        # Build dataset for this window
        return_this_month = f"{asset}_Return_this_month"
        vol_this_month = f"{asset}_Vol_this_month"

        dataset = TimeSeriesDataSet(
            combined,
            time_idx="time_idx",
            target=target_variable,
            group_ids=["Group_ID"],
            max_encoder_length=max_encoder,
            max_prediction_length=1,
            time_varying_known_reals=[
                "CPI_actual", "Unemployemnt Rate", "1-Year T-Bill Rate",
                "10-Year T-Bill Rate", "PMI_Actual", "CPI_Forecast_of_this_month",
                return_this_month, vol_this_month
            ],
            time_varying_unknown_reals=[target_variable],
            time_varying_known_categoricals=["Target_Month"],
            target_normalizer=GroupNormalizer(groups=["Group_ID"]),
            add_encoder_length=True,
        )

        dl = dataset.to_dataloader(train=False, batch_size=1, num_workers=8,)

        predictions = model.predict(dl, mode="quantiles",trainer_kwargs=dict(accelerator="gpu")).cpu().numpy()
        print(f"Original Predictions shape: {predictions.shape}")  # (2, 1, 3)

        # Correct reshaping clearly:
        predictions = predictions[:, 0, :]  # remove batch dimension explicitly
        print(f"Reshaped Predictions shape: {predictions.shape}")  # Should now be (2,3)

        # Take the LAST timestep predictions (your prediction horizon = 1 step ahead)
        last_step_preds = predictions[-1]

        # Append clearly:
        results.append({
            "Date": current_date,
            "Predicted_5%": last_step_preds[0],
            "Predicted_50%": last_step_preds[1],
            "Predicted_95%": last_step_preds[2],
            "Actual": predict_row[target_variable].values[0]
        })



        current_date -= pd.DateOffset(months=1)

    return pd.DataFrame(results)


def main():
    # Paths
    # Model saved path
    model_path = r""
    # trianing data path
    train_data_path = r""
    # data being tested and predicted
    val_data_path = r""
    # path to save predictions
    output_csv_path = r""
    # Target variable
    target_variable = ""
    # Assset class of model
    asset = ""
    # look back period ( Please amend the paramers in the load_model() function)
    max_encoder = 7
    
    start_month=""  # adjust as needed
    end_month=""
    
    # Load training data to initialize the model
    train_data = pd.read_csv(train_data_path)
    # Convert 'Date' to datetime if not already
    train_data["Date"] = pd.to_datetime(train_data["Date"])
    train_data['Target_Month'] = train_data['Target_Month'].astype(str)  # Convert Target_Month to string

    # Extract year and month
    train_data["YearMonth"] = train_data["Date"].dt.to_period("M")  # Keeps year and month only

    # Drop duplicates for the same year and month (if any)
    train_data = train_data.drop_duplicates(subset=["YearMonth"]).reset_index(drop=True)

    # Recompute time_idx
    train_data["time_idx"] = range(len(train_data))  # Ensure consistent consecutive indexing
    train_data["Group_ID"] = "THIS ASSET"  # Add a dummy 'month' column with a constant value

    # Load the trained model
    print("Loading model...")
    tft = load_model(model_path, train_data, target_variable,asset ,max_encoder=max_encoder)

    # Prepare validation data
    print("Preparing validation data...")
    val_data = pd.read_csv(val_data_path)




   # Run rolling predictions (this replaces your missing functions)
    print("Running rolling predictions...")
    prediction_results = rolling_predict(
        model=tft,
        full_data=val_data,
        target_variable=target_variable,
        asset=asset,
        max_encoder=max_encoder,
        start_month=start_month,  # adjust as needed
        end_month=end_month
    )

    # Save the rolling prediction results to CSV
    print("Saving rolling predictions to CSV...")
    prediction_results.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")

    # Optionally, plot predictions
    plot_predictions(
        prediction_results["Date"],
        prediction_results["Actual"],
        prediction_results["Predicted_5%"],
        prediction_results["Predicted_50%"],
        prediction_results["Predicted_95%"]
    )




if __name__ == "__main__":
    main()
