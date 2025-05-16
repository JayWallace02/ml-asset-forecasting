import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna
import torch.optim as optim

class LSTMModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2, learning_rate=1.5589606213416575e-05, quantiles=(0.05, 0.5, 0.95)):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, len(quantiles))  # Output size matches the number of quantiles
        self.learning_rate = learning_rate
        self.quantiles = quantiles

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the last time step's output
        return out

    def quantile_loss(self, predictions, targets):
        """
        Pinball loss for quantile regression.
        """
        # Ensure targets have shape [batch_size, 1]
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)  # Add a second dimension

        # Expand targets to match predictions' shape
        targets = targets.expand_as(predictions)  # Shape: [batch_size, num_quantiles]

        losses = []
        for i, q in enumerate(self.quantiles):
            errors = targets[:, i] - predictions[:, i]
            losses.append(torch.max((q - 1) * errors, q * errors).mean())
        return torch.stack(losses).mean()


    def calculate_metrics(self, predictions, targets):
        """
        Calculate RMSE, MSE, and SMAPE using the 50th percentile (median) prediction.
        """
        median_predictions = predictions[:, 1]  # Index 1 corresponds to the 50th percentile
        mse = nn.MSELoss()(median_predictions, targets)
        rmse = torch.sqrt(mse)
        smape = self.calculate_smape(median_predictions, targets)
        return mse, rmse, smape

    @staticmethod
    def calculate_smape(predictions, targets):
        """
        Calculate SMAPE (Symmetric Mean Absolute Percentage Error)
        """
        denominator = (torch.abs(predictions) + torch.abs(targets)) / 2.0
        diff = torch.abs(predictions - targets)
        smape = torch.mean(diff / denominator) * 100
        return smape

    def training_step(self, batch, batch_idx):
        X, y = batch
        predictions = self(X)

        # Calculate quantile loss
        loss = self.quantile_loss(predictions, y.unsqueeze(1).expand(-1, len(self.quantiles)))

        # Calculate other metrics
        mse, rmse, smape = self.calculate_metrics(predictions, y)

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_mse", mse, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_rmse", rmse, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_smape", smape, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        predictions = self(X)

        # Debugging shape mismatch
        print(f"Predictions shape: {predictions.shape}")
        print(f"Targets shape: {y.shape}")

       # Compute quantile loss
        # Calculate quantile loss
        val_loss = self.quantile_loss(predictions, y.unsqueeze(1).expand(-1, len(self.quantiles)))

        # Calculate other metrics
        mse, rmse, smape = self.calculate_metrics(predictions, y)

        # Log metrics
        self.log("val_loss", val_loss, prog_bar=True, logger=True)
        self.log("val_mse", mse, prog_bar=True, logger=True)
        self.log("val_rmse", rmse, prog_bar=True, logger=True)
        self.log("val_smape", smape, prog_bar=True, logger=True)

        return val_loss


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # ReduceLROnPlateau will reduce LR by a factor of 0.5 if val_loss does not improve for x epochs
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20, verbose=True)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # Monitor validation loss
                "interval": "epoch",
                "frequency": 1
            }
        }

 
def create_sequences(input_data, target_data, seq_length):
    sequences = []
    labels = []
    for i in range(len(input_data) - seq_length):
        seq = input_data[i:i + seq_length]  # Sequence of features
        label = target_data[i + seq_length]  # Corresponding target value
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)


def validate_lstm_predictions(p5, p50, p95, actuals_rescaled, val_data):
    """
    Align LSTM predictions (quantiles) with actuals and dates from val_data.
    """
    # Match dates from the end of val_data
    matched_dates = val_data["Date"].iloc[-len(p50):].reset_index(drop=True).values
    actuals = actuals_rescaled

    # Check alignment
    if not (len(p50) == len(actuals) == len(matched_dates)):
        raise ValueError("Mismatch in prediction and date alignment.")

    # Print some sample output
    print("Date\t\t\t5%\t\t50%\t\t95%\t\tActual")
    for i in range(min(5, len(p50))):
        print(f"{matched_dates[i]}:\t{p5[i]:.4f}\t{p50[i]:.4f}\t{p95[i]:.4f}\t{actuals[i]:.4f}")

    return p5, p50, p95, actuals, matched_dates

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_predictions(dates, actuals, lower_bound, median, upper_bound):
    """
    Plot the predictions (with intervals) against actual values using date-based x-axis.
    """
    plt.figure(figsize=(14, 6))

    # Plot actual values
    plt.plot(dates, actuals, label="Actual", color="black", linewidth=1)

    # Plot median predictions
    plt.plot(dates, median, label="Median Prediction (50%)", color="blue", linewidth=1.5)

    # Plot prediction intervals
    plt.fill_between(dates, lower_bound, upper_bound, color="blue", alpha=0.2, label="Prediction Interval (5%–95%)")

    # Format x-axis to look cleaner
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    # ax.xaxis.set_major_locator(mdates.YearLocator(2))  # Every 2 years
    plt.xticks(rotation=45)

    # Grid and layout
    plt.title("LSTM Model Predictions with Intervals")
    plt.xlabel("Date")
    plt.ylabel("Target Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # === Load and preprocess the validation data ===

    train_data = pd.read_csv(r"")
    val_data = pd.read_csv(r"")
    # asset class of model being used/predicted
    asset_class = ""
    # the models target or whaat is being predicted
    target_column = ""
    path_of_model = ""
    path_to_save_predictions = ""
    
    # Model parameters Must be same as the trained model
    seq_length = 1
    
    num_layers = 1
    hidden_size = 1

    dropout = 0.1
 
    
    ############ Ignore below ###############
    
    
    # input features Amend as see fit 
    x_return_this_month = asset_class +"_Return_this_month"
    x_vol_this_month = asset_class +"_Vol_this_month"

    input_features = ["CPI_actual", "Unemployemnt Rate", "1-Year T-Bill Rate",
            "10-Year T-Bill Rate", "PMI_Actual", "CPI_Forecast_of_this_month",
            x_return_this_month, x_vol_this_month]
    
    
    
    

    input_size = len(input_features)
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaler_target = MinMaxScaler(feature_range=(0, 1))

    # For proper scaling, load the same scaler or re-fit on train data and apply here
    
    scaler_features.fit(train_data[input_features])
    scaler_target.fit(train_data[target_column].values.reshape(-1, 1))

    val_data[input_features] = scaler_features.transform(val_data[input_features])
    val_data[target_column] = scaler_target.transform(val_data[target_column].values.reshape(-1, 1))

    
    val_input_data = val_data[input_features].values
    val_target_data = val_data[target_column].values
    print("Validation data rows:", len(val_data))
    max_seq_length = len(val_data) - 1
    print("Maximum sequence length that still gives at least one sequence:", max_seq_length)

    X_val, y_val = create_sequences(val_input_data, val_target_data, seq_length)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

   # === Load the trained model weights from .ptt ===
   
    model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    model.load_state_dict(torch.load(path_of_model))
    model.eval()


 

    all_predictions = []
    all_actuals = []

    with torch.no_grad():
        for batch in val_loader:
            X_batch, y_batch = batch
            predictions = model(X_batch)  # Shape: [1, 3]
            all_predictions.append(predictions.cpu().numpy())  
            all_actuals.append(y_batch.cpu().numpy())

    # Concatenate predictions correctly
    all_predictions = np.concatenate(all_predictions, axis=0)  # Now shape: [N, 3]
    all_actuals = np.concatenate(all_actuals, axis=0)          # Now shape: [N]

    print("Final predictions shape:", all_predictions.shape)  #  should [N, 3]
    # Inverse transform each quantile
    p5 = scaler_target.inverse_transform(all_predictions[:, 0:1]).flatten()
    p50 = scaler_target.inverse_transform(all_predictions[:, 1:2]).flatten()
    p95 = scaler_target.inverse_transform(all_predictions[:, 2:3]).flatten()
    all_actuals_rescaled = scaler_target.inverse_transform(all_actuals.reshape(-1, 1)).flatten()
    print("p50 shape:", p50.shape, "Sample:", p50[:5])
    print("p5 shape:", p5.shape, "Sample:", p5[:5])
    print("p95 shape:", p95.shape, "Sample:", p95[:5])
    print("Actuals shape:", all_actuals_rescaled.shape, "Sample:", all_actuals_rescaled[:5])
    


    
    p5, p50, p95, actuals, matched_dates = validate_lstm_predictions(p5, p50, p95, all_actuals_rescaled, val_data)
    predictions_df = pd.DataFrame({
    "Date": matched_dates,
    "Actual": actuals,
    "Lower_Quantile": p5,
    "Median_Prediction": p50,
    "Upper_Quantile": p95
})
    
    
    # Filter from Nov 2017 onward
    filtered_df = predictions_df[predictions_df["Date"] >= "2017-11-01"].reset_index(drop=True)
        # Ensure Date is datetime
    filtered_df["Date"] = pd.to_datetime(filtered_df["Date"])

    # Sort by Date (chronological order)
    filtered_df = filtered_df.sort_values("Date").reset_index(drop=True)



    
    # === Save predictions to CSV ===
    output_path = path_to_save_predictions
    filtered_df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
    print(filtered_df.head())
    plot_predictions(
    dates=filtered_df["Date"],
    actuals=filtered_df["Actual"],
    lower_bound=filtered_df["Lower_Quantile"],
    median=filtered_df["Median_Prediction"],
    upper_bound=filtered_df["Upper_Quantile"]
)


