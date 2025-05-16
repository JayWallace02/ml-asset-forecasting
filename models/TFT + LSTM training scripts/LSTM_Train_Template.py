import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import logging
import json
import optuna

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss, Metric
import lightning.pytorch as pl



class LSTMModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2, learning_rate=1e-3, quantiles=(0.05, 0.5, 0.95)):
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


def run_training(seq_length,num_layers,hidden_size,dropout,learning_rate, asset_class,target, train_data_path, val_data_path,logger_path,output_path):
    model_name = asset_class + "_"+ target
    
    x_return_this_month = asset_class +"_Return_this_month"
    x_vol_this_month = asset_class +"_Vol_this_month"
     # Select input features and target
     
     
     
    # Re-configure to require input features
    input_features = ["CPI_actual", "Unemployemnt Rate", "1-Year T-Bill Rate",
            "10-Year T-Bill Rate", "PMI_Actual", "CPI_Forecast_of_this_month",
            x_return_this_month, x_vol_this_month]
    target_column = target
        

    input_size = len(input_features)  # Number of input features
    
    
    seq_length = seq_length
    num_layers = num_layers
    hidden_size = hidden_size
    dropout = dropout
    learning_rate = learning_rate
    
    
    
    
    print("Starting data load...")
        # Load the data
    train_data = pd.read_csv(train_data_path)
    val_data = pd.read_csv(val_data_path)

    # Convert 'Date' to datetime
    train_data["Date"] = pd.to_datetime(train_data["Date"])
    val_data["Date"] = pd.to_datetime(val_data["Date"])

    # Drop the 'Direction' column
    train_data = train_data.drop(columns=["Direction"])
    val_data = val_data.drop(columns=["Direction"])


    # Normalize features and target
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    train_data[input_features] = scaler_features.fit_transform(train_data[input_features])
    val_data[input_features] = scaler_features.transform(val_data[input_features])
    train_data[target_column] = scaler_target.fit_transform(train_data[target_column].values.reshape(-1, 1))
    val_data[target_column] = scaler_target.transform(val_data[target_column].values.reshape(-1, 1))

    # Prepare sequences

    train_input_data = train_data[input_features].values
    train_target_data = train_data[target_column].values
    X_train, y_train = create_sequences(train_input_data, train_target_data, seq_length)

    val_input_data = val_data[input_features].values
    val_target_data = val_data[target_column].values
    X_val, y_val = create_sequences(val_input_data, val_target_data, seq_length)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    # Create DataLoader for training and validation
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    
    
    
    
    
    model = LSTMModel(input_size, hidden_size, num_layers, dropout,learning_rate)
    
    
    
    # Initialize PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=1000,
        
        accelerator="gpu" if torch.cuda.is_available() else "cpu",

        enable_progress_bar=True,
        logger=pl.loggers.TensorBoardLogger(logger_path, name=model_name),
        callbacks=[EarlyStopping(monitor="val_loss", patience=60)],  # Monitor val_loss
    )
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    
    # Save the trained model
    save_path = os.path.join(output_path, f"{model_name}.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure directory exists
    torch.save(model.state_dict(), save_path)
    print(f"Model training complete. Saved to: {output_path}")
    
        
        
def main():
    
    # Bes params identified during para finding phase 
    seq_length = 1
    num_layers = 1
    hidden_size = 1
    dropout = 0.1
    learning_rate = 1
    
    # Asset class being being trained on
    asset_class =""
    # What is being predicted
    target= ""
    # train_data_paath
    train_data_path= r""
    # Validation data path
    val_data_path=r""
    # Tensoor logs path
    logger_path= r"TensorLogs\LSTM"
    # Models save path
    output_path=r""
    
    
    
    run_training(seq_length,num_layers,hidden_size,dropout,learning_rate, asset_class,target, train_data_path, val_data_path,logger_path,output_path)
    
    
if __name__ == "__main__":
    main()
    
    
    
        
        
        