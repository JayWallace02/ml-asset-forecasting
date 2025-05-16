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
        return optim.Adam(self.parameters(), lr=self.learning_rate)
    
    
    
def create_sequences(input_data, target_data, seq_length):
    sequences = []
    labels = []
    for i in range(len(input_data) - seq_length):
        seq = input_data[i:i + seq_length]  # Sequence of features
        label = target_data[i + seq_length]  # Corresponding target value
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)


def objective(trial,tensor_path,Name_of_asset_class_and_target,target,train_data_path,val_data_path, input_features):
    try:
    
        input_features = input_features
         
        
        target_column = target
        
        #adjsut based on system capailbitiers 
        input_size = len(input_features)  # Number of input features
        seq_length = trial.suggest_int("seq_length", 1, 80)
  
        num_layers = trial.suggest_int("num_layers", 1, 10)
        hidden_size = trial.suggest_int("hidden_size", 32, 1000)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
        
        
        
        print("Starting data load...")
        # Load the data
        train_data = pd.read_csv(train_data_path)
        val_data = pd.read_csv(val_data_path)

        # Convert 'Date' to datetime
        train_data["Date"] = pd.to_datetime(train_data["Date"])
        val_data["Date"] = pd.to_datetime(val_data["Date"])


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
        
        
        # handles when look back period exceeds the available validation data
        if X_val.shape[0] == 0:
            print(f"Skipping trial with seq_length = {seq_length} due to empty validation set.")
            return None
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        # Define model
        input_size = len(input_features)  # Number of input features
        
        
       
        
        model = LSTMModel(input_size, hidden_size, num_layers, dropout,learning_rate)
        
        
        # Initialize PyTorch Lightning Trainer
        trainer = pl.Trainer(
            max_epochs=100,
            log_every_n_steps=10,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
    
            enable_progress_bar=True,
            logger=pl.loggers.TensorBoardLogger(tensor_path, name=Name_of_asset_class_and_target),
            callbacks=[EarlyStopping(monitor="val_loss", patience=10)],  # Monitor val_loss
        )
        
        # Train the model
        trainer.fit(model, train_loader, val_loader)
        print(f"Callback metrics: {trainer.callback_metrics}")

        # Get Validation Loss
        val_loss = trainer.callback_metrics["val_loss"].item()
        trial.report(val_loss, step=0)

        if trial.should_prune():
            raise optuna.TrialPruned()

        return val_loss
    
    except Exception as e:
        print(f"Error in objective function: {e}")
        raise


def run_Return():
    
    # path where training data is located
    training_data_path = r""
    # path where validation data is located
    validation_data_path = r""
    # path where tensor logs will be stored (V. Important) will use this to find best parameteres post training
    Tensorlog_path = r""
    # Unique name of what asset class and what is the target(want prediction)
    name_of_asset_class_and_target = ""
    # Relevant target (wanted prediction)
    target = ""
    
    # Select input features and target
    # amend to relevant features and asset clasee 
    input_features = ["CPI_actual", "Unemployemnt Rate", "1-Year T-Bill Rate",
            "10-Year T-Bill Rate", "PMI_Actual", "CPI_Forecast_of_this_month",
            "Asset_Return_this_month", "Asset_Vol_this_month"]
    
     # Define the storage path for the Optuna study amend name 
    study_storage_path = "sqlite:///optuna_.db"

    try:
        # Try to load an existing study
        study = optuna.load_study(
            study_name="optuna_LSTM",
            storage=study_storage_path,
        )
        print("Study loaded successfully.")
    except KeyError:
        # If the study does not exist, create a new one
        study = optuna.create_study(
            study_name="optuna_LSTM",
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(),
            storage=study_storage_path,
        )
        print("New study created.")
        
    # Optimize study
    study.optimize(
    lambda trial: objective(trial, tensor_path=Tensorlog_path,
                            Name_of_asset_class_and_target=name_of_asset_class_and_target,
                            target=target,
                            train_data_path=training_data_path,
                            val_data_path=validation_data_path,
                            input_features=input_features),
    n_trials=1000
)


    # Print best parameters
    print("Best hyperparameters: ", study.best_params)

    # Save the best parameters
    with open("best_hyperparameters_LSTM_Bond_Return.json", "w") as f:
        json.dump(study.best_params, f, indent=4)

    print("Best hyperparameters saved to 'best_hyperparameters_LSTM_Bond_Return.json'")

if __name__ == "__main__":

    
    
    
    
    
    
    run_Return()
    