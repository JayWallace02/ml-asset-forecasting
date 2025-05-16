
# Define the objective function
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import os
import logging
import os
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Import the scheduler
import optuna
import json

# Create a function to save the metrics after each trial






def load_data(path_to_validation_data,path_to_training_data):
    """
    Load training and validation data.
    """
    train_data = pd.read_csv(path_to_training_data)
    val_data = pd.read_csv(path_to_validation_data)
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


    print(train_data[["Date", "time_idx"]])
    print(train_data["time_idx"].diff().value_counts())
    
    
    
    
    # Convert 'Date' to datetime if not already
    val_data["Date"] = pd.to_datetime(val_data["Date"])
    val_data['Target_Month'] = val_data['Target_Month'].astype(str)  # Convert Target_Month to string

    # Extract year and month
    val_data["YearMonth"] = val_data["Date"].dt.to_period("M")  # Keeps year and month only

    # Drop duplicates for the same year and month (if any)
    val_data = val_data.drop_duplicates(subset=["YearMonth"]).reset_index(drop=True)

    # Recompute time_idx
    val_data["time_idx"] = range(len(val_data))  # Ensure consistent consecutive indexing
    val_data["Group_ID"] = "THIS ASSET"  # Add a dummy 'month' column with a constant value


    print(val_data[["Date", "time_idx"]])
    print(val_data["time_idx"].diff().value_counts())

    return train_data , val_data

def objective(trial,validaion_data_path,train_data_path,Target,time_known_reals,tensor_save_path,Tensor_logging_name):
    # Suggest hyperparameters
    max_encoder_length = trial.suggest_int("max_encoder_length", 5, 24)
    max_decoder_length = 1  # Keep decoder length fixed for forecasting the next step
    hidden_size = trial.suggest_int("hidden_size", 8, 1500)
    attention_head_size = trial.suggest_int("attention_head_size", 1, 1500)
    dropout = trial.suggest_float("dropout", 0.000001, 0.5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 5e-3)
    lstm_layers = trial.suggest_int("lstm_layers", 1, 20)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    hidden_continuous_size = trial.suggest_int("hidden_continuous_size", 8, 5000)
    
    
    train_data,val_data = load_data(path_to_validation_data=validaion_data_path,path_to_training_data=train_data_path)
    assert "time_idx" in train_data.columns, "time_idx is missing in train_data"
    assert "time_idx" in val_data.columns, "time_idx is missing in val_data"
    assert train_data["time_idx"].is_monotonic_increasing, "time_idx should be monotonically increasing"
    assert val_data["time_idx"].is_monotonic_increasing, "time_idx should be monotonically increasing"
        
    # Update dataset with the new encoder length
    train_dataset = TimeSeriesDataSet(
        train_data,
        time_idx="time_idx",  # Sequential time index
        target=Target,  # Target variable for regression
        group_ids=["Group_ID"],  # Grouping by asset class
        max_encoder_length=max_encoder_length,  # Lookback window
        max_prediction_length=max_decoder_length,  # Prediction window
        time_varying_known_reals=time_known_reals,
        time_varying_unknown_reals=[Target],  # Target variable to predict
        time_varying_known_categoricals=["Target_Month"],
 
        add_encoder_length=True,  # Add encoder length feature
    )
    
    
    # Define the TimeSeriesDataSet for validation
    val_dataset = TimeSeriesDataSet(
        val_data,
        time_idx="time_idx",  # Sequential time index
        target=Target,  # Target variable for regression
        group_ids=["Group_ID"],  # Grouping by asset class
        max_encoder_length=max_encoder_length,  # Lookback window
        max_prediction_length=max_decoder_length,  # Prediction window
        time_varying_known_reals=time_known_reals,
        time_varying_unknown_reals=[Target],  # Target variable to predict
        time_varying_known_categoricals=["Target_Month"],

        add_encoder_length=True,  # Add encoder length feature
    )
    
    # Create dataloaders
    train_dataloader = train_dataset.to_dataloader(train=True, batch_size=batch_size)
    val_dataloader = val_dataset.to_dataloader(train=False, batch_size=batch_size)
    print(len(val_dataloader.dataset))  # Check the size of validation dataset

    
    model = TemporalFusionTransformer.from_dataset(
        train_dataset,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            lstm_layers=lstm_layers,
            dropout=dropout,
            hidden_continuous_size=hidden_continuous_size,  # Optimize this
            loss=QuantileLoss(),
            log_interval=1,
            
        )
    

    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=10, verbose=False, mode="min"
    )
    
      # Set up TensorBoard logger
    logger = TensorBoardLogger(save_dir=tensor_save_path, name=Tensor_logging_name)
    trainer = Trainer(
        max_epochs=60,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback],
        enable_checkpointing=False,
        accelerator="gpu",
        logger = logger,
        devices=1,
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
        # Extract validation loss
    val_loss = trainer.callback_metrics["val_loss"].item()
    trial.report(val_loss, step=0)

    if trial.should_prune():
        raise optuna.TrialPruned()

    return val_loss
 
 
 
#  path to validation data
path_to_validation_data = "" 
# path to training data
path_to_training_data= ""
# target (want to predict)
Target = ""
# varibales used (Non Catigorical) Amned for each asset class
time_varying_known_reals=[
        "CPI_actual", "Unemployemnt Rate", "1-Year T-Bill Rate",
        "10-Year T-Bill Rate", "PMI_Actual", "CPI_Forecast_of_this_month",
        "Asset_Return_this_month", "Asset_Vol_this_month"
    ]
# Path to save Tensor logs
Tensor_save_path = ""
# Name of asset_class and what is being predicted
tensor_logging_name = ""



# # Optimize hyperparameters
study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
study.optimize(
    lambda trial: objective(
        trial,
        validaion_data_path=path_to_validation_data,
        train_data_path=path_to_training_data,
        Target=Target,
        time_known_reals=time_varying_known_reals,
        tensor_save_path=Tensor_save_path,
        Tensor_logging_name=tensor_logging_name
    ),
    n_trials=1000
)

# Print best parameters
print("Best hyperparameters: ", study.best_params)

# Save the best parameters to a JSON file
with open("best_hyperparameters_bond_Return.json", "w") as f:
    json.dump(study.best_params, f, indent=4)

print("Best hyperparameters saved to 'best_hyperparameters.json'")



