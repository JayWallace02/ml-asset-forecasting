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
import optuna.visualization as vis
import matplotlib
import traceback  # For detailed exception logging
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from pytorch_forecasting.data import GroupNormalizer

def setup_logger(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),  # Log to console
        ],
    )


def load_data(train_data_path,val_data_path):
    """
    Load training and validation data.
    """
    train_data = pd.read_csv(train_data_path)
    val_data = pd.read_csv(val_data_path)
    
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



def run_tft(target_variable , model_output_path, train_data_path,val_data_path,time_varing_know,Tensor_save_path,asset_class):
    
    # Set up logging
    
  
    """
    Runs the Temporal Fusion Transformer (TFT) pipeline: preprocessing, training, and saving the model.

    Args:
        target_variable (str): The target variable to predict.
        model_output_path (str): Path to save the trained model.
    """
    try:
        # Fixed parameters from the previous optimization
        # Best parameters identified during tuning phase
        max_encoder_length = 1
        hidden_size = 1
        attention_head_size = 1
        dropout = 0.1
    
        lstm_layers = 1
        batch_size = 1
        hidden_continuous_size = 1
        max_decoder_length = 1
    

        train_data, val_data = load_data(train_data_path=train_data_path,val_data_path=val_data_path)
        
        
         
         
         
#


        # Define the training dataset
        train_dataset = TimeSeriesDataSet(
            train_data,
            time_idx="time_idx",  # Sequential time index
            target=target_variable,  # Target variable for regression
            group_ids=["Group_ID"],  # Grouping by asset class
            max_encoder_length=max_encoder_length,  # Lookback window
            max_prediction_length=max_decoder_length,  # Prediction window
            time_varying_known_reals=time_varing_know,
            time_varying_unknown_reals=[target_variable],  # Target variable to predict
            time_varying_known_categoricals=["Target_Month"],
            target_normalizer=GroupNormalizer(
            groups=["Group_ID"]
             ),

            add_encoder_length=True,  # Add encoder length feature
            
    
  
        )

        # Define the validation dataset
        val_dataset = TimeSeriesDataSet(
            val_data,
            time_idx="time_idx",  # Sequential time index
            target=target_variable,  # Target variable for regression
            group_ids=["Group_ID"],  # Grouping by asset class
            max_encoder_length=max_encoder_length,  # Lookback window
            max_prediction_length=max_decoder_length,  # Prediction window
            time_varying_known_reals=time_varing_know,
            time_varying_unknown_reals=[target_variable],  # Target variable to predict
            time_varying_known_categoricals=["Target_Month"],

            add_encoder_length=True,  # Add encoder length feature
     
 
        )

        # Detect GPU availability
        devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        logging.info(f"Detected {devices} device(s). Using accelerator: {accelerator}.")

        # Step 3: Define model hyperparameters
        logging.info("Initializing Temporal Fusion Transformer.")
 

    # Create the TFT model
        tft = TemporalFusionTransformer.from_dataset(
            train_dataset,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            lstm_layers=lstm_layers,
            dropout=dropout,
            hidden_continuous_size=hidden_continuous_size,  # Optimize this
            
            loss=QuantileLoss(quantiles=[0.05,0.5, 0.95]),
            optimizer="Ranger",  # Explicitly specify optimizer
            reduce_on_plateau_patience=10,
        )


        # DataLoaders
        train_dataloader = train_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=16, persistent_workers=False)
        val_dataloader = val_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=16, persistent_workers=False)


        # TensorBoard Logger
        logger = TensorBoardLogger(save_dir=Tensor_save_path, name= asset_class +"_TFT_Model_" + target_variable)
        

        early_stopping = EarlyStopping(
            monitor="val_loss",  # Metric to monitor
            patience=50,  # Stop training if no improvement after 10 epochs
            mode="min",  # Minimize validation loss
        )
        
        # Trainer Configuration
        trainer = Trainer(
            max_epochs=1000,
            devices="auto", 
            logger=logger,
            callbacks=[early_stopping,
            
                LearningRateMonitor("epoch")
            ],
            enable_checkpointing=True,
        )

        # Step 5: Train the model

        logging.info("Starting model training.")
        print("Training the model...")
        trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        # Save the trained model
        save_path = os.path.join(model_output_path, f"{asset_class}_{target_variable}.pt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure directory exists
        torch.save(tft.state_dict(), save_path)
        print(f"Model training complete. Saved to: {model_output_path}")
        
        
        
        # Save model architecture to JSON
        architecture = {
            "hidden_size": hidden_size,
            "attention_head_size": attention_head_size,
            "lstm_layers": lstm_layers,
            "dropout": dropout,
            "hidden_continuous_size": hidden_continuous_size,
            "max_encoder_length": max_encoder_length,
            "max_decoder_length": max_decoder_length,
            "time_varying_known_reals": train_dataset.time_varying_known_reals,
            "time_varying_unknown_reals": train_dataset.time_varying_unknown_reals,
            "time_varying_known_categoricals": train_dataset.time_varying_known_categoricals,
        }

        json_path = os.path.join(model_output_path, f"{target_variable}_architecture.json")
        with open(json_path, "w") as f:
            json.dump(architecture, f, indent=4)
        print(f"Model architecture saved to: {json_path}")
    except Exception as e:
        logging.error(f"An error occurred during training: {e}")
        logging.error("Stack trace:", exc_info=True)  # Log stack trace for debugging
        
        
       
        

def main():

    try:
    # Specify the target variable and model output path
    #  !!!!! AMEND PARAMETERS IN run_tft() function
    
        # The target (whatt is being predicted)
        target = ""
        # path to save model too
        model_save_path = ""
        # training data path
        training_data_path = ""
        # validation data path
        validation_data_path = ""
        # Time varying input features (excluding catagorical features) and for each asset class
        time_varying_known_reals=[
                "CPI_actual", "Unemployemnt Rate", "1-Year T-Bill Rate",
                "10-Year T-Bill Rate", "PMI_Actual", "CPI_Forecast_of_this_month",
                "Asset_Return_this_month", "Asset_Vol_this_month"
            ]
        # Path to save tensor log too
        Tensor_save_path = ""
        # Asset class of traiing
        asset_class = ""
        


        print(f"Starting TFT training for target: {target}")
        run_tft(target_variable=target,model_output_path=model_save_path,train_data_path=training_data_path,val_data_path=validation_data_path,
                         time_varing_know=time_varying_known_reals,Tensor_save_path=Tensor_save_path,asset_class=asset_class)

        print("TFT training completed successfully!")
    except Exception as e:
        print(f"An error occurred during model training: {e}")

    
    
    
# Ensure the script runs only when executed directly
if __name__ == "__main__":
    main()