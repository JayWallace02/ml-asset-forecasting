import os
import yaml
import json
import tensorflow as tf

# Custom function to handle unknown YAML tags
def unknown_constructor(loader, tag_suffix, node):
    return f"Unsupported tag: {tag_suffix}"

# Add a constructor for unknown tags
yaml.add_multi_constructor("tag:yaml.org,2002:", unknown_constructor, Loader=yaml.FullLoader)

# Path to the directory containing all trials
log_dir = r""

# Initialize variables to track the smallest validation loss and corresponding trial details
smallest_val_loss = float("inf")
smallest_val_loss_trial = None
smallest_val_loss_step = None
smallest_val_loss_params = {}

# Iterate through all subdirectories (each subdirectory is assumed to represent a trial)
for root, dirs, files in os.walk(log_dir):
    trial_name = os.path.basename(root)  # Assuming each trial is in its own subdirectory
    for file in files:
        if file.startswith("events.out.tfevents"):  # Match TensorBoard log files
            log_file_path = os.path.join(root, file)
            print(f"Processing file: {log_file_path}")
            # Iterate through the events in the log file
            for event in tf.compat.v1.train.summary_iterator(log_file_path):
                for value in event.summary.value:
                    if "val_loss" in value.tag:  # Assuming validation loss is logged with "val_loss"
                        current_val_loss = value.simple_value
                        if current_val_loss < smallest_val_loss:
                            # Update the smallest validation loss and corresponding trial details
                            smallest_val_loss = current_val_loss
                            smallest_val_loss_trial = trial_name
                            smallest_val_loss_step = event.step

# If a trial with the smallest validation loss was found, read its hyperparameters
if smallest_val_loss_trial is not None:
    hparams_file = os.path.join(log_dir, smallest_val_loss_trial, "hparams.yaml")
    if os.path.exists(hparams_file):
        with open(hparams_file, "r") as f:
            try:
                smallest_val_loss_params = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as e:
                print(f"Error reading hparams.yaml: {e}")

    # Save the hyperparameters to a JSON file
    json_file_path = os.path.join(log_dir, f"{smallest_val_loss_trial}_hparams.json")
    with open(json_file_path, "w") as json_file:
        json.dump(smallest_val_loss_params, json_file, indent=4)
    print(f"Hyperparameters saved to {json_file_path}")

    # Display the result
    print(f"Trial with the smallest validation loss: {smallest_val_loss_trial}")
    print(f"Smallest validation loss: {smallest_val_loss}")
    print(f"At step: {smallest_val_loss_step}")
else:
    print("Validation loss not found in any trial.")
