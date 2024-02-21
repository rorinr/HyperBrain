import os
import json
import torch


def format_hyperparameter_name(hyperparameters: dict) -> str:
    # DEPRECATED
    """
    Formats hyperparameters into a string suitable for use as a directory name.

    This function takes a dictionary of hyperparameters, converts each value into a string,
    and concatenates them into a single string separated by underscores. For floating point values,
    the decimal point is replaced with 'p' to ensure file system compatibility.

    Args:
        hyperparameters (dict): A dictionary containing the hyperparameters.

    Returns:
        str: A string representation of the hyperparameters.
    """
    formatted_params = []
    for key, value in hyperparameters.items():
        formatted_value = (
            str(value).replace(".", "p") if isinstance(value, float) else str(value)
        )
        formatted_params.append(f"{key}{formatted_value}")
    return "_".join(formatted_params)

def generate_next_id(parent_directory):
    """
    Scan the parent directory to find the highest existing ID and
    generate the next ID for the new model directory.

    Args:
    parent_directory (str): The path to the parent directory containing model directories.

    Returns:
    int: The next ID to use for a new model directory.
    """
    # Ensure the parent directory exists to avoid errors
    if not os.path.exists(parent_directory):
        os.makedirs(parent_directory)
        return 1

    # List all items in the parent directory
    directories = os.listdir(parent_directory)
    
    # Filter out items that are not directories or cannot be converted to integers
    valid_ids = [int(dir_name) for dir_name in directories if dir_name.isdigit() and os.path.isdir(os.path.join(parent_directory, dir_name))]
    
    # Find the highest existing ID if any exist, otherwise start with 0
    highest_id = max(valid_ids) if valid_ids else 0
    
    # Generate the next ID
    next_id = highest_id + 1
    
    return next_id


def save_model(
    models: dict,
    hyperparameters: dict,
    coarse_loss_history: list,
    fine_loss_history: list,
    base_path: str = "../../models/",
) -> str:
    """
    Saves the models, hyperparameters, and loss histories in a uniquely named directory.

    This functio checks for existing directories with similar names,
    and applies indexing to avoid overwriting. Each model's state dictionary, the hyperparameters, and the loss histories
    are saved in this directory.

    Args:
        models (dict): A dictionary of models to save, where keys are model names.
        hyperparameters (dict): A dictionary containing the hyperparameters used for the models.
        coarse_loss_history (list): A list containing the history of coarse loss values.
        fine_loss_history (list): A list containing the history of fine loss values.
        base_path (str): The base path where the model directories will be created. Defaults to '../../models/'.

    Returns:
        str: The path to the directory where the models, hyperparameters, and loss histories are saved.
    """

    index = generate_next_id(base_path)
    final_dir = os.path.join(base_path, f"{index}")

    # Create the directory and save models
    os.makedirs(final_dir, exist_ok=True)
    for model_name, model in models.items():
        torch.save(model.state_dict(), os.path.join(final_dir, f"{model_name}.pt"))

    # Save hyperparameters
    with open(os.path.join(final_dir, "details.json"), "w") as f:
        json.dump(hyperparameters, f)

    # Save loss histories
    with open(os.path.join(final_dir, "coarse_loss_history.json"), "w") as f:
        json.dump(coarse_loss_history, f)
    with open(os.path.join(final_dir, "fine_loss_history.json"), "w") as f:
        json.dump(fine_loss_history, f)

    return final_dir
