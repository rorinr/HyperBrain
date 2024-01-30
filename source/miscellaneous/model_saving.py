import os
import json
import torch


def format_hyperparameter_name(hyperparameters: dict) -> str:
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


def save_model(
    models: dict, 
    hyperparameters: dict, 
    coarse_loss_history: list, 
    fine_loss_history: list, 
    base_path: str = "../../models/"
) -> str:
    """
    Saves the models, hyperparameters, and loss histories in a uniquely named directory based on the hyperparameters.

    This function formats the hyperparameters into a directory name, checks for existing directories with similar names,
    and applies versioning to avoid overwriting. Each model's state dictionary, the hyperparameters, and the loss histories 
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
    # Format directory name from hyperparameters and implement versioning
    dir_name = format_hyperparameter_name(hyperparameters)
    version = 1
    while os.path.exists(os.path.join(base_path, f"{dir_name}_v{version}")):
        version += 1
    final_dir = os.path.join(base_path, f"{dir_name}_v{version}")

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
