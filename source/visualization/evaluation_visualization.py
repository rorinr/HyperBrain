import matplotlib.pyplot as plt
from typing import List, Dict

def plot_loss_curve(
    loss_values: list,
    title: str = "Loss Curve",
    xlabel: str = "Batch No.",
    ylabel: str = "Loss",
):
    """
    Plots a loss curve from a list of loss values.

    Args:
        loss_values (list): A list of floats representing the loss values.
        title (str): The title of the plot. Defaults to "Loss Curve".
        xlabel (str): The label for the x-axis. Defaults to "Number of Data Points".
        ylabel (str): The label for the y-axis. Defaults to "Loss".
    """
    # Create a range for the x-axis based on the length of the loss values
    x_values = list(range(1, len(loss_values) + 1))

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, loss_values, marker="o", color="b", linestyle="-")

    # Add title and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Optional: Add a grid for easier readability
    plt.grid(True)

    # Show the plot
    plt.show()

def plot_precision_vs_threshold(*predictors: Dict[float, float], labels: List[str]) -> None:
    """
    Plot precision versus pixel threshold for multiple predictors.

    Args:
        predictors: A variable number of dictionaries, each representing a different predictor,
                    where keys are pixel thresholds and values are precision scores.
        labels: A list of labels for each predictor, in the same order as the predictors.

    Returns:
        None: This function only plots the data and does not return any value.
    """
    plt.figure(figsize=(10, 6))  # Set the figure size for better readability
    for predictor, label in zip(predictors, labels):
        # Unpack the pixel thresholds and precision scores from each predictor
        thresholds, precisions = zip(*predictor.items())
        plt.plot(thresholds, precisions, label=label)  # Add marker for clarity
    
    plt.title('Precision vs Pixel Threshold')  # Title of the plot
    plt.xlabel('Pixel Threshold')  # Label for the x-axis
    plt.ylabel('Precision')  # Label for the y-axis
    plt.legend()  # Show legend to identify predictors
    plt.grid(True)  # Add grid for easier reading of the plot
    plt.show()  # Display the plot