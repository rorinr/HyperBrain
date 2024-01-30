import matplotlib.pyplot as plt


def plot_loss_curve(
    loss_values: list,
    title: str = "Loss Curve",
    xlabel: str = "Number of Data Points",
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
