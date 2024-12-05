import matplotlib.pyplot as plt

def plot_tx_rx_locations(tx_loc, rx_x=None, rx_y=None):
    """
    Scatter plot TX and RX locations.

    Parameters:
    - tx_loc: Tuple (x, y) for the TX location.
    - rx_x: List or array of X coordinates for RX locations (optional).
    - rx_y: List or array of Y coordinates for RX locations (optional).

    Returns:
    - None
    """
    plt.figure(figsize=(8, 8))
    if rx_x is not None and rx_y is not None:
        plt.scatter(rx_x, rx_y, color="blue", label="RX Locations")
    plt.scatter(tx_loc[0], tx_loc[1], color="red", label="TX Location", marker="x", s=200)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("TX and RX Locations")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_real_vs_generated_signals(real_signal, generated_signal, title="Real vs Generated Signals"):
    """
    Stem plot for real and generated signals.

    Parameters:
    - real_signal: Array or list of values for the real signal.
    - generated_signal: Array or list of values for the generated signal.
    - title: Title for the plot (default: "Real vs Generated Signals").

    Returns:
    - None
    """
    if len(real_signal) != len(generated_signal):
        raise ValueError("Real and generated signals must have the same length.")

    plt.figure(figsize=(10, 6))
    markerline_real, stemlines_real, baseline_real = plt.stem(real_signal, label="Real Signal", linefmt='b-', markerfmt='bo', basefmt=' ')
    markerline_gen, stemlines_gen, baseline_gen = plt.stem(generated_signal, label="Generated Signal", linefmt='r--', markerfmt='ro', basefmt=' ')

    plt.setp(stemlines_real, 'linewidth', 1.5)
    plt.setp(stemlines_gen, 'linewidth', 1.5)

    plt.xlabel("Index")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
