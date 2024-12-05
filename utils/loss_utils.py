import numpy as np

def calculate_delays(signal_length, sampling_frequency):
    """
    Calculate delays for a given signal length and sampling frequency.

    Parameters:
    - signal_length: Length of the signal (number of samples).
    - sampling_frequency: Sampling frequency (Hz).

    Returns:
    - delays: Array of delay values.
    """
    return np.arange(signal_length) / sampling_frequency

def normalize_delays(delays):
    """
    Normalize delays globally.

    Parameters:
    - delays: Array of delay values.

    Returns:
    - normalized_delays: Globally normalized delays.
    """
    return (delays - np.min(delays)) / (np.max(delays) - np.min(delays))

def calculate_mean_excess_delay(normalized_delays, power_values):
    """
    Calculate the mean excess delay as the delay where the main peak (maximum power) occurs.

    Parameters:
    - normalized_delays: Normalized delay values.
    - power_values: Array of power values corresponding to the delays.

    Returns:
    - mean_excess_delay: Mean excess delay.
    """
    max_power_index = np.argmax(power_values)  # Index of the maximum power
    return normalized_delays[max_power_index]

def calculate_delay_spread(normalized_delays, power_values, mean_excess_delay):
    """
    Calculate the delay spread.

    Parameters:
    - normalized_delays: Normalized delay values.
    - power_values: Array of power values corresponding to the delays.
    - mean_excess_delay: Mean excess delay.

    Returns:
    - delay_spread: Delay spread.
    """
    return np.sqrt(
        np.sum(power_values * (normalized_delays - mean_excess_delay) ** 2) / np.sum(power_values)
    )

def calculate_loss(real_signal, generated_signal, sampling_frequency, power_real, power_gen):
    """
    Calculate the loss between real and generated signals based on mean excess delay and delay spread.

    Parameters:
    - real_signal: Array of real signal values.
    - generated_signal: Array of generated signal values.
    - sampling_frequency: Sampling frequency (Hz).
    - power_real: Array of power values for the real signal.
    - power_gen: Array of power values for the generated signal.

    Returns:
    - loss: Calculated loss value.
    """
    # Calculate delays
    tau_real = calculate_delays(len(real_signal), sampling_frequency)
    tau_gen = calculate_delays(len(generated_signal), sampling_frequency)

    # Normalize delays
    tau_real_normalized = normalize_delays(tau_real)
    tau_gen_normalized = normalize_delays(tau_gen)

    # Calculate mean excess delay
    mean_excess_delay_real = calculate_mean_excess_delay(tau_real_normalized, power_real)
    mean_excess_delay_gen = calculate_mean_excess_delay(tau_gen_normalized, power_gen)

    # Calculate delay spread
    delay_spread_real = calculate_delay_spread(tau_real_normalized, power_real, mean_excess_delay_real)
    delay_spread_gen = calculate_delay_spread(tau_gen_normalized, power_gen, mean_excess_delay_gen)

    # Calculate loss
    loss = 0.5 * abs(mean_excess_delay_real - mean_excess_delay_gen) + \
           0.5 * abs(delay_spread_real - delay_spread_gen)

    return loss