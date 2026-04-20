import numpy as np

from sklearn.datasets import load_digits


def _normalize_signal(signal):
    signal = np.asarray(signal, dtype=float)
    max_abs = np.max(np.abs(signal))
    if max_abs < 1e-12:
        return np.zeros_like(signal)
    return signal / max_abs


def _smoothed_noise(rng, size, smooth_window):
    noise = rng.normal(0.0, 1.0, size)

    if smooth_window <= 1:
        return _normalize_signal(noise)

    kernel = np.ones(int(smooth_window), dtype=float) / float(smooth_window)
    smooth_noise = np.convolve(noise, kernel, mode="same")
    return _normalize_signal(smooth_noise)


def _run_lengths(binary_values):
    binary_values = np.asarray(binary_values, dtype=int)
    if binary_values.size == 0:
        return {0: [], 1: []}

    lengths = {0: [], 1: []}
    current_value = int(binary_values[0])
    current_length = 1

    for value in binary_values[1:]:
        value = int(value)
        if value == current_value:
            current_length += 1
            continue

        lengths[current_value].append(current_length)
        current_value = value
        current_length = 1

    lengths[current_value].append(current_length)
    return lengths


def sequence_statistics(momentum_labels):
    momentum_labels = np.asarray(momentum_labels, dtype=int)
    run_lengths = _run_lengths(momentum_labels)
    switches = int(np.sum(momentum_labels[1:] != momentum_labels[:-1])) if momentum_labels.size > 1 else 0

    return {
        "n_samples": int(momentum_labels.size),
        "momentum_rate": float(np.mean(momentum_labels)) if momentum_labels.size else 0.0,
        "switches": switches,
        "avg_run_momentum": float(np.mean(run_lengths[1])) if run_lengths[1] else 0.0,
        "avg_run_zero": float(np.mean(run_lengths[0])) if run_lengths[0] else 0.0,
        "max_run_momentum": int(max(run_lengths[1])) if run_lengths[1] else 0,
        "max_run_zero": int(max(run_lengths[0])) if run_lengths[0] else 0,
    }


def build_original_temporal_digits():
    digits = load_digits()
    original_index = np.arange(len(digits.target), dtype=int)
    digit_value = digits.target.astype(int)
    momentum = (digit_value >= 5).astype(int)

    return {
        "data": digits.data,
        "original_index": original_index,
        "digit_value": digit_value,
        "momentum": momentum,
        "sampling_weight": np.full(len(digit_value), np.nan, dtype=float),
        "sequence_stats": sequence_statistics(momentum),
        "mode": "original",
    }


def load_temporal_digits(
    mode="original",
    random_state=42,
    jitter=0.22,
    smooth_window=48,
    min_probability=0.05,
    max_probability=0.95,
):
    """Charge le dataset en mode temporel original ou stochastic."""
    if mode == "original":
        return build_original_temporal_digits()

    if mode == "stochastic":
        return build_stochastic_temporal_digits(
            random_state=random_state,
            jitter=jitter,
            smooth_window=smooth_window,
            min_probability=min_probability,
            max_probability=max_probability,
        )

    raise ValueError(f"Mode temporel inconnu: {mode}")


def build_stochastic_temporal_digits(
    random_state=42,
    jitter=0.22,
    smooth_window=48,
    min_probability=0.05,
    max_probability=0.95,
):
    digits = load_digits()
    rng = np.random.default_rng(random_state)

    original_index = np.arange(len(digits.target), dtype=int)
    digit_value = digits.target.astype(int)
    momentum = (digit_value >= 5).astype(int)

    momentum_indices = original_index[momentum == 1].copy()
    zero_indices = original_index[momentum == 0].copy()
    rng.shuffle(momentum_indices)
    rng.shuffle(zero_indices)

    base_rate = float(np.mean(momentum))
    signal = _smoothed_noise(rng, len(momentum), smooth_window=smooth_window)
    weights = np.clip(
        base_rate + float(jitter) * signal,
        float(min_probability),
        float(max_probability),
    )
    weights = weights / np.sum(weights)

    momentum_positions = rng.choice(
        len(momentum),
        size=len(momentum_indices),
        replace=False,
        p=weights,
    )

    stochastic_momentum = np.zeros(len(momentum), dtype=int)
    stochastic_momentum[momentum_positions] = 1

    reordered_index = np.empty(len(momentum), dtype=int)
    momentum_cursor = 0
    zero_cursor = 0

    for t, value in enumerate(stochastic_momentum):
        if value == 1:
            reordered_index[t] = momentum_indices[momentum_cursor]
            momentum_cursor += 1
        else:
            reordered_index[t] = zero_indices[zero_cursor]
            zero_cursor += 1

    return {
        "data": digits.data[reordered_index],
        "original_index": reordered_index,
        "digit_value": digit_value[reordered_index],
        "momentum": stochastic_momentum,
        "sampling_weight": weights,
        "sequence_stats": sequence_statistics(stochastic_momentum),
        "mode": "stochastic",
    }
