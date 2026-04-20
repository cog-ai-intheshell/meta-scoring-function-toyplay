import numpy as np


# helpers ------------------------------------------------------------------

def _as_float_array(values):
    """Convertit une suite numerique en tableau NumPy flottant."""
    return np.asarray(values, dtype=float)


def _safe_divide(numerator, denominator):
    """Effectue une division sure en renvoyant 0.0 si le denominateur est nul."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


# Gain metrics -------------------------------------------------------------

def mean_return(returns):
    """Rendement moyen par periode."""
    returns = _as_float_array(returns)
    if returns.size == 0:
        return 0.0
    return float(np.mean(returns))


def cumulative_return(returns):
    """Rendement cumule multiplicatif."""
    returns = _as_float_array(returns)
    if returns.size == 0:
        return 0.0
    return float(np.prod(1.0 + returns) - 1.0)


def volatility(returns, annualization_factor=None):
    """Volatilite empirique des rendements."""
    returns = _as_float_array(returns)
    if returns.size == 0:
        return 0.0

    volatility_value = float(np.std(returns))

    if annualization_factor is not None:
        volatility_value *= float(np.sqrt(annualization_factor))

    return volatility_value


def downside_volatility(returns, min_acceptable_return=0.0, annualization_factor=None):
    """Volatilite des rendements situes sous un seuil minimal."""
    returns = _as_float_array(returns)
    if returns.size == 0:
        return 0.0

    downside = np.minimum(returns - min_acceptable_return, 0.0)
    downside_value = float(np.sqrt(np.mean(downside ** 2)))

    if annualization_factor is not None:
        downside_value *= float(np.sqrt(annualization_factor))

    return downside_value


def sharpe_ratio(returns, risk_free_rate=0.0, annualization_factor=None):
    """Rendement excedentaire moyen rapporte a la volatilite totale."""
    returns = _as_float_array(returns)
    if returns.size == 0:
        return 0.0

    mean_excess_return = mean_return(returns) - risk_free_rate
    volatility_value = volatility(returns)
    sharpe_value = _safe_divide(mean_excess_return, volatility_value)

    if annualization_factor is not None:
        sharpe_value *= float(np.sqrt(annualization_factor))

    return float(sharpe_value)


def sortino_ratio(returns, min_acceptable_return=0.0, annualization_factor=None):
    """Rendement excedentaire moyen rapporte a la volatilite baissiere."""
    returns = _as_float_array(returns)
    if returns.size == 0:
        return 0.0

    mean_excess_return = mean_return(returns) - min_acceptable_return
    downside_value = downside_volatility(returns, min_acceptable_return=min_acceptable_return)
    sortino_value = _safe_divide(mean_excess_return, downside_value)

    if annualization_factor is not None:
        sortino_value *= float(np.sqrt(annualization_factor))

    return float(sortino_value)


def max_drawdown(returns, initial_value=1.0):
    """Perte maximale entre un sommet historique et le creux qui suit."""
    returns = _as_float_array(returns)
    if returns.size == 0:
        return 0.0

    portfolio_values = np.concatenate(
        ([initial_value], initial_value * np.cumprod(1.0 + returns))
    )
    peaks = np.maximum.accumulate(portfolio_values)
    drawdowns = np.divide(
        peaks - portfolio_values,
        peaks,
        out=np.zeros_like(portfolio_values, dtype=float),
        where=peaks != 0,
    )

    return float(np.max(drawdowns))


def hit_rate_trade(trade_returns, include_zero_as_win=False):
    """Proportion de trades gagnants parmi les trades executes."""
    trade_returns = _as_float_array(trade_returns)
    if trade_returns.size == 0:
        return 0.0

    if include_zero_as_win:
        wins = np.sum(trade_returns >= 0.0)
    else:
        wins = np.sum(trade_returns > 0.0)

    return float(_safe_divide(wins, trade_returns.size))


def profit_factor(trade_returns):
    """Ratio entre gains bruts et pertes brutes."""
    trade_returns = _as_float_array(trade_returns)
    if trade_returns.size == 0:
        return 0.0

    gross_gains = float(np.sum(np.maximum(trade_returns, 0.0)))
    gross_losses = float(np.sum(np.maximum(-trade_returns, 0.0)))

    if gross_losses == 0.0:
        if gross_gains > 0.0:
            return float("inf")
        return 0.0

    return float(gross_gains / gross_losses)


def market_reference_quantile(market_reference_gains, quantile=0.75):
    """Quantile de reference des gains de marche admissibles sur une fenetre."""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile doit appartenir a [0, 1]")

    market_reference_gains = _as_float_array(market_reference_gains)
    market_reference_gains = market_reference_gains[np.isfinite(market_reference_gains)]
    if market_reference_gains.size == 0:
        return 0.0

    return float(np.quantile(market_reference_gains, quantile))


def r_eff_i(realized_gain, market_reference_gains=None, market_q75=None, quantile=0.75, epsilon=1e-6):
    """Calcule R_eff_i = Gain_reel^(i) / Q_0.75(M_i) avec garde-fou sur le denominateur."""
    if market_q75 is None:
        if market_reference_gains is None:
            raise ValueError("market_reference_gains ou market_q75 doit etre fourni")
        market_q75 = market_reference_quantile(market_reference_gains, quantile=quantile)

    realized_gain = float(realized_gain)
    market_q75 = float(market_q75)

    if not np.isfinite(realized_gain) or not np.isfinite(market_q75) or market_q75 <= float(epsilon):
        return float("nan")

    return float(realized_gain / market_q75)


def trade_return_series(y_pred, y_true, tp_gain, fp_loss):
    """Retourne les rendements des seuls trades executes sur une fenetre."""
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    if y_pred.shape != y_true.shape:
        raise ValueError("y_pred et y_true doivent avoir la meme forme")

    trade_mask = y_pred == 1

    if not np.any(trade_mask):
        return np.asarray([], dtype=float)

    trade_truth = y_true[trade_mask]
    return np.where(trade_truth == 1, float(tp_gain), -float(fp_loss)).astype(float)


def _normalize_no_opportunity_effective(realized_return, window_size, fp_loss):
    """Normalise la qualite d'une fenetre sans opportunite entre pire perte et zero trade."""
    window_size = int(window_size)
    if window_size <= 0:
        return 1.0

    worst_trade_returns = np.full(window_size, -float(fp_loss), dtype=float)
    worst_return = cumulative_return(worst_trade_returns)

    if abs(worst_return) < 1e-12:
        return 1.0

    normalized = (float(realized_return) - worst_return) / (0.0 - worst_return)
    return float(np.clip(normalized, 0.0, 1.0))


def balance_metric_dict(y_pred, y_true, balance_start, tp_gain, fp_loss):
    """Construit le bloc des metriques de gain pour une fenetre."""
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    balance_start = float(balance_start)

    trade_returns = trade_return_series(y_pred, y_true, tp_gain=tp_gain, fp_loss=fp_loss)
    realized_return = cumulative_return(trade_returns)
    balance_end = balance_start * (1.0 + realized_return)
    gain_realized = balance_end - balance_start

    n_true_momentum = int(np.sum(y_true == 1))
    balance_max = balance_start * ((1.0 + float(tp_gain)) ** n_true_momentum)
    gain_max_possible = balance_max - balance_start

    if gain_max_possible == 0:
        gain_effective = _normalize_no_opportunity_effective(
            realized_return,
            window_size=y_true.size,
            fp_loss=fp_loss,
        )
    else:
        gain_effective = gain_realized / gain_max_possible

    return {
        "balance_start": balance_start,
        "balance_end": balance_end,
        "gain_realized": gain_realized,
        "gain_max_possible": gain_max_possible,
        "gain_effective": gain_effective,
        "n_true_momentum": n_true_momentum,
        "n_predicted_positive": int(np.sum(y_pred == 1)),
    }
