
from dataclasses import dataclass


@dataclass
class VerifierResult:
    passed: bool
    points_above_threshold: int
    window_size: int
    reason: str


def verify_sustained_threshold(
    values: list[float],
    threshold: float,
    k_min_ratio: float = 0.8,
    direction: str = "above",
) -> VerifierResult:
    
    window_size = len(values)
    if window_size == 0:
        return VerifierResult(False, 0, 0, "fenêtre vide")

    if direction == "above":
        count = sum(1 for v in values if v > threshold)
    else:
        count = sum(1 for v in values if v < threshold)

    k_min = int(window_size * k_min_ratio)
    passed = count >= k_min

    reason = (
        f"{count}/{window_size} points {'>' if direction == 'above' else '<'} {threshold} "
        f"(seuil requis: {k_min} points, ratio={k_min_ratio})"
    )
    return VerifierResult(passed, count, window_size, reason)


def verify_fluctuation(values: list[float], std_ratio_threshold: float = 0.3) -> VerifierResult:
    
    n = len(values)
    if n < 2:
        return VerifierResult(False, 0, n, "pas assez de points")

    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = variance**0.5
    ratio = std / mean if mean != 0 else 0

    passed = ratio > std_ratio_threshold
    reason = f"écart-type relatif={ratio:.3f} (seuil={std_ratio_threshold})"
    return VerifierResult(passed, n, n, reason)


def _robust_stats(values: list[float]) -> tuple[float, float]:
    """Médiane + MAD (Median Absolute Deviation, échelle normale via *1.4826)."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    median = sorted_vals[n // 2] if n else 0
    abs_dev = sorted([abs(v - median) for v in values])
    mad = abs_dev[n // 2] if n else 0
    mad_scaled = mad * 1.4826 if mad > 0 else (abs(median) * 0.1 or 1.0)
    return median, mad_scaled


def verify_metrics_agent_output(
    agent_output: dict,
    raw_values: list[float],
    baseline_values: list[float] | None = None,
) -> dict:
    """
    Point d'entrée principal: prend la sortie JSON du Metrics Agent, la
    fenêtre courante (raw_values) et une fenêtre de référence "normale"
    antérieure (baseline_values). Le seuil est calculé sur la baseline,
    PAS sur raw_values lui-même -- sinon un spike prolongé finit par
    devenir sa propre "normalité" statistique et n'est plus détectable.

    Si aucune baseline n'est fournie, on retombe sur raw_values (mode
    dégradé, utile seulement pour anomalies courtes/minoritaires dans
    la fenêtre).
    """
    if not agent_output.get("is_anomaly"):
        return {"verified": True, "final_is_anomaly": False, "reason": "agent n'a pas déclaré d'anomalie"}

    pattern = agent_output.get("pattern")
    reference = baseline_values if baseline_values else raw_values
    median, mad_scaled = _robust_stats(reference)

    threshold_high = median + 3 * mad_scaled
    threshold_low = median - 3 * mad_scaled

    if pattern in ("spike", "gradual_increase"):
        result = verify_sustained_threshold(raw_values, threshold_high, direction="above", k_min_ratio=0.3)
    elif pattern in ("dip", "gradual_decrease"):
        result = verify_sustained_threshold(raw_values, threshold_low, direction="below", k_min_ratio=0.3)
    elif pattern == "fluctuation":
        result = verify_fluctuation(raw_values)
    else:
        return {"verified": False, "final_is_anomaly": False, "reason": f"pattern inconnu: {pattern}"}

    return {
        "verified": result.passed,
        "final_is_anomaly": result.passed,
        "reason": result.reason,
        "threshold_used": round(threshold_high if pattern in ("spike", "gradual_increase") else threshold_low, 3),
    }
