from dataclasses import dataclass


@dataclass
class VerifierResult:
    passed: bool
    longest_consecutive_run: int
    window_size: int
    reason: str


def verify_sustained_threshold(
    values: list[float],
    threshold: float,
    min_consecutive: int = 3,
    direction: str = "above",
) -> VerifierResult:
    """
    Vérifie si `values` contient au moins `min_consecutive` points CONSÉCUTIFS
    dépassant `threshold` (direction="above") ou en-dessous (direction="below").

    IMPORTANT: cette logique doit rester identique à celle de
    `check_sustained_exceedance` dans metrics_mcp_server.py -- c'est le tool
    que l'agent utilise pour prendre sa décision, et ce verifier est censé la
    RE-vérifier de façon indépendante. Si les deux implémentent des critères
    différents (ex: ratio global vs run consécutif), l'agent et le verifier
    peuvent être en désaccord sur des cas où l'un des deux a raison et l'autre
    pas, sans qu'on sache lequel -- ça casse la garantie que le verifier
    apporte au système (cf. section 4.2 du paper CloudAnoAgent).
    """
    window_size = len(values)
    if window_size == 0:
        return VerifierResult(False, 0, 0, "fenêtre vide")

    def _satisfies(v: float) -> bool:
        return v > threshold if direction == "above" else v < threshold

    best_len = 0
    cur_len = 0
    for v in values:
        if _satisfies(v):
            cur_len += 1
            best_len = max(best_len, cur_len)
        else:
            cur_len = 0

    passed = best_len >= min_consecutive
    reason = (
        f"plus long run consécutif = {best_len}/{window_size} points "
        f"{'>' if direction == 'above' else '<'} {threshold} "
        f"(min requis: {min_consecutive} points consécutifs)"
    )
    return VerifierResult(passed, best_len, window_size, reason)


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
    """
    Médiane + MAD (Median Absolute Deviation, échelle normale via *1.4826).

    Cas MAD=0 (>=50% des valeurs identiques à la médiane -- fréquent sur des
    métriques à faible amplitude type net_in/disk_io qui restent souvent
    pinnées près de 0) : l'ancien fallback `abs(median)*0.1 or 1.0` était
    arbitraire et particulièrement fragile quand median est proche de 0 (le
    seuil retombait sur une constante fixe de 1.0 sans rapport avec l'échelle
    réelle de la métrique). On utilise ici l'écart-type classique comme
    estimateur de repli -- moins robuste que le MAD face aux outliers, mais
    au moins cohérent avec la dispersion réelle des données. Seulement si
    l'écart-type est AUSSI nul (série parfaitement constante) on retombe sur
    un epsilon plancher, pour éviter une division par zéro en aval.
    """
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 0:
        return 0.0, 1.0

    median = sorted_vals[n // 2]
    abs_dev = sorted(abs(v - median) for v in values)
    mad = abs_dev[n // 2]

    if mad > 0:
        mad_scaled = mad * 1.4826
    else:
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        std = variance**0.5
        mad_scaled = std if std > 0 else max(abs(median) * 0.05, 0.01)

    return median, mad_scaled


def verify_metrics_agent_output(
    agent_output: dict,
    raw_values: list[float],
    baseline_values: list[float] | None = None,
    min_consecutive: int = 3,
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

    `min_consecutive` doit rester égal au `min_consecutive` utilisé par
    l'agent via check_sustained_exceedance (3 par défaut, cf. prompt.py
    Étape 2bis) -- c'est ce qui garantit que l'agent et le verifier jugent
    la consécutivité de la même façon.
    """
    if not agent_output.get("is_anomaly"):
        return {"verified": True, "final_is_anomaly": False, "reason": "agent n'a pas déclaré d'anomalie"}

    pattern = agent_output.get("pattern")
    reference = baseline_values if baseline_values else raw_values
    median, mad_scaled = _robust_stats(reference)

    threshold_high = median + 3 * mad_scaled
    threshold_low = median - 3 * mad_scaled

    if pattern in ("spike", "gradual_increase"):
        result = verify_sustained_threshold(
            raw_values, threshold_high, direction="above", min_consecutive=min_consecutive
        )
    elif pattern in ("dip", "gradual_decrease"):
        result = verify_sustained_threshold(
            raw_values, threshold_low, direction="below", min_consecutive=min_consecutive
        )
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