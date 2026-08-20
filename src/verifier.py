from dataclasses import dataclass

from changepoint import detect_changepoint


@dataclass
class VerifierResult:
    passed: bool
    longest_consecutive_run: int
    window_size: int
    reason: str
    mean_excess_ratio: float | None = None


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
    best_start_idx = None
    cur_len = 0
    cur_start_idx = None
    for i, v in enumerate(values):
        if _satisfies(v):
            if cur_len == 0:
                cur_start_idx = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start_idx = cur_start_idx
        else:
            cur_len = 0
            cur_start_idx = None

    passed = best_len >= min_consecutive
    reason = (
        f"plus long run consécutif = {best_len}/{window_size} points "
        f"{'>' if direction == 'above' else '<'} {threshold} "
        f"(min requis: {min_consecutive} points consécutifs)"
    )

    # Même calcul que côté tool MCP (`evaluate_metric_window` exceedance.mean_excess_ratio
    # dans metrics_mcp_server.py) -- dupliqué ici volontairement pour la même raison que
    # compute_data_completeness ci-dessous: le verifier doit pouvoir recalculer l'ampleur
    # du dépassement de façon INDÉPENDANTE, pas juste recopier le chiffre de l'agent.
    mean_excess_ratio = None
    if best_start_idx is not None and threshold != 0:
        run_values = values[best_start_idx: best_start_idx + best_len]
        if direction == "above":
            excess_ratios = [(v - threshold) / abs(threshold) for v in run_values]
        else:
            excess_ratios = [(threshold - v) / abs(threshold) for v in run_values]
        mean_excess_ratio = round(sum(excess_ratios) / len(excess_ratios), 3)

    return VerifierResult(passed, best_len, window_size, reason, mean_excess_ratio)


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

    # Pas de notion de "run consécutif" pour une fluctuation (c'est le bruit
    # global de la fenêtre qui compte, pas une séquence de points) -- on
    # réutilise mean_excess_ratio pour porter le dépassement relatif du seuil
    # de variance, afin que compute_confidence() ait quand même un signal
    # d'ampleur pour ce pattern.
    mean_excess_ratio = round((ratio - std_ratio_threshold) / std_ratio_threshold, 3) if std_ratio_threshold != 0 else None
    return VerifierResult(passed, n, n, reason, mean_excess_ratio)


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


def compute_data_completeness(n_points_used: int, n_points_requested: int) -> float:
    """
    Même formule que côté tool MCP (`get_baseline_stats` dans
    metrics_mcp_server.py) -- dupliquée ici volontairement, pas importée,
    pour que le verifier puisse recalculer la complétude de façon
    INDÉPENDANTE à partir de ce qu'il observe lui-même (`baseline_values`),
    plutôt que de faire aveuglément confiance au chiffre recopié par
    l'agent dans sa sortie JSON. Garder cette fonction en synchro avec son
    équivalent dans metrics_mcp_server.py si la formule change un jour.
    """
    if n_points_requested <= 0:
        return 0.0
    return round(min(n_points_used / n_points_requested, 1.0), 2)


def compute_confidence(
    run_ratio: float,
    mean_excess_ratio: float | None,
    data_completeness: float,
) -> float:
    """
    Score de confiance CONTINU entre 0 et 1, en complément du booléen
    `final_is_anomaly` -- pas un remplacement.

    Pourquoi une formule déterministe plutôt qu'un score verbalisé par le LLM
    (ex: "confidence: 0.8" dans le JSON de l'agent): la littérature sur la
    calibration des LLM (voir ex. Xiong et al. 2024, Kadavath et al. 2022)
    montre de façon assez constante que la confiance auto-rapportée par un
    LLM est mal calibrée et souvent sur-confiante, surtout après instruction
    tuning -- exactement le même problème que `threshold_used` ou
    `points_above_threshold` inventés "au jugé" que ce projet évite déjà en
    forçant l'agent à recopier des chiffres calculés par les tools/le
    verifier plutôt que de les halluciner. Ce score suit le même principe:
    100% dérivé de statistiques calculées, jamais d'un jugement du LLM.

    À noter (cf. papers SAGE / AGENT-X sur la détection d'anomalies par
    LLM): un tel score doit être lu comme une force de preuve ORDINALE
    (evidence strength), pas comme une probabilité calibrée au sens
    statistique strict (ex: "0.8" ne veut pas dire "80% de chances d'être
    une vraie anomalie" sur un jeu de données donné) -- un calibrage formal
    (Platt/temperature scaling) nécessiterait un jeu de validation labellisé,
    hors scope ici. Utile telle quelle pour PONDÉRER/prioriser entre
    plusieurs verdicts (ex: fusion avec le Log Agent), pas pour un seuil de
    décision absolu.

    Trois composantes, combinées par moyenne pondérée puis bornées à [0, 1]:

    - run_component (poids 0.5): combien le run consécutif dépasse le
      minimum requis. run_ratio = longest_consecutive_run / min_consecutive.
      Mappé sur [0, 1] via min(run_ratio / 2, 1.0) -- un run pile au minimum
      (run_ratio=1.0) donne 0.5, un run deux fois plus long (run_ratio=2.0)
      ou plus sature à 1.0. Le facteur /2 est un choix arbitraire raisonnable
      (pas dérivé d'un jeu de validation) -- à ajuster si l'usage en aval
      montre qu'il faut être plus/moins généreux.
    - magnitude_component (poids 0.35): ampleur MOYENNE du dépassement par
      rapport au seuil (mean_excess_ratio, ex: 0.20 = 20% au-dessus du
      seuil en moyenne sur le run). Mappé sur [0, 1] via min(ratio / 0.5, 1.0)
      -- un dépassement de 50%+ sature à 1.0. Absent (None, ex: pattern
      "fluctuation" qui n'a pas de run à mesurer) -> composante omise et
      poids redistribué sur les deux autres plutôt que forcée à 0, pour ne
      pas punir injustement un pattern qui n'a simplement pas cette info.
    - completeness_component (poids 0.15): `data_completeness` de la
      baseline -- un verdict basé sur une baseline pauvre en points reste
      moins fiable même si le run observé est net.

    Args:
        run_ratio: longest_consecutive_run / min_consecutive (>= 0).
        mean_excess_ratio: dépassement relatif moyen du run par rapport au
            seuil, ou None si non applicable (ex: pattern "fluctuation").
        data_completeness: complétude de la baseline utilisée, entre 0 et 1.

    Returns:
        Score arrondi à 3 décimales, dans [0, 1].
    """
    components: list[tuple[float, float]] = []  # (valeur_dans_[0,1], poids)

    run_component = min(max(run_ratio, 0.0) / 2.0, 1.0)
    components.append((run_component, 0.5))

    if mean_excess_ratio is not None:
        magnitude_component = min(max(mean_excess_ratio, 0.0) / 0.5, 1.0)
        components.append((magnitude_component, 0.35))

    completeness_component = min(max(data_completeness, 0.0), 1.0)
    components.append((completeness_component, 0.15))

    total_weight = sum(w for _, w in components)
    score = sum(v * w for v, w in components) / total_weight if total_weight > 0 else 0.0
    return round(min(max(score, 0.0), 1.0), 3)


def verify_metrics_agent_output(
    agent_output: dict,
    raw_values: list[float],
    baseline_values: list[float] | None = None,
    min_consecutive: int = 3,
    raw_timestamps: list[str] | None = None,
    baseline_window_requested: int | None = None,
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

    `raw_timestamps` (optionnel, même longueur/ordre que raw_values): si
    fourni, le verifier recalcule un point de rupture (CUSUM offline, via
    `detect_changepoint` -- EXACTEMENT la même fonction que celle appelée
    côté tool MCP `evaluate_metric_window`, cf. metrics_mcp_server.py) et
    l'expose dans `changepoint_timestamp` pour RE-vérifier indépendamment
    l'approx_timestamp rapporté par l'agent -- même logique que pour
    `verify_sustained_threshold` vs `check_sustained_exceedance`: si les deux
    divergent significativement, ça mérite d'être signalé plutôt que
    silencieusement ignoré. Si absent, ce champ est simplement omis --
    n'affecte jamais `verified`/`final_is_anomaly` (l'agrément sur le
    timestamp exact n'est pas une condition de validation de l'anomalie
    elle-même, seulement une info de diagnostic supplémentaire).

    `baseline_window_requested` (optionnel, = le `window` passé à
    `get_baseline_stats` côté agent, 60 par défaut cf. metrics_mcp_server.py):
    si fourni, le verifier recalcule `data_completeness` de façon indépendante
    à partir de `len(baseline_values)` et le compare à
    `agent_output["evidence"]["data_completeness"]`. Un désaccord est signalé
    dans `data_completeness_disagreement` SANS jamais changer `verified` ni
    `final_is_anomaly` -- comme pour `changepoint_disagreement`, c'est un
    signal de diagnostic, pas une condition de validation. Un
    `low_confidence_warning` est ajouté séparément si la complétude recalculée
    est basse (< 0.5), que l'agent ait déclaré une anomalie ou non -- c'est
    pour ça que ce bloc est évalué avant le `return` anticipé du cas
    `is_anomaly: false` ci-dessous, plutôt qu'après.
    """
    data_completeness_diag: dict = {}
    if baseline_window_requested is not None:
        n_used = len(baseline_values) if baseline_values else 0
        recomputed = compute_data_completeness(n_used, baseline_window_requested)
        data_completeness_diag["data_completeness_recomputed"] = recomputed

        agent_completeness = agent_output.get("evidence", {}).get("data_completeness")
        if agent_completeness is not None and abs(agent_completeness - recomputed) > 0.05:
            data_completeness_diag["data_completeness_disagreement"] = (
                f"agent a rapporté evidence.data_completeness={agent_completeness}, verifier "
                f"recalcule {recomputed} à partir de {n_used}/{baseline_window_requested} points "
                f"de baseline observés -- à examiner, ne change pas automatiquement verified/"
                f"final_is_anomaly."
            )
        if recomputed < 0.5:
            data_completeness_diag["low_confidence_warning"] = (
                f"Verdict basé sur seulement {n_used}/{baseline_window_requested} points de "
                f"baseline ({recomputed * 100:.0f}%) -- fiabilité statistique réduite, à "
                f"pondérer en conséquence côté consommateur de ce résultat."
            )

    if not agent_output.get("is_anomaly"):
        return {
            "verified": True,
            "final_is_anomaly": False,
            "reason": "agent n'a pas déclaré d'anomalie",
            **data_completeness_diag,
        }

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
        return {
            "verified": False,
            "final_is_anomaly": False,
            "reason": f"pattern inconnu: {pattern}",
            **data_completeness_diag,
        }

    # run_ratio: pour fluctuation, il n'y a pas de "run" à proprement parler
    # (cf. verify_fluctuation) -- on fixe run_ratio=1.0 (ni pénalisé ni
    # bonifié par cette composante) et on laisse mean_excess_ratio porter
    # tout le signal d'ampleur pour ce pattern.
    if pattern == "fluctuation":
        run_ratio = 1.0
    else:
        run_ratio = round(result.longest_consecutive_run / min_consecutive, 3) if min_consecutive > 0 else 0.0

    recomputed_completeness = data_completeness_diag.get(
        "data_completeness_recomputed",
        agent_output.get("evidence", {}).get("data_completeness", 0.0) or 0.0,
    )

    confidence = compute_confidence(
        run_ratio=run_ratio,
        mean_excess_ratio=result.mean_excess_ratio,
        data_completeness=recomputed_completeness,
    )

    output = {
        "verified": result.passed,
        "final_is_anomaly": result.passed,
        "confidence": confidence,
        "reason": result.reason,
        "threshold_used": round(threshold_high if pattern in ("spike", "gradual_increase") else threshold_low, 3),
        **data_completeness_diag,
    }

    agent_confidence = agent_output.get("evidence", {}).get("confidence")
    if agent_confidence is not None and abs(agent_confidence - confidence) > 0.15:
        output["confidence_disagreement"] = (
            f"agent a rapporté evidence.confidence={agent_confidence} (via "
            f"exceedance.run_ratio/mean_excess_ratio d'evaluate_metric_window), verifier "
            f"recalcule {confidence} de façon indépendante -- à examiner, ne change pas "
            f"automatiquement verified/final_is_anomaly."
        )

    if raw_timestamps is not None and pattern in ("spike", "dip", "gradual_increase", "gradual_decrease"):
        cp = detect_changepoint(raw_values)
        agent_timestamp = agent_output.get("approx_timestamp")
        cp_timestamp = raw_timestamps[cp.index] if cp.detected and cp.index is not None else None

        output["changepoint_timestamp"] = cp_timestamp
        output["changepoint_reason"] = cp.reason
        if cp_timestamp is not None and agent_timestamp is not None and cp_timestamp != agent_timestamp:
            output["changepoint_disagreement"] = (
                f"agent a rapporté approx_timestamp={agent_timestamp}, mais le changepoint "
                f"CUSUM indépendant place le début de la rupture à {cp_timestamp} -- à examiner, "
                f"ne change pas automatiquement final_is_anomaly."
            )

    return output