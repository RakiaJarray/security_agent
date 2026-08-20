"""
Détection du point de rupture (changepoint) dans une fenêtre de métrique --
utilisé pour remplacer l'estimation actuelle de `approx_timestamp` (premier
point qui dépasse le seuil, ou lecture à l'oeil du LLM) par un point
mathématiquement justifié.

Méthode: CUSUM OFFLINE (rétroactif), pas la version séquentielle "en ligne".
--------------------------------------------------------------------------
La version séquentielle classique de CUSUM (Page, 1954) est conçue pour du
streaming: elle accumule les écarts au fil de l'eau et déclenche une alarme
dès qu'un seuil est franchi, sans jamais revenir en arrière. Ici ce n'est PAS
notre cas d'usage: `evaluate_metric_window` reçoit une fenêtre déjà complète
(30 points passés), donc on peut se permettre de regarder toute la fenêtre
d'un coup et de chercher directement LE point qui maximise la séparation
entre "avant" et "après" -- c'est la version offline/rétroactive de CUSUM.

Principe (cf. Kats CUSUM detector, Meta -- même idée, plus simple ici car
on ne fait pas le test de vraisemblance complet, juste la localisation):
1. Centrer la série sur sa moyenne globale.
2. Calculer la somme cumulée des écarts centrés: S_0=0, S_i = S_{i-1} + (x_i - mean).
3. Le point de rupture le plus probable est l'indice qui maximise
   |S_i - médiane(S)| (équivalent à argmax/argmin de la somme cumulée) --
   intuition: si la série est stable puis bascule à un niveau plus haut (ou
   plus bas) à partir de l'indice k, la somme cumulée croît (ou décroît) de
   façon quasi-monotone à partir de k, donc son extremum se situe précisément
   à la fin de la dérive, ce qui donne le DÉBUT du nouveau régime en
   remontant d'une fenêtre de confirmation (cf. `_refine_onset` ci-dessous).

Pourquoi pas juste "argmax(|S_i|)" brut ? Sur une série bruitée mais stable,
la somme cumulée fait une marche aléatoire qui peut avoir un extremum loin
de tout vrai changement, juste par hasard. On ajoute donc une confirmation:
le changepoint n'est retenu que si la différence de moyenne avant/après ce
point dépasse un multiple de l'écart-type global (cf. `min_effect_size_std`)
-- sinon on retombe sur "pas de rupture significative détectée" plutôt que
de reporter un point de rupture qui n'est que du bruit.
"""
from dataclasses import dataclass


@dataclass
class ChangepointResult:
    detected: bool
    index: int | None          # indice dans `values` (0-based) du point de rupture estimé
    mean_before: float | None
    mean_after: float | None
    effect_size_std: float | None   # |mean_after - mean_before| / std_global, mesure de la "force" de la rupture
    reason: str


def _cumulative_sum(values: list[float]) -> list[float]:
    mean = sum(values) / len(values)
    cusum = [0.0]
    for v in values:
        cusum.append(cusum[-1] + (v - mean))
    return cusum[1:]  # on droppe le S_0=0 initial, longueur = len(values)


def detect_changepoint(
    values: list[float],
    min_effect_size_std: float = 1.0,
    min_segment_len: int = 3,
) -> ChangepointResult:
    """
    Localise le point de rupture le plus probable dans `values` via CUSUM
    offline, avec un garde-fou sur la significativité de l'effet.

    `min_effect_size_std`: la différence de moyenne avant/après le point
    candidat doit dépasser ce multiple de l'écart-type global pour être
    retenue -- évite de reporter un "changepoint" qui n'est qu'une
    fluctuation aléatoire de la marche cumulée. 1.0 est un seuil raisonnable
    par défaut (effet d'au moins 1 écart-type), à ajuster si trop/pas assez
    sensible en pratique sur les données réelles.

    `min_segment_len`: nombre minimum de points de part et d'autre du point
    candidat pour que les moyennes avant/après soient estimables de façon
    fiable -- élimine les candidats trop proches des bords de la fenêtre
    (ex: index 0 ou len-1, où "avant" ou "après" n'a presque pas de points).
    """
    n = len(values)
    if n < 2 * min_segment_len:
        return ChangepointResult(
            False, None, None, None, None,
            f"fenêtre trop courte ({n} points, minimum {2 * min_segment_len} requis)",
        )

    mean_global = sum(values) / n
    variance = sum((v - mean_global) ** 2 for v in values) / n
    std_global = variance ** 0.5

    if std_global == 0:
        return ChangepointResult(False, None, None, None, None, "série constante, aucune rupture possible")

    cusum = _cumulative_sum(values)

    # On ne cherche le candidat que dans la zone où avant ET après ont assez
    # de points (cf. min_segment_len) -- pas d'argmax sur toute la fenêtre.
    search_range = range(min_segment_len - 1, n - min_segment_len)
    if not search_range:
        return ChangepointResult(False, None, None, None, None, "fenêtre trop courte pour la recherche")

    best_idx = max(search_range, key=lambda i: abs(cusum[i]))

    # Le changepoint estimé est le point APRÈS lequel le nouveau régime
    # commence (convention: best_idx est le dernier point de l'ancien
    # régime, donc l'anomalie démarre à best_idx + 1).
    before = values[: best_idx + 1]
    after = values[best_idx + 1:]
    mean_before = sum(before) / len(before)
    mean_after = sum(after) / len(after)
    effect_size = abs(mean_after - mean_before) / std_global

    if effect_size < min_effect_size_std:
        return ChangepointResult(
            False, None, round(mean_before, 3), round(mean_after, 3), round(effect_size, 3),
            f"rupture la plus probable trouvée à l'indice {best_idx + 1}, mais effet trop faible "
            f"({effect_size:.2f} écarts-types < seuil {min_effect_size_std}) -- probablement du bruit",
        )

    return ChangepointResult(
        True, best_idx + 1, round(mean_before, 3), round(mean_after, 3), round(effect_size, 3),
        f"rupture détectée à l'indice {best_idx + 1}/{n} "
        f"(moyenne avant={mean_before:.2f}, après={mean_after:.2f}, effet={effect_size:.2f}σ)",
    )


def changepoint_timestamp(
    timestamps: list[str],
    values: list[float],
    min_effect_size_std: float = 1.0,
    min_segment_len: int = 3,
) -> tuple[str | None, ChangepointResult]:
    """
    Point d'entrée pratique: prend `timestamps` alignés avec `values`
    (même longueur, même ordre chronologique -- typiquement la sortie de
    `get_metric_window`) et renvoie directement le timestamp du point de
    rupture, prêt à remplacer `approx_timestamp` dans la sortie de l'agent.

    Renvoie (None, result) si aucune rupture significative n'a été trouvée
    -- dans ce cas, garder le fallback actuel (premier point au-dessus du
    seuil) plutôt que de laisser `approx_timestamp` à null.
    """
    if len(timestamps) != len(values):
        raise ValueError(
            f"timestamps ({len(timestamps)}) et values ({len(values)}) doivent avoir la même longueur"
        )
    result = detect_changepoint(values, min_effect_size_std, min_segment_len)
    if not result.detected or result.index is None:
        return None, result
    return timestamps[result.index], result
