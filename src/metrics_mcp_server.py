"""
Serveur MCP dédié au Metrics Agent, avec des tools métier nommés (au lieu du
SQL brut exposé par @executeautomation/database-server). Objectif: rendre
l'agent robuste au changement de dataset (CloudAnoBench, NAB, RS-Anomic...)
sans réécrire son prompt à chaque fois -- l'agent découvre le schéma
disponible via `describe_metrics_schema` plutôt que de le supposer.

Ce fichier remplace terme par terme la brique MCP_CONFIG de agent.py: au lieu
de lancer `npx @executeautomation/database-server`, on lance ce serveur en
stdio. Les tools additionnels (gpu_metrics, statistical_baseline_query, etc.)
s'ajoutent ici au fur et à mesure -- un @mcp.tool() par tool, pas de
consolidation en un seul "query anything".

Usage (test manuel):
    python metrics_mcp_server.py

Intégration dans agent.py -- remplacer MCP_CONFIG par:
    MCP_CONFIG = {
        "metrics-agent-tools": {
            "command": "python",
            "args": [os.path.join(os.path.dirname(__file__), "metrics_mcp_server.py")],
            "transport": "stdio",
        }
    }
"""
import os
import sqlite3
import statistics
import sys
import time

from mcp.server.fastmcp import FastMCP

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cloudwatch_metrics.db")

# IMPORTANT: en stdio, stdout est réservé au protocole MCP (JSON-RPC) --
# tout print() sur stdout casse la communication avec l'agent. Les logs de
# debug doivent TOUJOURS aller sur stderr. Activé via variable d'env pour ne
# pas polluer la sortie par défaut (mets METRICS_MCP_DEBUG=1 avant de lancer
# agent.py / eval.py / eval_cloudanobench.py pour voir chaque appel de tool).
DEBUG = os.environ.get("METRICS_MCP_DEBUG", "0") == "1"


def _log_tool_call(tool_name: str, kwargs: dict, result_preview: str, duration_ms: float):
    if not DEBUG:
        return
    print(
        f"[metrics_mcp_server] {tool_name}({kwargs}) -> {result_preview} "
        f"({duration_ms:.0f}ms)",
        file=sys.stderr, flush=True,
    )

mcp = FastMCP("metrics-agent-tools")


def _query(sql: str, params: tuple = ()) -> list[tuple]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()
    return rows


@mcp.tool()
def describe_metrics_schema(instance_id: str) -> dict:
    """
    Tool 1/N du Metrics Agent.

    Renvoie les métriques réellement disponibles pour une instance donnée,
    sans supposer un schéma fixe -- indispensable pour généraliser au-delà
    d'un seul dataset (CloudAnoBench a cpu_usage/mem_usage/disk_io/net_in/
    net_out, NAB a une seule métrique par série, RS-Anomic aura latence/
    throughput, etc.). L'agent doit appeler ce tool avant toute analyse.

    Args:
        instance_id: identifiant de l'instance/cas à inspecter.

    Returns:
        {
          "instance_id": ...,
          "available_metrics": ["cpu_usage", "mem_usage", ...],
          "n_points_per_metric": {"cpu_usage": 90, ...},
          "time_range": {"start": "...", "end": "..."}
        }
    """
    _t0 = time.perf_counter()
    rows = _query(
        "SELECT metric_name, COUNT(*), MIN(timestamp), MAX(timestamp) "
        "FROM cloudwatch_metrics WHERE instance_id = ? GROUP BY metric_name",
        (instance_id,),
    )

    if not rows:
        result = {
            "instance_id": instance_id,
            "available_metrics": [],
            "n_points_per_metric": {},
            "time_range": None,
            "warning": "Aucune métrique trouvée pour cet instance_id -- vérifie l'orthographe "
                       "ou utilise list_instances pour voir les instances disponibles.",
        }
        _log_tool_call("describe_metrics_schema", {"instance_id": instance_id},
                        f"AUCUNE métrique trouvée", (time.perf_counter() - _t0) * 1000)
        return result

    n_points = {m: n for m, n, _, _ in rows}
    global_start = min(r[2] for r in rows)
    global_end = max(r[3] for r in rows)

    result = {
        "instance_id": instance_id,
        "available_metrics": sorted(n_points.keys()),
        "n_points_per_metric": n_points,
        "time_range": {"start": global_start, "end": global_end},
    }
    _log_tool_call(
        "describe_metrics_schema", {"instance_id": instance_id},
        f"{len(n_points)} métriques: {sorted(n_points.keys())}",
        (time.perf_counter() - _t0) * 1000,
    )
    return result


@mcp.tool()
def get_metric_points(instance_id: str, metric_name: str, limit: int = 30,
                       end_timestamp: str | None = None) -> dict:
    """
    Tool 2/N du Metrics Agent -- remplace le SQL brut pour récupérer une
    série temporelle. Renvoie les `limit` derniers points d'une métrique
    pour une instance, en ordre chronologique, optionnellement bornés par
    un timestamp de fin (pour rejouer une fenêtre passée plutôt que les
    tout derniers points).

    Args:
        instance_id: identifiant de l'instance/cas (voir describe_metrics_schema).
        metric_name: nom exact de la métrique (voir describe_metrics_schema
            pour la liste des métriques disponibles -- ne pas deviner).
        limit: nombre de points à renvoyer (défaut 30).
        end_timestamp: si fourni, ne renvoie que les points <= ce timestamp.

    Returns:
        {
          "instance_id": ..., "metric_name": ...,
          "points": [{"timestamp": "...", "value": ...}, ...]  # ordre chronologique
        }
    """
    _t0 = time.perf_counter()
    if end_timestamp:
        rows = _query(
            "SELECT timestamp, value FROM cloudwatch_metrics "
            "WHERE instance_id = ? AND metric_name = ? AND timestamp <= ? "
            "ORDER BY timestamp DESC LIMIT ?",
            (instance_id, metric_name, end_timestamp, limit),
        )
    else:
        rows = _query(
            "SELECT timestamp, value FROM cloudwatch_metrics "
            "WHERE instance_id = ? AND metric_name = ? "
            "ORDER BY timestamp DESC LIMIT ?",
            (instance_id, metric_name, limit),
        )

    points = [{"timestamp": ts, "value": val} for ts, val in reversed(rows)]

    call_kwargs = {"instance_id": instance_id, "metric_name": metric_name,
                    "limit": limit, "end_timestamp": end_timestamp}

    if not points:
        result = {
            "instance_id": instance_id,
            "metric_name": metric_name,
            "points": [],
            "warning": "Aucun point trouvé -- vérifie le nom exact de la métrique via "
                       "describe_metrics_schema (ne devine jamais un nom de métrique).",
        }
        _log_tool_call("get_metric_points", call_kwargs, "AUCUN point trouvé",
                        (time.perf_counter() - _t0) * 1000)
        return result

    _log_tool_call(
        "get_metric_points", call_kwargs,
        f"{len(points)} points, {points[0]['timestamp']} -> {points[-1]['timestamp']}",
        (time.perf_counter() - _t0) * 1000,
    )
    return {"instance_id": instance_id, "metric_name": metric_name, "points": points}


@mcp.tool()
def get_baseline_stats(instance_id: str, metric_name: str, before_timestamp: str,
                        window: int = 60) -> dict:
    """
    Tool 3/N du Metrics Agent -- calcule une statistique de référence ("normale")
    sur une fenêtre ANTÉRIEURE à before_timestamp, pour donner à l'agent un seuil
    explicite calculé sur données réelles au lieu qu'il en invente un.

    Utilise médiane + MAD (Median Absolute Deviation) plutôt que
    moyenne/écart-type: plus robuste si la fenêtre de référence contient déjà
    quelques points bruités. Le facteur 1.4826 rend le MAD comparable à un
    écart-type sous hypothèse de normalité (constante de consistance standard),
    ce qui permet d'exprimer un seuil "3 sigma" de façon robuste.

    L'agent DOIT appeler ce tool avant de juger si une valeur observée est
    anormale, et reporter les valeurs renvoyées ici (pas des valeurs inventées)
    dans son champ evidence.threshold_used.

    Args:
        instance_id: identifiant de l'instance/cas.
        metric_name: nom exact de la métrique (voir describe_metrics_schema).
        before_timestamp: borne supérieure EXCLUSIVE de la fenêtre de référence --
            typiquement le timestamp de début de la fenêtre suspecte analysée,
            pour ne jamais laisser l'anomalie elle-même contaminer la baseline.
        window: nombre de points de référence à utiliser (défaut 60).

    Returns:
        {
          "instance_id": ..., "metric_name": ...,
          "median": ..., "mad": ...,
          "suggested_threshold_low": médiane - 3 * 1.4826 * MAD,
          "suggested_threshold_high": médiane + 3 * 1.4826 * MAD,
          "n_points_used": ...,
          "reference_range": {"start": "...", "end": "..."}
        }
        En cas d'historique insuffisant, renvoie un champ "warning" et pas de stats.
    """
    _t0 = time.perf_counter()
    call_kwargs = {"instance_id": instance_id, "metric_name": metric_name,
                    "before_timestamp": before_timestamp, "window": window}

    rows = _query(
        "SELECT timestamp, value FROM cloudwatch_metrics "
        "WHERE instance_id = ? AND metric_name = ? AND timestamp < ? "
        "ORDER BY timestamp DESC LIMIT ?",
        (instance_id, metric_name, before_timestamp, window),
    )

    # Un MAD nécessite un minimum de points pour être statistiquement
    # significatif -- sous ce seuil on préfère prévenir l'agent plutôt que
    # renvoyer des stats calculées sur une poignée de valeurs.
    MIN_POINTS = 10
    if len(rows) < MIN_POINTS:
        result = {
            "instance_id": instance_id,
            "metric_name": metric_name,
            "warning": (
                f"Historique insuffisant avant {before_timestamp} pour établir une "
                f"baseline fiable ({len(rows)} points trouvés, {MIN_POINTS} minimum). "
                f"Ne pas halluciner de seuil -- baser le jugement uniquement sur la "
                f"forme de la série observée (get_metric_points)."
            ),
            "n_points_used": len(rows),
        }
        _log_tool_call("get_baseline_stats", call_kwargs,
                        f"insuffisant ({len(rows)} pts)", (time.perf_counter() - _t0) * 1000)
        return result

    rows = rows[::-1]  # ordre chronologique
    values = [v for _, v in rows]

    median = statistics.median(values)
    mad = statistics.median([abs(v - median) for v in values])
    # Évite un MAD nul (série de référence parfaitement plate) qui rendrait
    # tout seuil trivialement franchi par le moindre bruit.
    scaled_mad = 1.4826 * mad if mad > 0 else 1e-6

    result = {
        "instance_id": instance_id,
        "metric_name": metric_name,
        "median": median,
        "mad": mad,
        "suggested_threshold_low": median - 3 * scaled_mad,
        "suggested_threshold_high": median + 3 * scaled_mad,
        "n_points_used": len(values),
        "reference_range": {"start": rows[0][0], "end": rows[-1][0]},
    }
    _log_tool_call(
        "get_baseline_stats", call_kwargs,
        f"median={median:.2f} mad={mad:.2f} range=[{result['suggested_threshold_low']:.2f}, "
        f"{result['suggested_threshold_high']:.2f}]",
        (time.perf_counter() - _t0) * 1000,
    )
    return result


@mcp.tool()
def check_sustained_exceedance(
    instance_id: str,
    metric_name: str,
    threshold: float,
    direction: str,
    end_timestamp: str | None = None,
    limit: int = 30,
    min_consecutive: int = 3,
) -> dict:
    """
    Tool 4/N du Metrics Agent -- vérifie si la fenêtre observée contient une
    séquence d'au moins `min_consecutive` points CONSÉCUTIFS qui dépassent
    `threshold` (direction='above') ou passent en-dessous (direction='below').

    À utiliser TOUJOURS après get_baseline_stats: passe `threshold` =
    suggested_threshold_high (pour direction='above', cas spike / gradual
    increase) ou suggested_threshold_low (pour direction='below', cas dip /
    gradual decrease).

    L'agent NE DOIT JAMAIS compter les points consécutifs lui-même en lisant
    le JSON de get_metric_points -- ce calcul est fait ici de façon
    déterministe et doit être la seule source de vérité pour le champ
    `evidence.points_above_threshold` de la sortie finale.

    Args:
        instance_id: identifiant de l'instance/cas.
        metric_name: nom exact de la métrique (voir describe_metrics_schema).
        threshold: seuil numérique à comparer -- typiquement
            suggested_threshold_high ou suggested_threshold_low renvoyé par
            get_baseline_stats. Ne jamais inventer cette valeur.
        direction: "above" pour détecter un dépassement par le haut (spike,
            gradual_increase), "below" pour un dépassement par le bas (dip,
            gradual_decrease).
        end_timestamp: si fourni, borne la fenêtre analysée (comme pour
            get_metric_points) -- doit être cohérent avec la fenêtre déjà
            récupérée à l'étape 1 du prompt.
        limit: nombre de points de la fenêtre à analyser (défaut 30, doit
            correspondre à la fenêtre déjà récupérée via get_metric_points).
        min_consecutive: nombre minimal de points consécutifs requis pour
            confirmer l'anomalie (défaut 3, conforme à la règle du prompt).

    Returns:
        {
          "instance_id": ..., "metric_name": ...,
          "direction": "above" | "below",
          "threshold": ...,
          "window_size": nombre de points analysés,
          "longest_consecutive_run": longueur du plus long run trouvé,
          "run_start_timestamp": timestamp de début de ce run (ou null),
          "run_end_timestamp": timestamp de fin de ce run (ou null),
          "passed": true si longest_consecutive_run >= min_consecutive,
          "min_consecutive_required": min_consecutive
        }
    """
    _t0 = time.perf_counter()

    if direction not in ("above", "below"):
        result = {
            "instance_id": instance_id,
            "metric_name": metric_name,
            "error": f"direction invalide: '{direction}' -- doit être 'above' ou 'below'.",
        }
        _log_tool_call("check_sustained_exceedance", {"direction": direction},
                        "ERREUR direction invalide", (time.perf_counter() - _t0) * 1000)
        return result

    if end_timestamp:
        rows = _query(
            "SELECT timestamp, value FROM cloudwatch_metrics "
            "WHERE instance_id = ? AND metric_name = ? AND timestamp <= ? "
            "ORDER BY timestamp DESC LIMIT ?",
            (instance_id, metric_name, end_timestamp, limit),
        )
    else:
        rows = _query(
            "SELECT timestamp, value FROM cloudwatch_metrics "
            "WHERE instance_id = ? AND metric_name = ? "
            "ORDER BY timestamp DESC LIMIT ?",
            (instance_id, metric_name, limit),
        )

    rows = rows[::-1]  # ordre chronologique, cohérent avec get_metric_points

    call_kwargs = {
        "instance_id": instance_id, "metric_name": metric_name,
        "threshold": threshold, "direction": direction,
        "end_timestamp": end_timestamp, "limit": limit,
        "min_consecutive": min_consecutive,
    }

    if not rows:
        result = {
            "instance_id": instance_id,
            "metric_name": metric_name,
            "direction": direction,
            "threshold": threshold,
            "window_size": 0,
            "longest_consecutive_run": 0,
            "run_start_timestamp": None,
            "run_end_timestamp": None,
            "passed": False,
            "min_consecutive_required": min_consecutive,
            "warning": "Aucun point trouvé pour cette fenêtre -- vérifie instance_id/metric_name.",
        }
        _log_tool_call("check_sustained_exceedance", call_kwargs,
                        "AUCUN point trouvé", (time.perf_counter() - _t0) * 1000)
        return result

    # Recherche du plus long run consécutif de points satisfaisant la condition.
    def _satisfies(v: float) -> bool:
        return v > threshold if direction == "above" else v < threshold

    best_len = 0
    best_start_idx = None
    cur_len = 0
    cur_start_idx = None

    for i, (_, val) in enumerate(rows):
        if _satisfies(val):
            if cur_len == 0:
                cur_start_idx = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start_idx = cur_start_idx
        else:
            cur_len = 0
            cur_start_idx = None

    if best_start_idx is not None:
        run_start_ts = rows[best_start_idx][0]
        run_end_ts = rows[best_start_idx + best_len - 1][0]
    else:
        run_start_ts = None
        run_end_ts = None

    passed = best_len >= min_consecutive

    result = {
        "instance_id": instance_id,
        "metric_name": metric_name,
        "direction": direction,
        "threshold": threshold,
        "window_size": len(rows),
        "longest_consecutive_run": best_len,
        "run_start_timestamp": run_start_ts,
        "run_end_timestamp": run_end_ts,
        "passed": passed,
        "min_consecutive_required": min_consecutive,
    }
    _log_tool_call(
        "check_sustained_exceedance", call_kwargs,
        f"longest_run={best_len} passed={passed} "
        f"(seuil requis={min_consecutive}, direction={direction})",
        (time.perf_counter() - _t0) * 1000,
    )
    return result


@mcp.tool()
def classify_trend(
    instance_id: str,
    metric_name: str,
    end_timestamp: str | None = None,
    limit: int = 30,
    slope_ratio_threshold: float = 0.15,
) -> dict:
    """
    Tool 5/N du Metrics Agent -- classe la fenêtre observée comme "stable" ou
    "trending" (up/down) AVANT d'interpréter les seuils de get_baseline_stats.

    Pourquoi ce tool existe: get_baseline_stats calcule une médiane + MAD, ce
    qui suppose implicitement que la série de référence oscille autour d'une
    valeur centrale stable. Pour les patterns "gradual_increase" /
    "gradual_decrease" (dérive lente et soutenue), cette hypothèse est fausse
    -- une comparaison à un seuil fixe médiane+3*MAD peut soit rater une
    dérive lente qui reste sous le seuil pendant longtemps, soit interpréter
    à tort une dérive normale (ex: montée de charge progressive prévue) comme
    un dépassement franc. classify_trend calcule une régression linéaire
    simple sur la fenêtre et renvoie une pente normalisée, permettant à
    l'agent de savoir AVANT d'interpréter les seuils si la série est stable
    (comparer aux seuils MAD a du sens) ou en tendance (chercher une pente
    soutenue plutôt qu'un dépassement de seuil ponctuel).

    À utiliser en complément de get_baseline_stats et check_sustained_exceedance,
    typiquement juste après get_metric_points (étape 1) et avant d'interpréter
    la baseline (étape 2).

    Args:
        instance_id: identifiant de l'instance/cas.
        metric_name: nom exact de la métrique (voir describe_metrics_schema).
        end_timestamp: si fourni, borne la fenêtre analysée (cohérent avec la
            fenêtre déjà récupérée via get_metric_points).
        limit: nombre de points de la fenêtre à analyser (défaut 30, doit
            correspondre à la fenêtre déjà récupérée via get_metric_points).
        slope_ratio_threshold: seuil au-delà duquel la pente normalisée est
            considérée comme une tendance plutôt que du bruit (défaut 0.15,
            soit 15% de variation relative sur la fenêtre). Une valeur plus
            basse rend la détection de tendance plus sensible.

    Returns:
        {
          "instance_id": ..., "metric_name": ...,
          "window_size": nombre de points analysés,
          "slope": pente brute (unité de la métrique par point),
          "normalized_slope": pente normalisée (slope * window_size / mean),
          "mean_value": moyenne de la fenêtre,
          "classification": "stable" | "trending_up" | "trending_down",
          "slope_ratio_threshold": valeur du seuil utilisé
        }
        Si la fenêtre a moins de 2 points, renvoie un warning et
        classification="stable" par défaut (pas assez de données pour juger
        une tendance -- rester conservateur plutôt que d'halluciner une pente).
    """
    _t0 = time.perf_counter()

    if end_timestamp:
        rows = _query(
            "SELECT timestamp, value FROM cloudwatch_metrics "
            "WHERE instance_id = ? AND metric_name = ? AND timestamp <= ? "
            "ORDER BY timestamp DESC LIMIT ?",
            (instance_id, metric_name, end_timestamp, limit),
        )
    else:
        rows = _query(
            "SELECT timestamp, value FROM cloudwatch_metrics "
            "WHERE instance_id = ? AND metric_name = ? "
            "ORDER BY timestamp DESC LIMIT ?",
            (instance_id, metric_name, limit),
        )

    rows = rows[::-1]  # ordre chronologique

    call_kwargs = {
        "instance_id": instance_id, "metric_name": metric_name,
        "end_timestamp": end_timestamp, "limit": limit,
        "slope_ratio_threshold": slope_ratio_threshold,
    }

    n = len(rows)
    if n < 2:
        result = {
            "instance_id": instance_id,
            "metric_name": metric_name,
            "window_size": n,
            "slope": None,
            "normalized_slope": None,
            "mean_value": None,
            "classification": "stable",
            "slope_ratio_threshold": slope_ratio_threshold,
            "warning": "Moins de 2 points disponibles -- impossible de calculer une "
                       "pente fiable, classification par défaut 'stable'.",
        }
        _log_tool_call("classify_trend", call_kwargs,
                        "insuffisant, defaut stable", (time.perf_counter() - _t0) * 1000)
        return result

    values = [v for _, v in rows]

    # Régression linéaire simple (moindres carrés) sur l'indice de position
    # (0..n-1) plutôt que sur le timestamp brut -- les points sont supposés
    # régulièrement espacés (cohérent avec le reste du pipeline, cf.
    # get_metric_points qui ne fait aucune interpolation).
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n

    numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    slope = numerator / denominator if denominator > 0 else 0.0

    # Pente normalisée: variation totale sur la fenêtre, rapportée à la
    # moyenne -- rend le seuil comparable indépendamment de l'échelle de la
    # métrique (CPU en % vs bytes réseau, par exemple).
    total_variation = slope * (n - 1)
    normalized_slope = total_variation / y_mean if y_mean != 0 else 0.0

    if abs(normalized_slope) < slope_ratio_threshold:
        classification = "stable"
    elif normalized_slope > 0:
        classification = "trending_up"
    else:
        classification = "trending_down"

    result = {
        "instance_id": instance_id,
        "metric_name": metric_name,
        "window_size": n,
        "slope": slope,
        "normalized_slope": normalized_slope,
        "mean_value": y_mean,
        "classification": classification,
        "slope_ratio_threshold": slope_ratio_threshold,
    }
    _log_tool_call(
        "classify_trend", call_kwargs,
        f"classification={classification} normalized_slope={normalized_slope:.3f}",
        (time.perf_counter() - _t0) * 1000,
    )
    return result


@mcp.tool()
def list_instances(limit: int = 50) -> list[str]:
    """
    Liste les instance_id disponibles dans la base courante (utile pour
    l'agent s'il doit explorer avant de recevoir une instance précise en
    consigne, ou pour du debug manuel).
    """
    rows = _query(
        "SELECT DISTINCT instance_id FROM cloudwatch_metrics ORDER BY instance_id LIMIT ?",
        (limit,),
    )
    instances = [r[0] for r in rows]
    _log_tool_call("list_instances", {"limit": limit}, f"{len(instances)} instances", 0)
    return instances


if __name__ == "__main__":
    mcp.run(transport="stdio")