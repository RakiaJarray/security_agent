"""
Serveur MCP dédié au Metrics Agent, avec des tools métier nommés (au lieu du
SQL brut exposé par @executeautomation/database-server). Objectif: rendre
l'agent robuste au changement de dataset (CloudAnoBench, NAB, RS-Anomic...)
sans réécrire son prompt à chaque fois -- l'agent découvre le schéma
disponible via `describe_metrics_schema` plutôt que de le supposer.

Ce fichier remplace terme par terme la brique MCP_CONFIG de agent.py: au lieu
de lancer `npx @executeautomation/database-server`, on lance ce serveur en
stdio. Les tools additionnels s'ajoutent ici au fur et à mesure -- un
@mcp.tool() par tool, pas de consolidation en un seul "query anything".

Tools exposés (5): describe_metrics_schema, get_metric_window,
get_baseline_stats, evaluate_metric_window, list_instances.

--- Historique de fusion (v2) ---
v1 avait 7 tools. Deux paires ont été fusionnées pour réduire l'ambiguïté de
sélection de tool par le LLM (au-delà d'une dizaine de tools, et surtout
quand deux tools se chevauchent sémantiquement, la sélection se dégrade et
le prompt doit compenser avec des paragraphes d'arbitrage -- signe qu'il
fallait fusionner plutôt que documenter la distinction) :

1. get_metric_points + get_correlated_metrics -> get_metric_window
   get_correlated_metrics était un sur-ensemble strict de get_metric_points
   (même requête, boucle sur 1 ou N métriques). Un seul tool qui accepte
   `metric_names` en str OU liste supprime le choix ambigu.

2. classify_trend + check_sustained_exceedance -> evaluate_metric_window
   Les deux interrogeaient exactement la même fenêtre (même instance_id/
   metric_name/end_timestamp/limit) pour calculer deux statistiques
   différentes. Fusionnés en un seul tool à "deux phases" via le paramètre
   optionnel `threshold` (fourni seulement après get_baseline_stats) --
   garantit aussi que trend et exceedance portent sur EXACTEMENT la même
   fenêtre (un seul fetch SQL).

get_baseline_stats N'A PAS été fusionné avec les deux ci-dessus bien qu'il
s'agisse aussi d'une requête sur fenêtre temporelle: il interroge une
fenêtre ANTÉRIEURE (timestamp < before_timestamp) alors que les deux autres
interrogent la fenêtre COURANTE (timestamp <= end_timestamp). Fusionner
aurait cassé la garantie "le seuil est calculé sur la baseline, jamais sur
la fenêtre suspecte elle-même" (cf. verifier.py) -- une confusion de fenêtre
ici serait bien plus grave qu'une redondance de tool.

Avant tout nouvel ajout de tool, vérifier qu'il ne chevauche pas
sémantiquement un existant.

Usage (test manuel):
    python metrics_mcp_server.py

Intégration dans agent.py -- MCP_CONFIG inchangé (même chemin de fichier):
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


def _fetch_window(instance_id: str, metric_name: str, end_timestamp: str | None,
                   limit: int) -> list[tuple[str, float]]:
    """
    Helper interne partagé -- récupère les `limit` derniers points d'UNE
    métrique, en ordre chronologique, optionnellement bornés par
    end_timestamp. Utilisé par get_metric_window (par métrique) et par
    evaluate_metric_window (fetch unique partagé entre trend et exceedance).
    """
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
    return rows[::-1]  # ordre chronologique


@mcp.tool()
def describe_metrics_schema(instance_id: str) -> dict:
    """
    Tool 1/5 du Metrics Agent -- OBLIGATOIRE en premier.

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
                        "AUCUNE métrique trouvée", (time.perf_counter() - _t0) * 1000)
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
def get_metric_window(
    instance_id: str,
    metric_names: list[str] | str,
    limit: int = 30,
    end_timestamp: str | None = None,
) -> dict:
    """
    Tool 2/5 du Metrics Agent -- récupère une OU plusieurs métriques,
    alignées sur la même fenêtre temporelle, en un seul appel.

    Remplace les anciens get_metric_points / get_correlated_metrics: passe
    une seule métrique (str) si tu dois isoler une série pour
    evaluate_metric_window, ou une liste de métriques si describe_metrics_schema
    a montré plus d'une métrique disponible -- certains patterns (crypto-mining,
    DDoS, exfiltration de données) ne se voient que si plusieurs métriques
    bougent EN MÊME TEMPS, ce qui est invisible en lisant les métriques une
    par une. N'appelle JAMAIS ce tool en boucle une fois par métrique --
    passe directement la liste complète en un seul appel.

    Args:
        instance_id: identifiant de l'instance/cas (voir describe_metrics_schema).
        metric_names: nom exact d'UNE métrique (str), ou liste de noms exacts
            de plusieurs métriques (voir describe_metrics_schema pour la
            liste des métriques disponibles -- ne jamais deviner un nom).
        limit: nombre de points par métrique à renvoyer (défaut 30).
        end_timestamp: si fourni, ne renvoie que les points <= ce timestamp.

    Returns:
        {
          "instance_id": ...,
          "metrics": {
            "cpu_usage": [{"timestamp": "...", "value": ...}, ...],  # ordre chronologique
            "net_in": [{"timestamp": "...", "value": ...}, ...],
            ...
          },
          "aligned_timestamps": true/false  -- false si les métriques n'ont
              pas exactement les mêmes timestamps (peut arriver si une
              métrique a des trous) -- dans ce cas, comparer les métriques
              par position relative dans la fenêtre plutôt que par égalité
              stricte de timestamp.
        }
        Si une métrique demandée n'existe pas pour cette instance, elle est
        simplement absente de "metrics" (pas d'erreur bloquante) -- vérifie
        les clés présentes avant de raisonner dessus.
    """
    _t0 = time.perf_counter()

    if isinstance(metric_names, str):
        metric_names = [metric_names]

    call_kwargs = {"instance_id": instance_id, "metric_names": metric_names,
                    "limit": limit, "end_timestamp": end_timestamp}

    if not metric_names:
        result = {
            "instance_id": instance_id,
            "metrics": {},
            "aligned_timestamps": True,
            "warning": "Aucune métrique demandée -- utilise describe_metrics_schema "
                       "d'abord pour connaître les métriques disponibles.",
        }
        _log_tool_call("get_metric_window", call_kwargs,
                        "aucune métrique demandée", (time.perf_counter() - _t0) * 1000)
        return result

    metrics_out: dict[str, list[dict]] = {}
    timestamp_sets = []

    for metric_name in metric_names:
        rows = _fetch_window(instance_id, metric_name, end_timestamp, limit)
        if not rows:
            continue  # métrique absente pour cette instance -- on l'omet simplement
        points = [{"timestamp": ts, "value": val} for ts, val in rows]
        metrics_out[metric_name] = points
        timestamp_sets.append(tuple(p["timestamp"] for p in points))

    if not metrics_out:
        result = {
            "instance_id": instance_id,
            "metrics": {},
            "aligned_timestamps": True,
            "warning": "Aucun point trouvé pour les métriques demandées -- vérifie les "
                       "noms exacts via describe_metrics_schema (ne devine jamais un nom).",
        }
        _log_tool_call("get_metric_window", call_kwargs,
                        "AUCUN point trouvé", (time.perf_counter() - _t0) * 1000)
        return result

    aligned = len(set(timestamp_sets)) <= 1 if timestamp_sets else True

    result = {
        "instance_id": instance_id,
        "metrics": metrics_out,
        "aligned_timestamps": aligned,
    }
    _log_tool_call(
        "get_metric_window", call_kwargs,
        f"{len(metrics_out)} métrique(s) récupérée(s): {sorted(metrics_out.keys())}, "
        f"aligned={aligned}",
        (time.perf_counter() - _t0) * 1000,
    )
    return result


@mcp.tool()
def get_baseline_stats(instance_id: str, metric_name: str, before_timestamp: str,
                        window: int = 60) -> dict:
    """
    Tool 3/5 du Metrics Agent -- calcule une statistique de référence ("normale")
    sur une fenêtre ANTÉRIEURE à before_timestamp, pour donner à l'agent un seuil
    explicite calculé sur données réelles au lieu qu'il en invente un.

    Utilise médiane + MAD (Median Absolute Deviation) plutôt que
    moyenne/écart-type: plus robuste si la fenêtre de référence contient déjà
    quelques points bruités. Le facteur 1.4826 rend le MAD comparable à un
    écart-type sous hypothèse de normalité (constante de consistance standard),
    ce qui permet d'exprimer un seuil "3 sigma" de façon robuste.

    NE PAS fusionner ce tool avec evaluate_metric_window: il interroge une
    fenêtre ANTÉRIEURE (timestamp < before_timestamp), jamais la fenêtre
    suspecte elle-même -- sinon un spike prolongé finirait par devenir sa
    propre "normalité" statistique et ne serait plus détectable.

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
                f"forme de la série observée (get_metric_window)."
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
def evaluate_metric_window(
    instance_id: str,
    metric_name: str,
    end_timestamp: str | None = None,
    limit: int = 30,
    slope_ratio_threshold: float = 0.15,
    threshold: float | None = None,
    direction: str | None = None,
    min_consecutive: int = 3,
) -> dict:
    """
    Tool 4/5 du Metrics Agent -- analyse la fenêtre COURANTE d'une métrique en
    UN SEUL appel. Fusionne ce qui était avant deux tools séparés (classify_trend
    et check_sustained_exceedance), qui interrogeaient la même fenêtre deux fois
    -- un seul fetch ici garantit que trend et exceedance portent sur EXACTEMENT
    les mêmes points.

    Ce tool s'utilise en DEUX PHASES, dans cet ordre :

    PHASE 1 (obligatoire, avant get_baseline_stats) -- appelle SANS `threshold` :
        evaluate_metric_window(instance_id, metric_name, end_timestamp, limit)
    Renvoie uniquement la classification de tendance ("stable" / "trending_up"
    / "trending_down") calculée par régression linéaire simple sur la fenêtre.
    NE JUGE JAMAIS toi-même si une série "monte lentement" en lisant les valeurs
    brutes -- utilise exclusivement le champ `classification` renvoyé ici.
    Si "stable": une comparaison à médiane+3*MAD (get_baseline_stats puis
    PHASE 2) a du sens. Si "trending_up"/"trending_down": privilégie le
    pattern gradual_increase/gradual_decrease et utilise `normalized_slope`
    comme preuve quantitative -- un seuil fixe peut être trompeur sur une
    dérive lente.

    PHASE 2 (obligatoire si un seuil a été obtenu via get_baseline_stats) --
    rappelle ce même tool en fournissant `threshold` (= suggested_threshold_high
    ou suggested_threshold_low renvoyé par get_baseline_stats) ET `direction`
    ("above" pour un dépassement par le haut, "below" par le bas) :
        evaluate_metric_window(instance_id, metric_name, end_timestamp, limit,
                                threshold=..., direction=...)
    Renvoie EN PLUS de la tendance le résultat de dépassement soutenu: `passed`
    (bool) et `longest_consecutive_run` (nombre de points CONSÉCUTIFS au-delà
    du seuil). NE COMPTE JAMAIS toi-même les points consécutifs en relisant le
    JSON de get_metric_window -- ce comptage est fait ici de façon déterministe
    et constitue ta SEULE source de vérité pour le champ
    evidence.points_above_threshold de ta sortie finale.

    Args:
        instance_id: identifiant de l'instance/cas.
        metric_name: nom exact de la métrique (voir describe_metrics_schema).
        end_timestamp: si fourni, borne la fenêtre analysée (doit être cohérent
            entre PHASE 1 et PHASE 2 -- même fenêtre).
        limit: nombre de points de la fenêtre à analyser (défaut 30, doit
            rester identique entre PHASE 1 et PHASE 2).
        slope_ratio_threshold: seuil de pente normalisée au-delà duquel la
            fenêtre est classée "trending" plutôt que "stable" (défaut 0.15,
            soit 15% de variation relative sur la fenêtre).
        threshold: seuil numérique de dépassement (PHASE 2 uniquement) --
            typiquement suggested_threshold_high/low de get_baseline_stats.
            Laisser à None en PHASE 1. Ne jamais inventer cette valeur.
        direction: "above" ou "below" (requis si `threshold` est fourni).
        min_consecutive: nombre minimal de points consécutifs requis pour
            confirmer l'anomalie (défaut 3, conforme à la règle du prompt).

    Returns (PHASE 1, threshold=None):
        {
          "instance_id": ..., "metric_name": ..., "window_size": ...,
          "slope": ..., "normalized_slope": ..., "mean_value": ...,
          "classification": "stable" | "trending_up" | "trending_down",
          "slope_ratio_threshold": ...,
          "exceedance": null
        }

    Returns (PHASE 2, threshold fourni) -- mêmes champs que ci-dessus, PLUS :
        {
          ...,
          "exceedance": {
            "threshold": ..., "direction": ...,
            "longest_consecutive_run": ...,
            "run_start_timestamp": ..., "run_end_timestamp": ...,
            "passed": true/false,
            "min_consecutive_required": ...
          }
        }
    """
    _t0 = time.perf_counter()

    call_kwargs = {
        "instance_id": instance_id, "metric_name": metric_name,
        "end_timestamp": end_timestamp, "limit": limit,
        "slope_ratio_threshold": slope_ratio_threshold,
        "threshold": threshold, "direction": direction,
        "min_consecutive": min_consecutive,
    }

    if threshold is not None and direction not in ("above", "below"):
        result = {
            "instance_id": instance_id,
            "metric_name": metric_name,
            "error": f"direction invalide: '{direction}' -- doit être 'above' ou 'below' "
                     f"quand `threshold` est fourni (PHASE 2).",
        }
        _log_tool_call("evaluate_metric_window", call_kwargs,
                        "ERREUR direction invalide", (time.perf_counter() - _t0) * 1000)
        return result

    rows = _fetch_window(instance_id, metric_name, end_timestamp, limit)
    n = len(rows)

    if n == 0:
        result = {
            "instance_id": instance_id,
            "metric_name": metric_name,
            "window_size": 0,
            "slope": None,
            "normalized_slope": None,
            "mean_value": None,
            "classification": "stable",
            "slope_ratio_threshold": slope_ratio_threshold,
            "exceedance": None,
            "warning": "Aucun point trouvé pour cette fenêtre -- vérifie instance_id/metric_name "
                       "via describe_metrics_schema.",
        }
        _log_tool_call("evaluate_metric_window", call_kwargs,
                        "AUCUN point trouvé", (time.perf_counter() - _t0) * 1000)
        return result

    # --- Partie 1: classification de tendance (toujours calculée) ---
    if n < 2:
        classification = "stable"
        slope = None
        normalized_slope = None
        y_mean = rows[0][1] if rows else None
        trend_warning = ("Moins de 2 points disponibles -- impossible de calculer une "
                          "pente fiable, classification par défaut 'stable'.")
    else:
        values = [v for _, v in rows]
        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        slope = numerator / denominator if denominator > 0 else 0.0

        total_variation = slope * (n - 1)
        normalized_slope = total_variation / y_mean if y_mean != 0 else 0.0

        if abs(normalized_slope) < slope_ratio_threshold:
            classification = "stable"
        elif normalized_slope > 0:
            classification = "trending_up"
        else:
            classification = "trending_down"
        trend_warning = None

    result = {
        "instance_id": instance_id,
        "metric_name": metric_name,
        "window_size": n,
        "slope": slope,
        "normalized_slope": normalized_slope,
        "mean_value": y_mean,
        "classification": classification,
        "slope_ratio_threshold": slope_ratio_threshold,
        "exceedance": None,
    }
    if trend_warning:
        result["warning"] = trend_warning

    # --- Partie 2: dépassement soutenu (seulement si threshold fourni -- PHASE 2) ---
    if threshold is not None:
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

        result["exceedance"] = {
            "threshold": threshold,
            "direction": direction,
            "longest_consecutive_run": best_len,
            "run_start_timestamp": run_start_ts,
            "run_end_timestamp": run_end_ts,
            "passed": passed,
            "min_consecutive_required": min_consecutive,
        }

    _log_tool_call(
        "evaluate_metric_window", call_kwargs,
        f"classification={classification}"
        + (f" passed={result['exceedance']['passed']} "
           f"run={result['exceedance']['longest_consecutive_run']}"
           if result["exceedance"] else " (phase 1, pas de seuil)"),
        (time.perf_counter() - _t0) * 1000,
    )
    return result


@mcp.tool()
def list_instances(limit: int = 50) -> list[str]:
    """
    Tool 5/5 -- liste les instance_id disponibles dans la base courante
    (utile pour l'agent s'il doit explorer avant de recevoir une instance
    précise en consigne, ou pour du debug manuel).
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