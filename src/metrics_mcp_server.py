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

Tools exposés (6): describe_metrics_schema, get_metric_window,
get_baseline_stats, evaluate_metric_window, check_multivariate_correlation,
list_instances.

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
from datetime import datetime, timedelta

from mcp.server.fastmcp import FastMCP

sys.path.insert(0, os.path.dirname(__file__))
from changepoint import detect_changepoint

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


def _fetch_seasonal_points(
    instance_id: str, metric_name: str, before_timestamp: str,
    lookback_days: int, tolerance_minutes: int, max_points: int,
) -> list[tuple[str, float]]:
    """
    Récupère les points situés à la même heure (± tolerance_minutes) que
    before_timestamp, sur chacun des `lookback_days` jours précédents --
    absorbe les patterns récurrents (backup nocturne, batch cron, pic de
    trafic du lundi matin) qui sont NORMAUX en récurrence mais ressemblent
    à une anomalie vus par une fenêtre "juste avant" seule (cf.
    `get_baseline_stats`, mode="seasonal").

    Si plus de `max_points` sont trouvés, on garde les points des jours les
    PLUS RÉCENTS (les plus proches de before_timestamp) -- pour qu'une dérive
    lente de la baseline sur plusieurs semaines (ex: montée progressive de
    charge) reste visible plutôt que diluée dans un échantillon arbitraire.
    """
    center = datetime.fromisoformat(before_timestamp)
    collected: list[tuple[str, float]] = []
    for d in range(1, lookback_days + 1):
        target = center - timedelta(days=d)
        lo = (target - timedelta(minutes=tolerance_minutes)).strftime("%Y-%m-%d %H:%M:%S")
        hi = (target + timedelta(minutes=tolerance_minutes)).strftime("%Y-%m-%d %H:%M:%S")
        rows = _query(
            "SELECT timestamp, value FROM cloudwatch_metrics "
            "WHERE instance_id = ? AND metric_name = ? AND timestamp BETWEEN ? AND ? "
            "ORDER BY timestamp",
            (instance_id, metric_name, lo, hi),
        )
        collected.extend(rows)

    collected.sort(key=lambda r: r[0])
    if len(collected) > max_points:
        collected = collected[-max_points:]
    return collected


@mcp.tool()
def get_baseline_stats(
    instance_id: str,
    metric_name: str,
    before_timestamp: str,
    window: int = 60,
    mode: str = "auto",
    lookback_days: int = 14,
    tolerance_minutes: int = 30,
) -> dict:
    """
    Tool 3/5 du Metrics Agent -- calcule une statistique de référence ("normale")
    pour donner à l'agent un seuil explicite calculé sur données réelles au lieu
    qu'il en invente un.

    Utilise médiane + MAD (Median Absolute Deviation) plutôt que
    moyenne/écart-type: plus robuste si la fenêtre de référence contient déjà
    quelques points bruités. Le facteur 1.4826 rend le MAD comparable à un
    écart-type sous hypothèse de normalité (constante de consistance standard),
    ce qui permet d'exprimer un seuil "3 sigma" de façon robuste.

    NE PAS fusionner ce tool avec evaluate_metric_window: la fenêtre de
    référence est TOUJOURS antérieure ou disjointe de before_timestamp, jamais
    la fenêtre suspecte elle-même -- sinon un spike prolongé finirait par
    devenir sa propre "normalité" statistique et ne serait plus détectable.

    ## Choix du mode (saisonnalité)
    Trois modes disponibles via `mode` :

    - "recent" (comportement historique) : les `window` derniers points
      STRICTEMENT AVANT before_timestamp (timestamp < before_timestamp).
      Simple, mais confond un pattern récurrent NORMAL (backup nocturne à 3h,
      batch cron, pic de trafic du lundi matin) avec une anomalie, puisqu'il
      compare uniquement à "ce qui précède immédiatement".

    - "seasonal" : compare à la MÊME HEURE (± tolerance_minutes) sur les
      `lookback_days` jours précédents, plutôt qu'à la période juste avant.
      Réduit les fausses alertes sur les tâches planifiées récurrentes.
      Si moins de 10 points saisonniers trouvés, renvoie un `warning` explicite
      (PAS de repli automatique -- utilise "auto" si tu veux un repli).

    - "auto" (RECOMMANDÉ, défaut) : tente "seasonal" d'abord ; si l'historique
      est insuffisant (< 10 points saisonniers, ex: série trop courte pour
      couvrir `lookback_days` jours), retombe silencieusement sur "recent" ET
      le signale explicitement via `mode_used` + `fallback_reason` dans la
      sortie. TOUJOURS lire `mode_used` avant de citer un seuil dans
      evidence.threshold_used -- ne suppose jamais quel mode a été utilisé.

    L'agent DOIT appeler ce tool avant de juger si une valeur observée est
    anormale, et reporter les valeurs renvoyées ici (pas des valeurs inventées)
    dans son champ evidence.threshold_used.

    Args:
        instance_id: identifiant de l'instance/cas.
        metric_name: nom exact de la métrique (voir describe_metrics_schema).
        before_timestamp: point de référence temporel -- typiquement le
            timestamp de début de la fenêtre suspecte analysée, pour ne
            jamais laisser l'anomalie elle-même contaminer la baseline.
        window: nombre MAXIMUM de points de référence à utiliser (défaut 60).
        mode: "recent" | "seasonal" | "auto" (défaut "auto", voir ci-dessus).
        lookback_days: nombre de jours précédents scrutés en mode "seasonal"/
            "auto" (défaut 14). Sans effet en mode "recent".
        tolerance_minutes: demi-largeur de la fenêtre horaire de comparaison
            en mode "seasonal"/"auto" (défaut 30, soit ± 30 min autour de la
            même heure chaque jour scruté). Sans effet en mode "recent".

    Returns:
        {
          "instance_id": ..., "metric_name": ..., "mode_used": "recent"|"seasonal",
          "median": ..., "mad": ...,
          "suggested_threshold_low": médiane - 3 * 1.4826 * MAD,
          "suggested_threshold_high": médiane + 3 * 1.4826 * MAD,
          "n_points_used": ...,
          "n_points_requested": ..., (= window)
          "data_completeness": n_points_used / n_points_requested, borné à [0, 1],
          "reference_range": {"start": "...", "end": "..."},
          "fallback_reason": "..." (présent seulement si mode="auto" a basculé
              vers "recent" faute d'historique saisonnier suffisant),
          "seasonal_params": {"lookback_days": ..., "tolerance_minutes": ...}
              (présent seulement si mode_used="seasonal" -- note aussi que
              reference_range est alors NON-CONTIGU, points de plusieurs
              jours distincts, pas une plage continue)
        }
        En cas d'historique insuffisant, renvoie un champ "warning" et pas de stats
        (mais "n_points_used"/"n_points_requested"/"data_completeness" sont quand
        même renvoyés -- même en l'absence de seuil numérique, savoir sur combien
        de points repose l'absence de baseline reste une information utile pour
        l'agent et pour tout code qui consomme cette sortie en aval).
    """
    _t0 = time.perf_counter()
    call_kwargs = {"instance_id": instance_id, "metric_name": metric_name,
                    "before_timestamp": before_timestamp, "window": window,
                    "mode": mode, "lookback_days": lookback_days,
                    "tolerance_minutes": tolerance_minutes}

    if mode not in ("recent", "seasonal", "auto"):
        result = {
            "instance_id": instance_id,
            "metric_name": metric_name,
            "error": f"mode invalide: '{mode}' -- doit être 'recent', 'seasonal' ou 'auto'.",
        }
        _log_tool_call("get_baseline_stats", call_kwargs,
                        "ERREUR mode invalide", (time.perf_counter() - _t0) * 1000)
        return result

    def _data_completeness(n_points_used: int, n_points_requested: int) -> float:
        """
        Fraction (bornée à 1.0) de la fenêtre de référence demandée qui a
        effectivement pu être remplie. Sert de signal de confiance CONTINU,
        séparé du verdict lui-même -- un `data_completeness` bas ne doit
        jamais faire basculer un verdict tout seul, il doit juste être
        reporté pour que qui lit la sortie sache si le seuil (ou son
        absence) repose sur beaucoup ou peu de données.
        """
        if n_points_requested <= 0:
            return 0.0
        return round(min(n_points_used / n_points_requested, 1.0), 2)

    # Un MAD nécessite un minimum de points pour être statistiquement
    # significatif -- sous ce seuil on préfère prévenir l'agent plutôt que
    # renvoyer des stats calculées sur une poignée de valeurs. Même seuil
    # appliqué aux deux modes (recent et seasonal) pour rester cohérent.
    MIN_POINTS = 10

    mode_used = mode
    fallback_reason = None
    rows: list[tuple[str, float]]

    if mode in ("seasonal", "auto"):
        seasonal_rows = _fetch_seasonal_points(
            instance_id, metric_name, before_timestamp,
            lookback_days, tolerance_minutes, window,
        )
        if len(seasonal_rows) >= MIN_POINTS:
            rows = seasonal_rows
            mode_used = "seasonal"
        elif mode == "seasonal":
            result = {
                "instance_id": instance_id,
                "metric_name": metric_name,
                "mode_used": "seasonal",
                "warning": (
                    f"Historique saisonnier insuffisant ({len(seasonal_rows)} points trouvés "
                    f"sur {lookback_days}j à ±{tolerance_minutes}min autour de la même heure, "
                    f"{MIN_POINTS} minimum). Essaie mode='auto' (repli automatique sur 'recent') "
                    f"ou mode='recent'. Ne pas halluciner de seuil."
                ),
                "n_points_used": len(seasonal_rows),
                "n_points_requested": window,
                "data_completeness": _data_completeness(len(seasonal_rows), window),
            }
            _log_tool_call("get_baseline_stats", call_kwargs,
                            f"seasonal insuffisant ({len(seasonal_rows)} pts)",
                            (time.perf_counter() - _t0) * 1000)
            return result
        else:  # mode == "auto", pas assez de points saisonniers -> repli
            fallback_reason = (
                f"seulement {len(seasonal_rows)} points saisonniers trouvés sur "
                f"{lookback_days}j à ±{tolerance_minutes}min (< {MIN_POINTS} requis) -- "
                f"repli sur mode 'recent'."
            )
            mode_used = "recent"

    if mode_used == "recent":
        recent_rows = _query(
            "SELECT timestamp, value FROM cloudwatch_metrics "
            "WHERE instance_id = ? AND metric_name = ? AND timestamp < ? "
            "ORDER BY timestamp DESC LIMIT ?",
            (instance_id, metric_name, before_timestamp, window),
        )
        if len(recent_rows) < MIN_POINTS:
            result = {
                "instance_id": instance_id,
                "metric_name": metric_name,
                "mode_used": "recent",
                "warning": (
                    f"Historique insuffisant avant {before_timestamp} pour établir une "
                    f"baseline fiable ({len(recent_rows)} points trouvés, {MIN_POINTS} minimum). "
                    f"Ne pas halluciner de seuil -- baser le jugement uniquement sur la "
                    f"forme de la série observée (get_metric_window)."
                ),
                "n_points_used": len(recent_rows),
                "n_points_requested": window,
                "data_completeness": _data_completeness(len(recent_rows), window),
            }
            if fallback_reason:
                result["fallback_reason"] = fallback_reason
            _log_tool_call("get_baseline_stats", call_kwargs,
                            f"insuffisant ({len(recent_rows)} pts)", (time.perf_counter() - _t0) * 1000)
            return result
        rows = recent_rows[::-1]  # ordre chronologique

    values = [v for _, v in rows]

    median = statistics.median(values)
    mad = statistics.median([abs(v - median) for v in values])
    # Évite un MAD nul (série de référence parfaitement plate) qui rendrait
    # tout seuil trivialement franchi par le moindre bruit.
    scaled_mad = 1.4826 * mad if mad > 0 else 1e-6

    result = {
        "instance_id": instance_id,
        "metric_name": metric_name,
        "mode_used": mode_used,
        "median": median,
        "mad": mad,
        "suggested_threshold_low": median - 3 * scaled_mad,
        "suggested_threshold_high": median + 3 * scaled_mad,
        "n_points_used": len(values),
        "n_points_requested": window,
        "data_completeness": _data_completeness(len(values), window),
        "reference_range": {"start": rows[0][0], "end": rows[-1][0]},
    }
    if fallback_reason:
        result["fallback_reason"] = fallback_reason
    if mode_used == "seasonal":
        result["seasonal_params"] = {
            "lookback_days": lookback_days,
            "tolerance_minutes": tolerance_minutes,
            "note": "reference_range non-contigu (points de plusieurs jours distincts).",
        }

    _log_tool_call(
        "get_baseline_stats", call_kwargs,
        f"mode={mode_used} median={median:.2f} mad={mad:.2f} range=[{result['suggested_threshold_low']:.2f}, "
        f"{result['suggested_threshold_high']:.2f}] completeness={result['data_completeness']}",
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

    PHASE 2 renvoie aussi `changepoint`, calculé par CUSUM offline sur la
    même fenêtre (cf. changepoint.py) -- c'est ta SEULE source légitime pour
    approx_timestamp. NE JUGE JAMAIS à l'oeil où l'anomalie "semble" avoir
    commencé. Si `changepoint.detected` est true, utilise
    `changepoint.onset_timestamp` comme approx_timestamp. Si false (rupture
    pas assez marquée par rapport au bruit de fond, ou fenêtre trop courte --
    voir `changepoint.reason`), retombe sur `exceedance.run_start_timestamp`
    comme actuellement.

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
          },
          "changepoint": {
            "detected": true/false,
            "onset_timestamp": ... | null,
            "mean_before": ..., "mean_after": ..., "effect_size_std": ...,
            "reason": "..."
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

        # --- Force du signal (pour un score de confiance CONTINU, pas juste passed:bool) ---
        # `passed` dit SI le run est assez long, mais pas de COMBIEN on est au-dessus du
        # minimum requis, ni de quelle AMPLEUR est le dépassement par rapport au seuil.
        # Deux cas "passed=true" peuvent avoir une fiabilité très différente : un run de
        # 3/3 points tout juste au-dessus du seuil n'est pas aussi solide qu'un run de
        # 20 points largement au-dessus. On calcule ici deux chiffres déterministes
        # (jamais estimés par le LLM) que l'agent recopie tels quels dans sa sortie, et
        # que le verifier recalcule indépendamment (cf. verifier.py::compute_confidence) :
        #   - run_ratio: longueur du run / minimum requis (>=1.0 si passed)
        #   - mean_excess_ratio: dépassement relatif MOYEN des points du run par rapport
        #     au seuil, ex: 0.20 = points en moyenne 20% au-dessus du seuil
        run_ratio = round(best_len / min_consecutive, 3) if min_consecutive > 0 else 0.0
        if best_start_idx is not None and threshold != 0:
            run_values = [v for _, v in rows[best_start_idx: best_start_idx + best_len]]
            if direction == "above":
                excess_ratios = [(v - threshold) / abs(threshold) for v in run_values]
            else:
                excess_ratios = [(threshold - v) / abs(threshold) for v in run_values]
            mean_excess_ratio = round(sum(excess_ratios) / len(excess_ratios), 3)
        else:
            mean_excess_ratio = None

        result["exceedance"] = {
            "threshold": threshold,
            "direction": direction,
            "longest_consecutive_run": best_len,
            "run_start_timestamp": run_start_ts,
            "run_end_timestamp": run_end_ts,
            "passed": passed,
            "min_consecutive_required": min_consecutive,
            "run_ratio": run_ratio,
            "mean_excess_ratio": mean_excess_ratio,
        }

        # Changepoint (CUSUM offline) sur la même fenêtre `rows` que
        # l'exceedance ci-dessus -- même fetch, donc mêmes points exactement,
        # pas de risque de désaccord de fenêtre entre les deux champs.
        cp_values = [v for _, v in rows]
        cp_timestamps = [ts for ts, _ in rows]
        cp_result = detect_changepoint(cp_values)
        onset_ts = cp_timestamps[cp_result.index] if cp_result.detected and cp_result.index is not None else None

        result["changepoint"] = {
            "detected": cp_result.detected,
            "onset_timestamp": onset_ts,
            "mean_before": cp_result.mean_before,
            "mean_after": cp_result.mean_after,
            "effect_size_std": cp_result.effect_size_std,
            "reason": cp_result.reason,
        }

    _log_tool_call(
        "evaluate_metric_window", call_kwargs,
        f"classification={classification}"
        + (f" passed={result['exceedance']['passed']} "
           f"run={result['exceedance']['longest_consecutive_run']} "
           f"changepoint={result['changepoint']['detected']}"
           if result["exceedance"] else " (phase 1, pas de seuil)"),
        (time.perf_counter() - _t0) * 1000,
    )
    return result


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson standard, stdlib only. None si variance nulle sur l'une des deux séries."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx ** 0.5 * vy ** 0.5)


@mcp.tool()
def check_multivariate_correlation(
    instance_id: str,
    metric_names: list[str],
    end_timestamp: str | None = None,
    limit: int = 30,
    z_threshold: float = 2.5,
) -> dict:
    """
    Tool 6/6 -- calcule un chiffre réel pour juger si plusieurs métriques
    bougent EN MÊME TEMPS, au lieu de laisser le LLM "impressionner" une
    corrélation en lisant get_metric_window à l'oeil.

    A appeler UNIQUEMENT si describe_metrics_schema/get_metric_window a
    montré >= 2 métriques disponibles pour cette instance ET que tu
    soupçonnes un pattern multivarié (crypto-mining, DDoS, exfiltration --
    cf. prompt.py). Ne pas appeler pour une instance à une seule métrique.

    Renvoie, pour chaque PAIRE de métriques, sur la fenêtre alignée par
    position (pas par égalité stricte de timestamp -- cf. `aligned_timestamps`
    de get_metric_window) :
      - pearson_r : corrélation linéaire classique sur toute la fenêtre.
        Capte "les deux séries montent/descendent ensemble" en tendance
        globale, mais peut être trompeur si le pattern n'est présent que
        sur une sous-partie de la fenêtre.
      - concurrent_exceedance_count : nombre de points où les DEUX métriques
        sont SIMULTANÉMENT à plus de `z_threshold` écarts-types robustes
        (médiane + MAD*1.4826, calculés sur cette même fenêtre) de leur
        propre valeur centrale. Complète pearson_r: capte un pic conjoint
        ponctuel même si la corrélation globale est faible.
      - concurrent_exceedance_timestamps : timestamps concernés (pour citer
        un approx_timestamp précis dans ta sortie).

    N'invente JAMAIS de lien de causalité ou de "corrélation" en texte libre
    si tu n'as pas appelé ce tool -- pearson_r et concurrent_exceedance_count
    sont ta SEULE source légitime pour affirmer que deux métriques bougent
    ensemble. Cite ces chiffres directement dans `description`
    (ex: "pearson_r=0.92 entre CPUUtilization et NetworkIn, 4 points de
    dépassement simultané").

    Args:
        instance_id: identifiant de l'instance/cas.
        metric_names: >= 2 noms exacts de métriques (voir describe_metrics_schema).
        end_timestamp: si fourni, borne la fenêtre (même sémantique que les
            autres tools -- garde-le identique aux appels get_metric_window/
            evaluate_metric_window sur la même analyse).
        limit: nombre de points par métrique (défaut 30).
        z_threshold: seuil en écarts-types robustes pour qu'un point compte
            comme "en exceedance" (défaut 2.5).

    Returns:
        {
          "instance_id": ..., "window_size": ...,
          "pairs": [
            {
              "metric_a": ..., "metric_b": ...,
              "pearson_r": ... | null,
              "concurrent_exceedance_count": ...,
              "concurrent_exceedance_timestamps": ["...", ...],
              "warning": "..." (si une métrique est absente ou variance nulle)
            }, ...
          ]
        }
    """
    _t0 = time.perf_counter()
    call_kwargs = {"instance_id": instance_id, "metric_names": metric_names,
                    "end_timestamp": end_timestamp, "limit": limit,
                    "z_threshold": z_threshold}

    if len(metric_names) < 2:
        result = {
            "instance_id": instance_id,
            "pairs": [],
            "warning": "Fournis au moins 2 métriques -- ce tool sert à comparer des métriques "
                       "entre elles, pas à en analyser une seule (utilise evaluate_metric_window).",
        }
        _log_tool_call("check_multivariate_correlation", call_kwargs,
                        "< 2 métriques fournies", (time.perf_counter() - _t0) * 1000)
        return result

    series: dict[str, list[tuple[str, float]]] = {}
    for m in metric_names:
        rows = _fetch_window(instance_id, m, end_timestamp, limit)
        if rows:
            series[m] = rows

    missing = [m for m in metric_names if m not in series]
    window_size = min((len(v) for v in series.values()), default=0)

    pairs = []
    names = list(series.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            rows_a, rows_b = series[a], series[b]
            n = min(len(rows_a), len(rows_b))
            # Alignement par POSITION (les n derniers points de chaque série),
            # pas par égalité stricte de timestamp -- cf. aligned_timestamps
            # de get_metric_window si les deux métriques ont des trous différents.
            vals_a = [v for _, v in rows_a[-n:]]
            vals_b = [v for _, v in rows_b[-n:]]
            ts_a = [t for t, _ in rows_a[-n:]]

            pair_result = {
                "metric_a": a, "metric_b": b,
                "pearson_r": _pearson(vals_a, vals_b),
                "concurrent_exceedance_count": 0,
                "concurrent_exceedance_timestamps": [],
            }

            def _robust_z(values: list[float]) -> list[float]:
                med = statistics.median(values)
                mad = statistics.median([abs(v - med) for v in values])
                scaled_mad = 1.4826 * mad if mad > 0 else 1e-6
                return [(v - med) / scaled_mad for v in values]

            if n >= 3:
                z_a = _robust_z(vals_a)
                z_b = _robust_z(vals_b)
                for k in range(n):
                    if abs(z_a[k]) >= z_threshold and abs(z_b[k]) >= z_threshold:
                        pair_result["concurrent_exceedance_count"] += 1
                        pair_result["concurrent_exceedance_timestamps"].append(ts_a[k])
            else:
                pair_result["warning"] = "Fenêtre trop courte (< 3 points communs) pour un z-score fiable."

            pairs.append(pair_result)

    result = {"instance_id": instance_id, "window_size": window_size, "pairs": pairs}
    if missing:
        result["warning"] = f"Métriques absentes pour cette instance (ignorées): {missing}"

    _log_tool_call(
        "check_multivariate_correlation", call_kwargs,
        f"{len(pairs)} paire(s): " + ", ".join(
            f"{p['metric_a']}/{p['metric_b']}=r:{p['pearson_r']}" for p in pairs
        ),
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