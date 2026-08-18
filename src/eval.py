"""
Évalue le Metrics Agent (agent LLM + Symbolic Verifier) sur toutes les
instances NAB/realAWSCloudwatch, contre les fenêtres d'anomalie labellisées.

Usage:
    export GOOGLE_API_KEY=...
    python eval.py
"""
import asyncio
import json
import os
import sqlite3
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from agent import run_metrics_agent

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cloudwatch_metrics.db")
WINDOW_SIZE = 30


def get_instances():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT instance_id, metric_name FROM cloudwatch_metrics")
    rows = cur.fetchall()
    conn.close()
    return rows


def get_last_window(instance_id: str) -> list[tuple[str, float]]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT timestamp, value FROM cloudwatch_metrics WHERE instance_id=? "
        "ORDER BY timestamp DESC LIMIT ?",
        (instance_id, WINDOW_SIZE),
    )
    rows = cur.fetchall()[::-1]  # ordre chronologique
    conn.close()
    return rows


def get_baseline_window(instance_id: str, before_ts: str, size: int = 60) -> list[float]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT value FROM cloudwatch_metrics WHERE instance_id=? AND timestamp < ? "
        "ORDER BY timestamp DESC LIMIT ?",
        (instance_id, before_ts, size),
    )
    rows = [r[0] for r in cur.fetchall()]
    conn.close()
    return rows


def get_anomaly_windows(instance_id: str) -> list[tuple[str, str]]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT window_start, window_end FROM anomaly_windows WHERE instance_id=? "
        "ORDER BY window_start",
        (instance_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def get_window_ending_at(instance_id: str, end_ts: str, size: int = WINDOW_SIZE) -> list[tuple[str, float]]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT timestamp, value FROM cloudwatch_metrics WHERE instance_id=? AND timestamp <= ? "
        "ORDER BY timestamp DESC LIMIT ?",
        (instance_id, end_ts, size),
    )
    rows = cur.fetchall()[::-1]  # ordre chronologique
    conn.close()
    return rows


def get_series_bounds(instance_id: str) -> tuple[str, str]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT MIN(timestamp), MAX(timestamp) FROM cloudwatch_metrics WHERE instance_id=?",
        (instance_id,),
    )
    row = cur.fetchone()
    conn.close()
    return row


def pick_control_timestamp(instance_id: str, windows: list[tuple[str, str]]) -> str | None:
    """
    Choisit un end_timestamp pour une fenêtre "normale" de contrôle : le dernier
    point de la série s'il n'est dans aucune fenêtre d'anomalie labellisée, sinon
    None (série trop courte / entièrement recouverte par des anomalies -- rare).
    """
    _, max_ts = get_series_bounds(instance_id)
    if max_ts is None:
        return None
    in_any_window = any(start <= max_ts <= end for start, end in windows)
    return None if in_any_window else max_ts


async def evaluate():
    instances = get_instances()
    results = []
    quota_exhausted = False

    for instance_id, metric_name in instances:
        print(f"=== {instance_id} ({metric_name}) ===", flush=True)
        windows = get_anomaly_windows(instance_id)

        # Cas cible: fenêtre se terminant au milieu de chaque anomalie labellisée
        # (test vrai positif). Cas contrôle: dernier point de la série s'il est
        # hors de toute fenêtre d'anomalie (test vrai négatif).
        cases = []
        for start, end in windows:
            # timestamp au milieu de la fenêtre d'anomalie -> teste la détection
            # au coeur de l'anomalie plutôt qu'à son tout début.
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute(
                "SELECT timestamp FROM cloudwatch_metrics WHERE instance_id=? "
                "AND timestamp BETWEEN ? AND ? ORDER BY timestamp",
                (instance_id, start, end),
            )
            in_window_ts = [r[0] for r in cur.fetchall()]
            conn.close()
            if not in_window_ts:
                continue
            mid_ts = in_window_ts[len(in_window_ts) // 2]
            cases.append(("anomaly", mid_ts))

        control_ts = pick_control_timestamp(instance_id, windows)
        if control_ts:
            cases.append(("normal", control_ts))

        for case_label, end_ts in cases:
            print(f"  --- fenêtre @ {end_ts} (attendu: {case_label}) ---", flush=True)
            try:
                agent_output = await run_metrics_agent(instance_id, metric_name, end_ts)
            except Exception as e:
                msg = str(e)
                if "GenerateRequestsPerDay" in msg or "RequestsPerDayPerProjectPerModel" in msg:
                    # Quota journalier épuisé -- tous les appels restants échoueront
                    # aussi, pas la peine de parcourir le reste des instances pour
                    # rien. On arrête proprement l'éval en gardant ce qui a déjà
                    # été collecté, plutôt que de laisser boucler jusqu'au bout.
                    print(
                        f"\n[ARRÊT] Quota journalier Google AI épuisé -- éval interrompue "
                        f"à l'instance {instance_id}/{metric_name}. Résultats partiels "
                        f"sauvegardés ci-dessous. Réessaie après le reset du quota "
                        f"(généralement minuit Pacific Time) ou avec une clé payante.",
                        file=sys.stderr,
                    )
                    quota_exhausted = True
                    break
                print(f"    ERREUR agent: {e}")
                continue

            window = get_window_ending_at(instance_id, end_ts)
            raw_values = [v for _, v in window]

            final_is_anomaly = agent_output.get("is_anomaly")
            ground_truth = case_label == "anomaly"

            record = {
                "instance_id": instance_id,
                "end_timestamp": end_ts,
                "case_label": case_label,
                "agent_is_anomaly": agent_output.get("is_anomaly"),
                "agent_pattern": agent_output.get("pattern"),
                "verifier_final_is_anomaly": final_is_anomaly,
                "ground_truth_has_anomaly": ground_truth,
            }
            results.append(record)
            print(f"    agent={record['agent_is_anomaly']} (pattern={record['agent_pattern']}) "
                  f"ground_truth={ground_truth}")

        if quota_exhausted:
            break

    tp = sum(1 for r in results if r["verifier_final_is_anomaly"] and r["ground_truth_has_anomaly"])
    fp = sum(1 for r in results if r["verifier_final_is_anomaly"] and not r["ground_truth_has_anomaly"])
    fn = sum(1 for r in results if not r["verifier_final_is_anomaly"] and r["ground_truth_has_anomaly"])
    tn = sum(1 for r in results if not r["verifier_final_is_anomaly"] and not r["ground_truth_has_anomaly"])

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print("\n=== Résultats", "(PARTIELS -- quota épuisé)" if quota_exhausted else "", "===")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f}")

    out_path = os.path.join(os.path.dirname(__file__), "..", "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "results": results,
                "metrics": {"precision": precision, "recall": recall, "f1": f1},
                "partial_quota_exhausted": quota_exhausted,
            },
            f, indent=2,
        )
    print(f"Détails sauvegardés dans {out_path}")


if __name__ == "__main__":
    asyncio.run(evaluate())