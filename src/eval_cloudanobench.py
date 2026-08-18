"""
Évalue le Metrics Agent sur CloudAnoBench (chargé via
data/load_cloudanobench_data.py), contre les labels au niveau du cas
(scenario_labels), pas des fenêtres temporelles comme dans NAB.

Chaque cas CloudAnoBench a plusieurs métriques (cpu_usage, mem_usage,
disk_io, net_in, net_out...). L'agent actuel (agent.py) analyse une
métrique à la fois -> ici on lance l'agent sur chaque métrique du cas
et on agrège: le cas est jugé "anomalie" par le système si le Symbolic
Verifier le dit pour AU MOINS UNE métrique.

Usage:
    export GOOGLE_API_KEY=...
    python eval_cloudanobench.py [--limit N] [--dataset-types anom,mali,norm]
                                 [--metrics cpu_usage,mem_usage] [--sleep-between 3]
"""
import argparse
import asyncio
import json
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from agent import run_metrics_agent

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cloudwatch_metrics.db")


def get_cases(dataset_types: list[str]) -> list[tuple[str, str, int]]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    placeholders = ",".join("?" for _ in dataset_types)
    cur.execute(
        f"SELECT instance_id, dataset_type, is_anomaly FROM scenario_labels "
        f"WHERE dataset_type IN ({placeholders}) ORDER BY instance_id",
        dataset_types,
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def get_metrics_for_instance(instance_id: str, metrics_filter: list[str] | None) -> list[str]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT DISTINCT metric_name FROM cloudwatch_metrics WHERE instance_id=?",
        (instance_id,),
    )
    rows = [r[0] for r in cur.fetchall()]
    conn.close()
    if metrics_filter:
        # Garde uniquement les métriques demandées ET réellement présentes pour ce cas
        # (certains scénarios n'ont pas forcément toutes les colonnes -- cf.
        # describe_metrics_schema côté agent, même logique appliquée ici).
        rows = [m for m in rows if m in metrics_filter]
    return rows


async def evaluate(limit: int | None, dataset_types: list[str],
                    metrics_filter: list[str] | None, sleep_between: float):
    cases = get_cases(dataset_types)
    if limit:
        cases = cases[:limit]

    results = []

    for instance_id, dataset_type, is_anomaly in cases:
        ground_truth = bool(is_anomaly)
        metrics = get_metrics_for_instance(instance_id, metrics_filter)
        print(f"=== {instance_id} ({dataset_type}, ground_truth={ground_truth}) ===", flush=True)

        if not metrics:
            print(f"  [skip] aucune métrique correspondant au filtre {metrics_filter} pour ce cas")
            continue

        case_agent_verdicts = {}
        case_final_is_anomaly = False

        for metric_name in metrics:
            try:
                agent_output = await run_metrics_agent(instance_id, metric_name)
            except Exception as e:
                print(f"  ERREUR agent sur {metric_name}: {e}")
                continue

            verdict = bool(agent_output.get("is_anomaly"))
            case_agent_verdicts[metric_name] = {
                "agent_is_anomaly": agent_output.get("is_anomaly"),
                "pattern": agent_output.get("pattern"),
            }
            if verdict:
                case_final_is_anomaly = True
            print(f"  [{metric_name}] agent={verdict} (pattern={agent_output.get('pattern')})")

            if sleep_between > 0:
                # Throttling volontaire entre appels agent -- évite de taper le
                # quota/minute et de repasser par le retry-with-backoff (plus
                # lent au global que d'espacer les appels dès le départ).
                time.sleep(sleep_between)

        record = {
            "instance_id": instance_id,
            "dataset_type": dataset_type,
            "ground_truth_has_anomaly": ground_truth,
            "case_final_is_anomaly": case_final_is_anomaly,
            "per_metric": case_agent_verdicts,
        }
        results.append(record)
        print(f"  -> verdict cas: {case_final_is_anomaly} (attendu: {ground_truth})")

    tp = sum(1 for r in results if r["case_final_is_anomaly"] and r["ground_truth_has_anomaly"])
    fp = sum(1 for r in results if r["case_final_is_anomaly"] and not r["ground_truth_has_anomaly"])
    fn = sum(1 for r in results if not r["case_final_is_anomaly"] and r["ground_truth_has_anomaly"])
    tn = sum(1 for r in results if not r["case_final_is_anomaly"] and not r["ground_truth_has_anomaly"])

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print("\n=== Résultats CloudAnoBench ===")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f}")

    out_path = os.path.join(os.path.dirname(__file__), "..", "eval_cloudanobench_results.json")
    with open(out_path, "w") as f:
        json.dump(
            {"results": results, "metrics": {"precision": precision, "recall": recall, "f1": f1}},
            f, indent=2, ensure_ascii=False,
        )
    print(f"Détails sauvegardés dans {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limiter le nombre de cas (test rapide)")
    parser.add_argument(
        "--dataset-types", default="anom,mali,norm",
        help="Sous-ensemble à évaluer, ex: 'anom,norm' pour aller plus vite",
    )
    parser.add_argument(
        "--metrics", default=None,
        help="Sous-ensemble de métriques à tester par cas, ex: 'cpu_usage' ou "
             "'cpu_usage,mem_usage'. Par défaut: toutes les métriques disponibles.",
    )
    parser.add_argument(
        "--sleep-between", type=float, default=0,
        help="Pause en secondes entre chaque appel agent (throttling pour rester "
             "sous le quota API en tier gratuit, ex: 4).",
    )
    args = parser.parse_args()
    metrics_filter = args.metrics.split(",") if args.metrics else None
    asyncio.run(evaluate(args.limit, args.dataset_types.split(","), metrics_filter, args.sleep_between))