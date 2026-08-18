"""
Charge les séries CloudWatch réelles de NAB (Numenta Anomaly Benchmark)
dans une base SQLite, avec une fenêtre glissante des 30 derniers points
par instance/métrique -- c'est cette base que le MCP server
(executeautomation/mcp-database-server) exposera au ReAct agent
via le tool `cloudwatch_metrics`.

Usage:
    python load_nab_data.py

Source des données: https://github.com/numenta/NAB
    data/realAWSCloudwatch/*.csv        -> séries brutes (timestamp, value)
    labels/combined_windows.json        -> fenêtres d'anomalie labellisées
"""
import csv
import json
import os
import sqlite3

NAB_DATA_DIR = "/tmp/NAB/data/realAWSCloudwatch"
NAB_LABELS_PATH = "/tmp/NAB/labels/combined_windows.json"
DB_PATH = os.path.join(os.path.dirname(__file__), "cloudwatch_metrics.db")


def infer_metric_name(filename: str) -> str:
    """ec2_cpu_utilization_24ae8d.csv -> CPUUtilization (style CloudWatch)."""
    name = filename.replace(".csv", "")
    if "cpu_utilization" in name:
        return "CPUUtilization"
    if "network_in" in name.lower() or "networkin" in name.lower():
        return "NetworkIn"
    if "disk_write" in name:
        return "DiskWriteBytes"
    if "request_count" in name:
        return "RequestCount"
    return "CustomMetric"


def build_db():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE cloudwatch_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            instance_id TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            value REAL NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE anomaly_windows (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            instance_id TEXT NOT NULL,
            window_start TEXT NOT NULL,
            window_end TEXT NOT NULL
        )
    """)

    with open(NAB_LABELS_PATH) as f:
        labels = json.load(f)

    n_rows = 0
    for filename in sorted(os.listdir(NAB_DATA_DIR)):
        if not filename.endswith(".csv"):
            continue
        instance_id = filename.replace(".csv", "")
        metric_name = infer_metric_name(filename)

        with open(os.path.join(NAB_DATA_DIR, filename)) as f:
            reader = csv.DictReader(f)
            rows = [(instance_id, metric_name, r["timestamp"], float(r["value"])) for r in reader]

        cur.executemany(
            "INSERT INTO cloudwatch_metrics (instance_id, metric_name, timestamp, value) VALUES (?, ?, ?, ?)",
            rows,
        )
        n_rows += len(rows)

        label_key = f"realAWSCloudwatch/{filename}"
        for window_start, window_end in labels.get(label_key, []):
            cur.execute(
                "INSERT INTO anomaly_windows (instance_id, window_start, window_end) VALUES (?, ?, ?)",
                (instance_id, window_start, window_end),
            )

    conn.commit()

    cur.execute("SELECT COUNT(DISTINCT instance_id) FROM cloudwatch_metrics")
    n_instances = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM anomaly_windows")
    n_windows = cur.fetchone()[0]

    conn.close()
    print(f"OK -> {DB_PATH}")
    print(f"  {n_instances} instances, {n_rows} points de métrique, {n_windows} fenêtres d'anomalie labellisées")


if __name__ == "__main__":
    build_db()
