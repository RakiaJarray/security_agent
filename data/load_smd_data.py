"""
Charge SMD (Server Machine Dataset, Su et al. 2019, via
github.com/NetManAIOps/OmniAnomaly/ServerMachineDataset) dans la même base
SQLite `cloudwatch_metrics.db`.

Contrairement à CloudAnoBench (cas discrets, label au niveau du cas entier),
SMD est un FLUX CONTINU par machine avec un label AU POINT (0/1 par
timestamp) -- c'est le même modèle que NAB. On réutilise donc le schéma
`anomaly_windows` (pas `scenario_labels`) et le `eval.py` d'origine
fonctionne tel quel dessus, sans modification.

Pas de vrais timestamps dans SMD (juste un ordre de lignes) -- on synthétise
un timestamp à intervalle de 1 minute par ligne, cohérent avec l'intervalle
de collecte documenté par les auteurs. Pas de noms de métriques significatifs
non plus (colonnes anonymisées) -- on les nomme dim_01, dim_02, ... dim_38 ;
`describe_metrics_schema` (voir metrics_mcp_server.py) permet à l'agent de
les découvrir sans les connaître à l'avance.

Chaque machine a un train (toujours normal, pas d'anomalie) et un test
(peut contenir des anomalies, labellisées point par point dans test_label).
On charge train + test concaténés comme une seule série continue par
instance, ce qui permet à l'agent d'avoir un historique "normal" avant les
anomalies -- comportement réaliste, alors que ne charger que test isolerait
l'agent sans aucune baseline.

Usage:
    python load_smd_data.py --root /chemin/vers/ServerMachineDataset
    (attendu: root/train/*.txt, root/test/*.txt, root/test_label/*.txt)
"""
import argparse
import csv
import os
import sqlite3
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), "cloudwatch_metrics.db")

SAMPLING_INTERVAL_MINUTES = 1
# Epoch arbitraire -- SMD n'a pas de vrais timestamps, seule la position
# relative des points compte pour l'agent (fenêtres glissantes).
BASE_EPOCH = datetime(2024, 1, 1, 0, 0, 0)


def ensure_schema(conn: sqlite3.Connection, fresh: bool):
    cur = conn.cursor()
    if fresh:
        cur.execute("DROP TABLE IF EXISTS cloudwatch_metrics")
        cur.execute("DROP TABLE IF EXISTS anomaly_windows")
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
    else:
        # Mode additif: permet de charger SMD en plus d'un dataset déjà en
        # base (ex. NAB ou CloudAnoBench) sans écraser -- utile pour un
        # papier qui compare plusieurs datasets dans le même run d'éval.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS cloudwatch_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                instance_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                value REAL NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS anomaly_windows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                instance_id TEXT NOT NULL,
                window_start TEXT NOT NULL,
                window_end TEXT NOT NULL
            )
        """)
    conn.commit()


def read_txt_matrix(path: str) -> list[list[float]]:
    """SMD .txt = CSV sans header, valeurs séparées par virgules."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            rows.append([float(v) for v in row])
    return rows


def read_label_vector(path: str) -> list[int]:
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip()]


def anomaly_segments(labels: list[int]) -> list[tuple[int, int]]:
    """Convertit un vecteur binaire 0/1 en segments contigus (start_idx, end_idx)."""
    segments = []
    start = None
    for i, v in enumerate(labels):
        if v == 1 and start is None:
            start = i
        elif v == 0 and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(labels) - 1))
    return segments


def load_machine(conn: sqlite3.Connection, root: str, machine_name: str) -> tuple[int, int]:
    train_path = os.path.join(root, "train", f"{machine_name}.txt")
    test_path = os.path.join(root, "test", f"{machine_name}.txt")
    label_path = os.path.join(root, "test_label", f"{machine_name}.txt")

    if not (os.path.exists(train_path) and os.path.exists(test_path) and os.path.exists(label_path)):
        print(f"  [skip] {machine_name}: fichiers train/test/test_label incomplets")
        return 0, 0

    train_rows = read_txt_matrix(train_path)
    test_rows = read_txt_matrix(test_path)
    test_labels = read_label_vector(label_path)

    if len(test_rows) != len(test_labels):
        print(f"  [warn] {machine_name}: test ({len(test_rows)} lignes) et "
              f"test_label ({len(test_labels)} lignes) de tailles différentes -- "
              f"troncature au plus court")
        n = min(len(test_rows), len(test_labels))
        test_rows, test_labels = test_rows[:n], test_labels[:n]

    n_dims = len(train_rows[0]) if train_rows else (len(test_rows[0]) if test_rows else 0)
    instance_id = f"smd_{machine_name}"

    cur = conn.cursor()
    n_rows_inserted = 0

    # train + test concaténés en une seule série continue (train toujours
    # normal, test peut contenir des anomalies) -- offset temporel constant
    # pour que get_metric_points / eval.py voient une seule timeline triable.
    all_rows = train_rows + test_rows
    for idx, row in enumerate(all_rows):
        ts = (BASE_EPOCH + timedelta(minutes=idx * SAMPLING_INTERVAL_MINUTES)).strftime("%Y-%m-%d %H:%M:%S")
        rows_to_insert = [
            (instance_id, f"dim_{dim_idx + 1:02d}", ts, row[dim_idx])
            for dim_idx in range(min(n_dims, len(row)))
        ]
        cur.executemany(
            "INSERT INTO cloudwatch_metrics (instance_id, metric_name, timestamp, value) "
            "VALUES (?, ?, ?, ?)",
            rows_to_insert,
        )
        n_rows_inserted += len(rows_to_insert)

    # Les anomalies labellisées ne concernent que la portion "test", décalée
    # de len(train_rows) dans la timeline concaténée.
    offset = len(train_rows)
    segments = anomaly_segments(test_labels)
    for start_idx, end_idx in segments:
        window_start = (BASE_EPOCH + timedelta(minutes=(offset + start_idx) * SAMPLING_INTERVAL_MINUTES)).strftime("%Y-%m-%d %H:%M:%S")
        window_end = (BASE_EPOCH + timedelta(minutes=(offset + end_idx) * SAMPLING_INTERVAL_MINUTES)).strftime("%Y-%m-%d %H:%M:%S")
        cur.execute(
            "INSERT INTO anomaly_windows (instance_id, window_start, window_end) VALUES (?, ?, ?)",
            (instance_id, window_start, window_end),
        )

    conn.commit()
    return 1, n_rows_inserted


def build_db(root: str, fresh: bool, machines_filter: list[str] | None):
    conn = sqlite3.connect(DB_PATH)
    ensure_schema(conn, fresh)

    train_dir = os.path.join(root, "train")
    if not os.path.isdir(train_dir):
        print(f"ERREUR: {train_dir} introuvable -- vérifie --root "
              f"(doit contenir train/, test/, test_label/)")
        return

    machine_names = sorted(
        f.replace(".txt", "") for f in os.listdir(train_dir) if f.endswith(".txt")
    )
    if machines_filter:
        machine_names = [m for m in machine_names if m in machines_filter]

    total_machines = 0
    total_rows = 0
    for machine_name in machine_names:
        print(f"Chargement {machine_name}...")
        n_m, n_rows = load_machine(conn, root, machine_name)
        total_machines += n_m
        total_rows += n_rows

    conn.close()
    print(f"\nOK -> {DB_PATH}")
    print(f"  {total_machines} machines chargées, {total_rows} points de métrique")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", required=True,
        help="Dossier ServerMachineDataset (contenant train/, test/, test_label/)",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Repart d'une base vide (par défaut: mode additif, garde les données "
             "déjà chargées d'un autre dataset, ex. NAB ou CloudAnoBench)",
    )
    parser.add_argument(
        "--machines", default=None,
        help="Sous-ensemble de machines à charger, ex: 'machine-1-1,machine-1-2' "
             "(par défaut: les 28 machines)",
    )
    args = parser.parse_args()
    machines_filter = args.machines.split(",") if args.machines else None
    build_db(args.root, args.fresh, machines_filter)
