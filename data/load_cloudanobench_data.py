"""
Charge CloudAnoBench (anom_dataset/, mali_dataset/, norm_dataset/) dans la
même base SQLite `cloudwatch_metrics.db`, avec un schéma compatible avec
agent.py/eval.py existants (table cloudwatch_metrics: instance_id,
metric_name, timestamp, value), plus deux tables spécifiques à
CloudAnoBench:

  - scenario_labels : le label est au niveau du *cas* entier (pas une
    fenêtre temporelle comme dans NAB). instance_id -> dataset_type
    (anom/mali/norm), scenario_id, is_anomaly.
  - cloudwatch_logs : le contenu brut du .log associé à chaque cas, pour
    un futur tool "logs" du ReAct agent (non utilisé par agent.py v1,
    qui ne consomme que les métriques).

Chaque CSV de CloudAnoBench est multi-métriques (une colonne par métrique :
cpu_usage, mem_usage, disk_io, net_in, net_out, ...). On "melt" ces colonnes
en lignes (une par métrique) pour rester compatible avec le schéma
mono-métrique existant.

Usage:
    python load_cloudanobench_data.py --root /chemin/vers/cloudanobench_raw
"""
import argparse
import csv
import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "cloudwatch_metrics.db")

DATASET_DIRS = {
    "anom": "anom_dataset",
    "mali": "mali_dataset",
    "norm": "norm_dataset",
}
# Seul norm_dataset est "normal" (deceptive normal) ; anom et mali sont
# tous deux des anomalies réelles (l'un système, l'autre malveillant).
IS_ANOMALY_BY_TYPE = {"anom": True, "mali": True, "norm": False}

NON_METRIC_COLUMNS = {"timestamp"}
# Certains fichiers du dataset peuvent nommer la colonne temps différemment
# ou avec une casse différente -- on les reconnaît tous et on les normalise
# en "timestamp" en interne.
TIMESTAMP_ALIASES = {"timestamp", "time", "datetime", "date_time", "ts"}


def ensure_schema(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS cloudwatch_metrics")
    cur.execute("DROP TABLE IF EXISTS scenario_labels")
    cur.execute("DROP TABLE IF EXISTS cloudwatch_logs")
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
        CREATE TABLE scenario_labels (
            instance_id TEXT PRIMARY KEY,
            dataset_type TEXT NOT NULL,      -- anom | mali | norm
            scenario_id TEXT NOT NULL,       -- e.g. scenario_1
            case_number TEXT NOT NULL,
            is_anomaly INTEGER NOT NULL      -- 0/1, label au niveau du cas entier
        )
    """)
    cur.execute("""
        CREATE TABLE cloudwatch_logs (
            instance_id TEXT PRIMARY KEY,
            log_text TEXT
        )
    """)
    conn.commit()


def parse_case_filename(csv_filename: str) -> tuple[str, str]:
    """
    anom_1_3.csv -> scenario_id='scenario_1', case_number='3'
    Le préfixe dataset_type ('anom'/'mali'/'norm') est déjà connu du
    dossier parent, donc on ne le redérive pas d'ici.
    """
    stem = csv_filename.replace(".csv", "")
    parts = stem.split("_")
    # ex: ['anom', '1', '3']
    scenario_num = parts[1] if len(parts) > 1 else "0"
    case_number = parts[2] if len(parts) > 2 else "0"
    return f"scenario_{scenario_num}", case_number


def load_dataset_dir(conn: sqlite3.Connection, root: str, dataset_type: str):
    dir_path = os.path.join(root, DATASET_DIRS[dataset_type])
    if not os.path.isdir(dir_path):
        print(f"  [skip] {dir_path} introuvable")
        return 0, 0

    cur = conn.cursor()
    n_instances = 0
    n_rows = 0

    # Les CSV/LOG peuvent être directement dans le dossier du dataset_type,
    # ou organisés en sous-dossiers scenario_N/ selon la version du dump --
    # on gère les deux en marchant l'arborescence.
    for dirpath, _, filenames in os.walk(dir_path):
        for filename in sorted(filenames):
            if not filename.endswith(".csv"):
                continue

            scenario_id, case_number = parse_case_filename(filename)
            # Si le fichier est déjà dans un sous-dossier scenario_N/, on
            # préfère ce nom de dossier (plus fiable que le parsing du nom).
            parent_dir = os.path.basename(dirpath)
            if parent_dir.startswith("scenario_"):
                scenario_id = parent_dir

            instance_id = f"{dataset_type}_{scenario_id}_{case_number}"
            csv_path = os.path.join(dirpath, filename)

            try:
                with open(csv_path, newline="", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    fieldnames = reader.fieldnames or []
                    if not fieldnames:
                        print(f"  [skip] {csv_path}: fichier vide ou sans en-tête")
                        continue

                    # Colonne timestamp: repérage insensible à la casse/alias,
                    # normalisation du nom exact tel que présent dans le fichier.
                    ts_col = None
                    for col in fieldnames:
                        if col.strip().lower() in TIMESTAMP_ALIASES:
                            ts_col = col
                            break
                    if ts_col is None:
                        print(f"  [skip] {csv_path}: aucune colonne timestamp trouvée "
                              f"(colonnes: {fieldnames})")
                        continue

                    metric_cols = [c for c in fieldnames if c != ts_col]

                    rows_to_insert = []
                    for row in reader:
                        ts = row.get(ts_col)
                        if not ts:
                            continue
                        for metric in metric_cols:
                            raw_val = row.get(metric)
                            if raw_val in (None, ""):
                                continue
                            try:
                                val = float(raw_val)
                            except ValueError:
                                continue
                            rows_to_insert.append((instance_id, metric, ts, val))
            except Exception as e:
                print(f"  [skip] {csv_path}: erreur de lecture ({e})")
                continue

            if not rows_to_insert:
                print(f"  [skip] {csv_path}: aucune ligne exploitable")
                continue

            cur.executemany(
                "INSERT INTO cloudwatch_metrics (instance_id, metric_name, timestamp, value) "
                "VALUES (?, ?, ?, ?)",
                rows_to_insert,
            )
            n_rows += len(rows_to_insert)

            cur.execute(
                "INSERT OR REPLACE INTO scenario_labels "
                "(instance_id, dataset_type, scenario_id, case_number, is_anomaly) "
                "VALUES (?, ?, ?, ?, ?)",
                (instance_id, dataset_type, scenario_id, case_number,
                 int(IS_ANOMALY_BY_TYPE[dataset_type])),
            )

            log_path = csv_path.replace(".csv", ".log")
            if os.path.exists(log_path):
                try:
                    with open(log_path, encoding="utf-8", errors="replace") as f:
                        log_text = f.read()
                    cur.execute(
                        "INSERT OR REPLACE INTO cloudwatch_logs (instance_id, log_text) VALUES (?, ?)",
                        (instance_id, log_text),
                    )
                except Exception as e:
                    print(f"  [warn] {log_path}: log non chargé ({e})")

            n_instances += 1

    conn.commit()
    return n_instances, n_rows


def build_db(root: str):
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    ensure_schema(conn)

    total_instances = 0
    total_rows = 0
    for dataset_type in DATASET_DIRS:
        print(f"Chargement {dataset_type}_dataset...")
        n_inst, n_rows = load_dataset_dir(conn, root, dataset_type)
        print(f"  -> {n_inst} cas, {n_rows} points métrique")
        total_instances += n_inst
        total_rows += n_rows

    conn.close()
    print(f"\nOK -> {DB_PATH}")
    print(f"  {total_instances} cas au total, {total_rows} points de métrique")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", required=True,
        help="Dossier racine contenant anom_dataset/, mali_dataset/, norm_dataset/ "
             "(ex: data/cloudanobench_raw)",
    )
    args = parser.parse_args()
    build_db(args.root)