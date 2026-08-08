from __future__ import annotations

from pathlib import Path
import duckdb

from .config import RAW_CSV, WAREHOUSE


def build_warehouse(raw_csv: Path = RAW_CSV, db_path: Path = WAREHOUSE):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute("DROP TABLE IF EXISTS raw_telco")
    con.execute(
        "CREATE TABLE raw_telco AS SELECT * FROM read_csv_auto(?, header=true, all_varchar=false)",
        [str(raw_csv)],
    )
    sql_root = Path(__file__).resolve().parents[2] / "sql"
    for sql_file in sorted(sql_root.glob("*.sql")):
        con.execute(sql_file.read_text())
    return con
