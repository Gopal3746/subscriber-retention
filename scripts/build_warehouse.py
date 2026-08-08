from pathlib import Path
import pandas as pd

from retention_ltv.config import PROCESSED_CSV
from retention_ltv.features import prepare_customers
from retention_ltv.warehouse import build_warehouse

con = build_warehouse()
PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
con.execute(f"COPY customers TO '{PROCESSED_CSV.as_posix()}' (HEADER, DELIMITER ',')")
print(con.execute("SELECT COUNT(*) AS customers, AVG(churn) AS churn_rate FROM customers").df().to_string(index=False))
print(f"Wrote {PROCESSED_CSV}")
con.close()
