from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
PROCESSED_CSV = ROOT / "data" / "processed" / "customers.csv"
WAREHOUSE = ROOT / "data" / "retention.duckdb"
ARTIFACTS = ROOT / "artifacts"
RANDOM_STATE = 42
TEST_SIZE = 0.20
SURVIVAL_HORIZON_MONTHS = 72
