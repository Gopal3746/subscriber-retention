from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
art = ROOT / "artifacts"
metrics = json.loads((art / "metrics.json").read_text())
shap_df = pd.read_csv(art / "shap_global.csv").head(7).sort_values("mean_abs_shap")
ltv = pd.read_csv(art / "ltv_by_segment.csv")
survival = pd.read_csv(art / "survival_curves.csv")
risk = ltv[ltv["segment_dimension"] == "risk_segment"].set_index("segment")
contract_surv = survival[survival["cohort_dimension"] == "Contract"]

fig = plt.figure(figsize=(15, 8.5))
grid = fig.add_gridspec(2, 3, height_ratios=[0.75, 2.6])
ax0 = fig.add_subplot(grid[0, :]); ax0.axis("off")
ax0.text(0.00, 0.78, "Subscription Retention & LTV Analysis", fontsize=24, fontweight="bold")
ax0.text(0.00, 0.30, f"7,043 customers  |  XGBoost ROC-AUC {metrics['holdout_metrics']['xgboost']['roc_auc']:.3f}  |  Observed churn {metrics['observed_churn_rate']:.1%}  |  At-risk LTV gap {metrics['at_risk_ltv_discount_vs_healthy']:.0%}", fontsize=12.5)

ax1 = fig.add_subplot(grid[1, 0])
for name, group in contract_surv.groupby("cohort_value"):
    ax1.plot(group["month"], group["survival"], label=name, linewidth=2)
ax1.set_title("Kaplan–Meier retention by contract")
ax1.set_xlabel("Month N"); ax1.set_ylabel("Retention probability"); ax1.set_ylim(0, 1.02); ax1.legend(fontsize=8)

ax2 = fig.add_subplot(grid[1, 1])
ax2.barh(shap_df["source_feature"], shap_df["mean_abs_shap"])
ax2.set_title("Global churn drivers (mean |SHAP|)")
ax2.set_xlabel("Importance")

ax3 = fig.add_subplot(grid[1, 2])
order = [x for x in ["At-risk", "Middle", "Healthy"] if x in risk.index]
vals = [risk.loc[x, "estimated_ltv_72m"] for x in order]
ax3.bar(order, vals)
ax3.set_title("72-month survival-adjusted LTV")
ax3.set_ylabel("Estimated revenue ($)")
for i, v in enumerate(vals):
    ax3.text(i, v, f"${v:,.0f}", ha="center", va="bottom", fontsize=9)
fig.tight_layout()
fig.savefig(art / "dashboard_preview.png", dpi=160, bbox_inches="tight")
print(art / "dashboard_preview.png")
