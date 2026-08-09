import math
from retention_ltv.survival import kaplan_meier, restricted_mean_survival_time


def test_kaplan_meier_simple_case():
    curve = kaplan_meier([1, 2, 2], [1, 1, 0], horizon=3)
    assert math.isclose(curve.loc[curve.month == 1, "survival"].iloc[0], 2/3, rel_tol=1e-9)
    assert math.isclose(curve.loc[curve.month == 2, "survival"].iloc[0], 1/3, rel_tol=1e-9)
    assert restricted_mean_survival_time(curve, horizon=3) > 0
