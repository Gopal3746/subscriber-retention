-- Telco is a one-time customer snapshot, so it cannot support honest calendar
-- signup-month cohorts. This SQL instead creates censoring-aware survival
-- cohorts by contract, which is defensible with tenure + churn-event status.

CREATE OR REPLACE VIEW contract_monthly_hazard AS
WITH months AS (
    SELECT UNNEST(generate_series(1, 72)) AS month_number
), contracts AS (
    SELECT DISTINCT Contract FROM customers
), risk AS (
    SELECT
        c.Contract,
        m.month_number,
        SUM(CASE WHEN x.tenure >= m.month_number THEN 1 ELSE 0 END) AS at_risk,
        SUM(CASE WHEN x.tenure = m.month_number AND x.churn = 1 THEN 1 ELSE 0 END) AS events
    FROM contracts c
    CROSS JOIN months m
    JOIN customers x ON x.Contract = c.Contract
    GROUP BY 1, 2
)
SELECT
    *,
    CASE WHEN at_risk = 0 THEN 0 ELSE events::DOUBLE / at_risk END AS hazard
FROM risk;

CREATE OR REPLACE VIEW contract_retention AS
WITH survival AS (
    SELECT
        Contract,
        month_number,
        at_risk,
        events,
        hazard,
        EXP(SUM(CASE WHEN hazard >= 1 THEN -1000 ELSE LN(1 - hazard) END)
            OVER (PARTITION BY Contract ORDER BY month_number)) AS survival_probability
    FROM contract_monthly_hazard
)
SELECT * FROM survival;
