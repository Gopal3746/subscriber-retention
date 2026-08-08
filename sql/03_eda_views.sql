CREATE OR REPLACE VIEW churn_by_contract AS
SELECT Contract, COUNT(*) AS customers, AVG(churn) AS churn_rate, AVG(MonthlyCharges) AS avg_monthly_charge
FROM customers GROUP BY 1 ORDER BY churn_rate DESC;

CREATE OR REPLACE VIEW churn_by_payment_method AS
SELECT PaymentMethod, COUNT(*) AS customers, AVG(churn) AS churn_rate, AVG(MonthlyCharges) AS avg_monthly_charge
FROM customers GROUP BY 1 ORDER BY churn_rate DESC;

CREATE OR REPLACE VIEW churn_by_tenure_bucket AS
SELECT tenure_bucket, COUNT(*) AS customers, AVG(churn) AS churn_rate, AVG(MonthlyCharges) AS avg_monthly_charge
FROM customers GROUP BY 1;
