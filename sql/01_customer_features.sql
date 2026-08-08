CREATE OR REPLACE TABLE customers AS
SELECT
    customerID,
    gender,
    SeniorCitizen,
    Partner,
    Dependents,
    tenure,
    PhoneService,
    MultipleLines,
    InternetService,
    OnlineSecurity,
    OnlineBackup,
    DeviceProtection,
    TechSupport,
    StreamingTV,
    StreamingMovies,
    Contract,
    PaperlessBilling,
    PaymentMethod,
    MonthlyCharges,
    TRY_CAST(NULLIF(TRIM(CAST(TotalCharges AS VARCHAR)), '') AS DOUBLE) AS TotalCharges,
    Churn,
    CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END AS churn,
    (
      CASE WHEN PhoneService = 'Yes' THEN 1 ELSE 0 END +
      CASE WHEN MultipleLines = 'Yes' THEN 1 ELSE 0 END +
      CASE WHEN OnlineSecurity = 'Yes' THEN 1 ELSE 0 END +
      CASE WHEN OnlineBackup = 'Yes' THEN 1 ELSE 0 END +
      CASE WHEN DeviceProtection = 'Yes' THEN 1 ELSE 0 END +
      CASE WHEN TechSupport = 'Yes' THEN 1 ELSE 0 END +
      CASE WHEN StreamingTV = 'Yes' THEN 1 ELSE 0 END +
      CASE WHEN StreamingMovies = 'Yes' THEN 1 ELSE 0 END
    ) AS service_count,
    CASE
      WHEN tenure <= 6 THEN '0-6'
      WHEN tenure <= 12 THEN '7-12'
      WHEN tenure <= 24 THEN '13-24'
      WHEN tenure <= 48 THEN '25-48'
      ELSE '49-72'
    END AS tenure_bucket,
    CASE WHEN lower(PaymentMethod) LIKE '%automatic%' THEN 1 ELSE 0 END AS automatic_payment,
    CASE
      WHEN MonthlyCharges <= 35 THEN '<=35'
      WHEN MonthlyCharges <= 70 THEN '35-70'
      WHEN MonthlyCharges <= 90 THEN '70-90'
      ELSE '>90'
    END AS monthly_charge_band
FROM raw_telco;
