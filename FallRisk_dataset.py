import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000

# Simulate features
age = np.random.randint(65, 95, size=n_samples)
walking_speed = np.clip(np.random.normal(loc=1.0, scale=0.3, size=n_samples), 0.2, 2.0)  # m/s
missed_meals = np.random.poisson(lam=1.5, size=n_samples)  # meals missed per week
sleep_hours = np.clip(np.random.normal(loc=6.5, scale=1.5, size=n_samples), 3, 10)
med_adherence = np.random.binomial(1, 0.8, size=n_samples)  # 1 = adherent
past_falls = np.random.poisson(lam=0.7, size=n_samples)
mobility_aid = np.random.binomial(1, 0.4, size=n_samples)  # 1 = uses aid

# Create fall risk using rule-based logic
risk_score = (
    (age > 80).astype(int) +
    (walking_speed < 0.8).astype(int) +
    (missed_meals > 2).astype(int) +
    (sleep_hours < 5.5).astype(int) +
    (med_adherence == 0).astype(int) +
    (past_falls > 1).astype(int) +
    (mobility_aid == 1).astype(int)
)

# Label generation based on risk score
fall_risk = (risk_score >= 3).astype(int)

# Combine into a DataFrame
df = pd.DataFrame({
    "Age": age,
    "WalkingSpeed": walking_speed,
    "MissedMeals": missed_meals,
    "SleepHours": sleep_hours,
    "MedicationAdherence": med_adherence,
    "PastFalls": past_falls,
    "MobilityAid": mobility_aid,
    "FallRisk": fall_risk
})

# Preview
print(df.head())
