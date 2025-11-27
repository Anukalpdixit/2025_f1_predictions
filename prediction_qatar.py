import fastf1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Enable FastF1 caching (creates / uses a local cache folder)
fastf1.Cache.enable_cache("f1_cache")

# ------------------ 1. Historical race data ------------------
# 2024 Qatar GP race session (the most recent completed race at this circuit)
session_2024 = fastf1.get_session(2024, "Qatar", "R")
session_2024.load()

# Extract lap times and convert to seconds
laps_2024 = session_2024.laps[["Driver", "LapTime"]].copy()
laps_2024.dropna(subset=["LapTime"], inplace=True)
laps_2024["LapTime (s)"] = laps_2024["LapTime"].dt.total_seconds()

# ------------------ 2. 2025 Qatar qualifying data ------------------
# NOTE: Replace the placeholder qualifying times with the actual 2025 qualifying results when available
qualifying_2025 = pd.DataFrame({
    "Driver": [
        "Max Verstappen", "Lando Norris", "Oscar Piastri", "Charles Leclerc",
        "George Russell", "Lewis Hamilton", "Yuki Tsunoda", "Carlos Sainz",
        "Fernando Alonso", "Alexander Albon", "Pierre Gasly", "Lance Stroll"
    ],
    "QualifyingTime (s)": [
        96.215, 96.333, 96.410, 96.532,
        96.611, 96.720, 96.845, 96.870,
        97.024, 97.055, 97.077, 97.380
    ]
})

# Map full names to FastF1 3-letter driver codes
DRIVER_MAP = {
    "Max Verstappen": "VER", "Lando Norris": "NOR", "Oscar Piastri": "PIA",
    "Charles Leclerc": "LEC", "George Russell": "RUS", "Lewis Hamilton": "HAM",
    "Yuki Tsunoda": "TSU", "Carlos Sainz": "SAI", "Fernando Alonso": "ALO",
    "Alexander Albon": "ALB", "Pierre Gasly": "GAS", "Lance Stroll": "STR"
}
qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(DRIVER_MAP)

# ------------------ 3. Merge datasets ------------------
# Use the average lap time from the 2024 race as the target we are trying to predict
avg_lap_by_driver = laps_2024.groupby("Driver")["LapTime (s)"].mean().reset_index()
merged = qualifying_2025.merge(avg_lap_by_driver, left_on="DriverCode", right_on="Driver", how="left")

# Feature matrix and target vector
X = merged[["QualifyingTime (s)"]]
y = merged["LapTime (s)"]

# Guard against empty merge (e.g., driver codes not matching)
if X.empty:
    raise ValueError("Merged dataset is empty. Check driver codes or data availability.")

# ------------------ 4. Train/test split & model fit ------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = GradientBoostingRegressor(n_estimators=120, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

print(f"Validation MAE: {mean_absolute_error(y_test, model.predict(X_test)):.2f} s")

# ------------------ 5. Predict 2025 race times ------------------
qualifying_2025["PredictedRaceTime (s)"] = model.predict(qualifying_2025[["QualifyingTime (s)"]])
results = qualifying_2025.sort_values("PredictedRaceTime (s)")[["Driver", "PredictedRaceTime (s)"]]

print("\n🏁 Predicted 2025 Qatar GP Outcome 🏁\n")
print(results.to_string(index=False))
