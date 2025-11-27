import fastf1
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

"""
Prediction script for Qatar 2025 when qualifying times are NOT yet available.

Strategy:
1. Use 2024 Qatar GP average lap time as the ground-truth target.
2. Use driver season-to-date performance (points) and recent race pace at the previous
   2025 event (Monza chosen as example) as proxies for current form.
3. Train a small regression model to map those proxies to expected Qatar pace.
4. Predict and rank drivers for the 2025 Qatar race.

Update the `SEASON_POINTS_2025` mapping whenever new points are available and
change `LAST_RACE_NAME` once a more recent 2025 race has been completed.
"""

# -------------------------- CONFIG --------------------------
YEAR_TARGET_RACE = 2024          # year of the historical Qatar GP to train on
LAST_RACE_NAME = "Monza"         # most recent 2025 race with lap data
LAST_RACE_YEAR = 2025
TARGET_CIRCUIT_NAME = "Qatar"

# Season-to-date driver points (placeholder – update with real values)
SEASON_POINTS_2025 = {
    "VER": 314, "NOR": 242, "LEC": 215, "HAM": 199, "PIA": 162,
    "RUS": 155, "SAI": 140, "ALO": 82,  "TSU": 54,  "ALB": 33,
    "GAS": 27,  "OCO": 22,  "STR": 20,  "HUL": 19,
}

# ------------------------------------------------------------
fastf1.Cache.enable_cache("f1_cache")

# 1. Target variable: avg lap time at 2024 Qatar GP
session_qatar_2024 = fastf1.get_session(YEAR_TARGET_RACE, TARGET_CIRCUIT_NAME, "R")
session_qatar_2024.load()
laps_qatar_24 = session_qatar_2024.laps[["Driver", "LapTime"]].dropna()
laps_qatar_24["LapTime (s)"] = laps_qatar_24["LapTime"].dt.total_seconds()
qatar24_avg = laps_qatar_24.groupby("Driver")["LapTime (s)"].mean().reset_index()

# 2. Recent form feature: avg lap time at last completed 2025 race (e.g. Monza)
session_last = fastf1.get_session(LAST_RACE_YEAR, LAST_RACE_NAME, "R")
session_last.load()
laps_last = session_last.laps[["Driver", "LapTime"]].dropna()
laps_last["RecentLap (s)"] = laps_last["LapTime"].dt.total_seconds()
recent_avg = laps_last.groupby("Driver")["RecentLap (s)"].mean().reset_index()

# 3. Assemble feature frame
features = qatar24_avg.merge(recent_avg, on="Driver", how="left")
features["SeasonPoints"] = features["Driver"].map(SEASON_POINTS_2025)

# Drop drivers lacking recent data or season points
features = features.dropna(subset=["RecentLap (s)", "SeasonPoints"])

X = features[["RecentLap (s)", "SeasonPoints"]]
y = features["LapTime (s)"]

# 4. Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)
model = GradientBoostingRegressor(random_state=7)
model.fit(X_train, y_train)
print(f"Validation MAE: {mean_absolute_error(y_test, model.predict(X_test)):.2f} s")

# 5. Predict Qatar 2025
features["PredictedRaceTime (s)"] = model.predict(X)
forecast = features.sort_values("PredictedRaceTime (s)")[["Driver", "PredictedRaceTime (s)"]]
print("\n🏁 Predicted 2025 Qatar GP (No Quali Data) 🏁\n")
print(forecast.to_string(index=False))
