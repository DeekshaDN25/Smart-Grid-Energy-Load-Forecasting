# Smart Grid Energy Load Forecasting - Python
# Step 1: Load Data
# Step 2: Train Linear Regression Model
# Step 3: Predict and Evaluate
# Step 4: Visualize Results

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ── 1. LOAD DATA ──────────────────────────────────────────────────
# If CSV from MATLAB exists, load it. Otherwise generate sample data.
import os
if os.path.exists("smart_grid_data.csv"):
    df = pd.read_csv("smart_grid_data.csv")
    print("Loaded data from MATLAB CSV")
else:
    np.random.seed(42)
    n = 100
    hours = np.arange(1, n+1)
    temperature  = 25 + 10*np.sin(2*np.pi*hours/24) + 2*np.random.randn(n)
    energy_load  = 200 + 5*(temperature-20) + 30*np.sin(2*np.pi*hours/24) + 10*np.random.randn(n)
    df = pd.DataFrame({"hours": hours, "temperature": temperature, "energy_load": energy_load})
    print("Generated sample data")

print(df.head())

# ── 2. PREPARE FEATURES ──────────────────────────────────────────
X = df[["temperature", "hours"]]   # Features
y = df["energy_load"]              # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── 3. TRAIN LINEAR REGRESSION ───────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ── 4. EVALUATE ──────────────────────────────────────────────────
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"\nModel Results:")
print(f"  RMSE : {rmse:.2f} MW")
print(f"  R²   : {r2:.4f}")

# ── 5. VISUALIZE ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Smart Grid Energy Load Forecasting", fontsize=13, fontweight="bold")

# Plot 1: Temperature vs Load
axes[0].scatter(df["temperature"], df["energy_load"], color="steelblue", alpha=0.6)
axes[0].set_title("Temperature vs Energy Load")
axes[0].set_xlabel("Temperature (°C)")
axes[0].set_ylabel("Load (MW)")
axes[0].grid(True)

# Plot 2: Actual vs Predicted
axes[1].scatter(y_test, y_pred, color="orange", alpha=0.7)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1].set_title("Actual vs Predicted Load")
axes[1].set_xlabel("Actual Load (MW)")
axes[1].set_ylabel("Predicted Load (MW)")
axes[1].grid(True)

# Plot 3: Load trend over time
axes[2].plot(df["hours"], df["energy_load"], label="Actual",    color="blue",   lw=1.5)
axes[2].plot(X_test["hours"], y_pred,        label="Predicted", color="red",    lw=1.5, linestyle="--", marker="o", ms=4)
axes[2].set_title(f"Load Forecast  |  R²={r2:.3f}")
axes[2].set_xlabel("Hour")
axes[2].set_ylabel("Load (MW)")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig("smart_grid_result.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved → smart_grid_result.png")
