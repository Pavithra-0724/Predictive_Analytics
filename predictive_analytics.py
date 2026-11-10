
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------------
# 1. Generate Historical Sales Dataset
# -----------------------------------

np.random.seed(42)

dates = pd.date_range(
    start="2022-11-01",
    periods=48,
    freq="MS"
)

trend = np.arange(48)
seasonality = 1000 * np.sin(2 * np.pi * trend / 12)
noise = np.random.normal(0, 350, 48)

sales = 10000 + (trend * 250) + seasonality + noise

df = pd.DataFrame({
    "Date": dates,
    "Sales": sales.round(2)
})

# Introduce a few missing values for preprocessing demonstration
df.loc[[8, 25], "Sales"] = np.nan

df.to_csv("historical_sales.csv", index=False)

print("Historical dataset created successfully!")

# -----------------------------------
# 2. Data Cleaning and Preprocessing
# -----------------------------------

df["Date"] = pd.to_datetime(df["Date"])
df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")

df = df.sort_values("Date")
df["Sales"] = df["Sales"].interpolate(method="linear")

df = df.dropna()

df.to_csv("cleaned_sales.csv", index=False)

print("\nData cleaning completed!")
print(df.head())

# -----------------------------------
# 3. Feature Engineering
# -----------------------------------

df["Time_Index"] = np.arange(len(df))

df["Month_Sin"] = np.sin(
    2 * np.pi * df["Date"].dt.month / 12
)

df["Month_Cos"] = np.cos(
    2 * np.pi * df["Date"].dt.month / 12
)

features = ["Time_Index", "Month_Sin", "Month_Cos"]

X = df[features]
y = df["Sales"]

# -----------------------------------
# 4. Split Historical Data
# -----------------------------------

X_train = X.iloc[:-6]
X_test = X.iloc[-6:]

y_train = y.iloc[:-6]
y_test = y.iloc[-6:]

# -----------------------------------
# 5. Train Regression Model
# -----------------------------------

model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel training completed!")

# -----------------------------------
# 6. Predict and Evaluate
# -----------------------------------

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nMODEL EVALUATION")
print("-------------------------")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

evaluation = f"""
PREDICTIVE ANALYTICS - MODEL EVALUATION

Algorithm: Linear Regression

Mean Absolute Error: {mae:.2f}
Root Mean Squared Error: {rmse:.2f}
R2 Score: {r2:.4f}
"""

with open("model_evaluation.txt", "w") as file:
    file.write(evaluation)

# -----------------------------------
# 7. Actual vs Predicted Visualization
# -----------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    df["Date"].iloc[-6:],
    y_test,
    marker="o",
    label="Actual Sales"
)

plt.plot(
    df["Date"].iloc[-6:],
    predictions,
    marker="s",
    linestyle="--",
    label="Predicted Sales"
)

plt.title("Actual vs Predicted Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("actual_vs_predicted.png", dpi=300)
plt.show()

# -----------------------------------
# 8. Forecast Future Sales
# -----------------------------------

future_dates = pd.date_range(
    start=df["Date"].iloc[-1] + pd.offsets.MonthBegin(1),
    periods=6,
    freq="MS"
)

future_index = np.arange(len(df), len(df) + 6)

future_data = pd.DataFrame({
    "Time_Index": future_index,
    "Month_Sin": np.sin(
        2 * np.pi * future_dates.month / 12
    ),
    "Month_Cos": np.cos(
        2 * np.pi * future_dates.month / 12
    )
})

future_predictions = model.predict(future_data[features])

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted_Sales": future_predictions.round(2)
})

forecast_df.to_csv("future_sales_forecast.csv", index=False)

print("\nFUTURE SALES FORECAST")
print(forecast_df)

# -----------------------------------
# 9. Future Forecast Visualization
# -----------------------------------

plt.figure(figsize=(11, 6))

plt.plot(
    df["Date"],
    df["Sales"],
    label="Historical Sales",
    marker="o"
)

plt.plot(
    future_dates,
    future_predictions,
    label="Future Forecast",
    marker="s",
    linestyle="--"
)

plt.title("Future Sales Forecast Using Linear Regression")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("future_sales_forecast.png", dpi=300)
plt.show()

print("\n================================")
print("PREDICTIVE ANALYTICS COMPLETED!")
print("================================")

print("\nGenerated Files:")
print("1. historical_sales.csv")
print("2. cleaned_sales.csv")
print("3. model_evaluation.txt")
print("4. future_sales_forecast.csv")
print("5. actual_vs_predicted.png")
print("6. future_sales_forecast.png")
