import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import os

# ============== 1. Load Data ==============
DATA_PATH = "your_data/houses.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Dataset not found at {DATA_PATH}. Please add your houses.csv file.")

data = pd.read_csv(DATA_PATH)
print("📊 Dataset preview:")
print(data.head())

# ============== 2. Basic Cleaning ==============
data = data.dropna(subset=["price"])  # remove rows with missing target
data.fillna(data.median(numeric_only=True), inplace=True)

# ============== 3. Feature Selection ==============
features = ["sqft", "bedrooms", "bathrooms", "year_built", "distance_to_city"]
target = "price"

if not all(col in data.columns for col in features + [target]):
    raise ValueError("❌ Some feature columns are missing from the dataset!")

X = data[features]
y = data[target]

# ============== 4. Split Data ==============
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============== 5. Scaling ==============
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============== 6. Modeling ==============
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

model.fit(X_train_scaled, y_train)

# ============== 7. Evaluation ==============
y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📈 Evaluation Metrics:")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.2f}")

# ============== 8. Visualization ==============
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.6, color='royalblue')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

# ============== 9. Save Model ==============
os.makedirs("models", exist_ok=True)
joblib.dump((model, scaler), "models/house_price_model.pkl")
print("\n✅ Model saved successfully at 'models/house_price_model.pkl'")

# ============== 10. Predict New House Price ==============
def predict_new_house(model, scaler, sqft, bedrooms, bathrooms, year_built, distance):
    """Predict the price of a new house given its features."""
    new_data = pd.DataFrame([{
        "sqft": sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "year_built": year_built,
        "distance_to_city": distance
    }])
    new_scaled = scaler.transform(new_data)
    price = model.predict(new_scaled)[0]
    return price

# Example: predict new house
example_price = predict_new_house(model, scaler, sqft=2000, bedrooms=4, bathrooms=3, year_built=2012, distance=5)
print(f"\n💰 Predicted price for the new house: ${example_price:,.2f}")
