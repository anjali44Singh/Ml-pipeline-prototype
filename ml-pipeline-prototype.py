import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

print("--- STEP 1: Simulating Ingestion & Feature Datasets ---")
np.random.seed(42)
n_samples = 1000

# Simulating multi-source meteorological, soil, and geospatial satellite data
data = {
    'rainfall_mm': np.random.uniform(400, 1200, n_samples),
    'avg_temp_c': np.random.uniform(20, 38, n_samples),
    'soil_ph': np.random.uniform(5.5, 8.0, n_samples),
    'nitrogen_ppm': np.random.uniform(20, 140, n_samples),
    'ndvi_index': np.random.uniform(0.3, 0.9, n_samples)
}

df = pd.DataFrame(data)

# Simulating target crop yield (kg/ha)
df['yield_kg_ha'] = (
    1200 
    + 2.5 * df['rainfall_mm'] 
    - 15.0 * (df['avg_temp_c'] - 25)**2 
    + 80.0 * df['soil_ph'] 
    + 4.0 * df['nitrogen_ppm'] 
    + 1500.0 * df['ndvi_index'] 
    + np.random.normal(0, 100, n_samples)
)

print(f"Dataset generated successfully with shape: {df.shape}\n")

print("--- STEP 2: Data Splitting ---")
X = df[['rainfall_mm', 'avg_temp_c', 'soil_ph', 'nitrogen_ppm', 'ndvi_index']]
y = df['yield_kg_ha']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- STEP 3: Model Training ---")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model training completed.\n")

# Save the trained model
joblib.dump(model, 'crop_yield_model.pkl')
print("Model saved as 'crop_yield_model.pkl'\n")

print("--- STEP 4: Generating & Exporting Test Predictions ---")
predictions = model.predict(X_test)

# Create a dataframe comparing actual yields vs predicted yields
output_results = X_test.copy()
output_results['Actual_Yield_kg_ha'] = y_test
output_results['Predicted_Yield_kg_ha'] = predictions
output_results['Error'] = output_results['Actual_Yield_kg_ha'] - output_results['Predicted_Yield_kg_ha']

# Export predictions to a CSV file in your project folder
output_results.to_csv('pipeline_real_outputs.csv', index=False)
print("Successfully saved real test outputs to 'pipeline_real_outputs.csv'\n")

print("--- STEP 5: Detailed Model Evaluation Metrics ---")
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Absolute Error (MAE): {mae:.2f} kg/ha")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} kg/ha")
print(f"R-squared Score (R²): {r2:.4f}\n")

print("--- STEP 6: Generating Feature Importance Plot ---")
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(8, 4))
plt.barh(features, importances, color='#182B49')
plt.title('Feature Importance - Agricultural Yield Model')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Feature importance chart saved as 'feature_importance.png'\n")

print("--- Sample Predictions (First 5 Rows) ---")
print(output_results[['Actual_Yield_kg_ha', 'Predicted_Yield_kg_ha', 'Error']].head())