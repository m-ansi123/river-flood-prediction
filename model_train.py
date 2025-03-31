import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

# Load dataset
try:
    df = pd.read_csv('river_data.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: river_data.csv not found. Make sure the file is in the correct directory.")
    exit()

# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Normalize features
scaler = MinMaxScaler()
feature_columns = ['rainfall', 'river_level', 'temperature']
if not all(col in df.columns for col in feature_columns):
    print("Error: Missing required feature columns in the dataset.")
    exit()

df[feature_columns] = scaler.fit_transform(df[feature_columns])

# Define input features and target
X = df[feature_columns]
if 'flood_risk' not in df.columns:
    print("Error: 'flood_risk' column missing in dataset.")
    exit()
y = df['flood_risk']  # 0: Low, 1: Medium, 2: High

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, 'flood_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model trained and saved successfully!")
