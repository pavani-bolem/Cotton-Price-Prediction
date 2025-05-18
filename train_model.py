import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv('processed_cotton_prices.csv')

# Convert date to numerical format
df['arrival_date'] = pd.to_datetime(df['arrival_date'])
df['days_since_start'] = (df['arrival_date'] - df['arrival_date'].min()).dt.days

# Encode categorical features
label_encoders = {}
for col in ['state', 'district', 'market']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features & target
X = df[['state', 'district', 'market', 'days_since_start']]
y = df['modal_price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save model, scaler, and encoders
with open('cotton_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("Model trained and saved successfully!")

from sklearn.metrics import r2_score

# Predict on test set
y_pred = model.predict(X_test_scaled)

# Calculate R² score (accuracy)
accuracy = r2_score(y_test, y_pred)
print(f"Model Accuracy (R² Score): {accuracy:.4f}")
