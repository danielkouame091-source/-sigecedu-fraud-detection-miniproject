import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1. Load the dataset (replace 'sigecedu_data.csv' with your actual CSV file name if needed)
df = pd.read_csv('sigecedu_data.csv')
print("Dataset loaded successfully!")

# 2. Prepare features and target
# Replace 'target_column' with the exact name of the column you want to predict
X = df.drop(['target_column'], axis=1)
y = df['target_column']

# 3. Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

# 6. Evaluate the model
y_pred = model.predict(X_test_scaled)

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# 7. Save the model
joblib.dump(model, 'sigecedu_fraud_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nModel and scaler successfully saved!")
