# ================================
# Rising Waters: Flood Prediction
# Complete Machine Learning Script
# ================================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2. Load Dataset
# Dataset should contain:
# rainfall, river_level, temperature, humidity, flood (0/1)

data = pd.read_csv("flood_data.csv")

# 3. Data Preprocessing
data = data.dropna()

X = data.drop("flood", axis=1)
y = data["flood"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Model Evaluation
y_pred = model.predict(X_test)

print("\n===== Model Evaluation =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 6. Feature Importance Visualization
importances = model.feature_importances_
features = X.columns

plt.figure()
plt.bar(features, importances)
plt.xticks(rotation=45)
plt.title("Feature Importance for Flood Prediction")
plt.show()

# 7. Predict New Data
# Example: rainfall(mm), river_level(m), temperature(C), humidity(%)
new_data = np.array([[200, 5.2, 30, 85]])
new_data_scaled = scaler.transform(new_data)

prediction = model.predict(new_data_scaled)

print("\n===== New Prediction =====")
if prediction[0] == 1:
    print("⚠️ Flood Likely")
else:
    print("✅ No Flood Risk")
