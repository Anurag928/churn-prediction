import pandas as pd
import xgboost as xgb
import pickle
import os
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("Churn_Modelling_Indian_Names_Ordered.csv")

# Drop unnecessary columns
df = df.drop(['RowNumber', 'CustomerId', 'Name', 'Gender', 'Geography'], axis=1)

# Split features and target
X = df.drop('Exited', axis=1)
y = df['Exited']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Save the model
os.makedirs("model", exist_ok=True)
with open("model/xgb_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved to model/xgb_model.pkl")
