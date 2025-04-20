import pandas as pd
import xgboost as xgb
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def train_and_save_model():
    try:
        print("Loading dataset...")
        # Load dataset with error handling
        try:
            df = pd.read_csv("Churn_Modelling_Indian_Names_Ordered.csv")
            print(f"Dataset loaded successfully with {len(df)} rows")
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return False

        # Validate required columns
        required_columns = ['RowNumber', 'CustomerId', 'Name', 'Gender', 'Geography', 
                          'NumOfProducts', 'Exited', 'CreditScore', 'Age', 'Tenure', 
                          'Balance', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False

        print("Preprocessing data...")
        # Drop unnecessary columns
        df = df.drop(['RowNumber', 'CustomerId', 'Name', 'Gender', 'Geography', 'NumOfProducts'], axis=1)
        
        # Check for missing values
        if df.isnull().any().any():
            print("Warning: Dataset contains missing values. Handling them...")
            df = df.fillna(df.mean())
        
        # Split features and target
        X = df.drop('Exited', axis=1)
        y = df['Exited']
        
        print("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("Training XGBoost model...")
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        print("Evaluating model...")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("Saving model...")
        os.makedirs("model", exist_ok=True)
        model_path = os.path.join("model", "xgb_model.pkl")
        joblib.dump(model, model_path)
        
        print(f"✅ Model trained and saved to {model_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error during model training: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting model training process...")
    success = train_and_save_model()
    if not success:
        print("Model training failed. Please check the error messages above.")
    else:
        print("Model training completed successfully!")
