import pickle
import os
import pandas as pd
from flask import Flask, render_template, request

# Initialize Flask app
app = Flask(__name__)

# Set path to model
model_path = os.path.join('model', 'xgb_model.pkl')
model = pickle.load(open(model_path, 'rb'))  # Load model

# Path to the CSV file where history is saved
history_file = 'history.csv'

# Function to save prediction result to CSV
def save_to_csv(data):
   
    if not os.path.exists(history_file):
        data.to_csv(history_file, mode='w', header=True, index=False)
    else:
        data.to_csv(history_file, mode='a', header=False, index=False)
    

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        credit_score = int(request.form['CreditScore'])
        age = int(request.form['Age'])
        tenure = int(request.form['Tenure'])
        balance = float(request.form['Balance'])
        has_cr_card = int(request.form['HasCrCard'])
        is_active_member = int(request.form['IsActiveMember'])
        estimated_salary = float(request.form['EstimatedSalary'])
         
        # Prepare data for prediction
        input_features = [[credit_score, age, tenure, balance, has_cr_card, is_active_member, estimated_salary]]
        print("Input Features:", input_features)
        prediction = model.predict(input_features)

        # Prediction text
        prediction_text = "The customer will churn." if prediction[0] == 1 else "The customer will not churn."

        # Prepare data to save in CSV
        prediction_data = {
            'CreditScore': credit_score,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'HasCrCard': has_cr_card,
            'IsActiveMember': is_active_member,
            'EstimatedSalary': estimated_salary,
            'Prediction': prediction_text
        }
       
        print(prediction_data)
        # Convert prediction data to DataFrame and save
        df = pd.DataFrame([prediction_data])
        save_to_csv(df)

        return render_template('index.html', prediction_text=prediction_text)

    except Exception as e:
        return f"Error: {e}", 400

@app.route('/history')
def view_history():
    if os.path.exists(history_file):
        history_df = pd.read_csv(history_file)
    else:
        history_df = pd.DataFrame()

    history_html = history_df.to_html(classes='table table-bordered', index=False)
    return render_template('history.html', history_html=history_html)

if __name__ == '__main__':
    app.run(debug=True)
