from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model/xgb_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = [
            float(request.form['CreditScore']),
            float(request.form['Age']),
            float(request.form['Tenure']),
            float(request.form['Balance']),
            float(request.form['NumOfProducts']),
            float(request.form['HasCrCard']),
            float(request.form['IsActiveMember']),
            float(request.form['EstimatedSalary']),
        ]
        input_data = np.array(data).reshape(1, -1)
        prediction = model.predict(input_data)

        result = "Churn: Yes" if prediction[0] == 1 else "Churn: No"
        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=f"❌ Error: {e}")

# ✅ This is required to run the app
if __name__ == "__main__":
    app.run(debug=True)
