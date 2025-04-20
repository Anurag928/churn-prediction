import pickle
import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from datetime import datetime
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import secrets
from xgboost import XGBClassifier
import joblib
from flask_sqlalchemy import SQLAlchemy

# Initialize Flask app
app = Flask(__name__, static_url_path='/static', static_folder='static')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy with a workaround for compatibility issues
db = SQLAlchemy()
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Set secret key - for production, use environment variable
app.secret_key = os.environ.get('FLASK_SECRET_KEY')
if not app.secret_key:
    # Generate a new key if environment variable is not set
    app.secret_key = secrets.token_hex(24)

# Define User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    credit_score = db.Column(db.Integer, nullable=False)
    age = db.Column(db.Integer, nullable=False)
    tenure = db.Column(db.Integer, nullable=False)
    balance = db.Column(db.Float, nullable=False)
    has_credit_card = db.Column(db.Integer, nullable=False)
    is_active_member = db.Column(db.Integer, nullable=False)
    estimated_salary = db.Column(db.Float, nullable=False)
    result = db.Column(db.String(20), nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)

# Create all database tables
with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Set path to model
model_path = os.path.join('model', 'xgb_model.pkl')
model = None

def load_model():
    global model
    try:
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            model = joblib.load(model_path)
            print("Model loaded successfully")
        else:
            print(f"Error: Model file not found at {model_path}")
            model = None
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        model = None

# Load the model when the application starts
load_model()

@app.route('/')
def home():
    if current_user.is_authenticated:
        return render_template('index.html', username=current_user.username)
    return redirect(url_for('register'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        
        # Check if username or email already exists
        existing_user = User.query.filter((User.username == username) | (User.email == email)).first()
        if existing_user:
            if existing_user.username == username:
                flash('Username already exists. Please choose a different username.', 'error')
            else:
                flash('Email already registered. Please use a different email.', 'error')
            return render_template('register.html')
        
        try:
            hashed_password = generate_password_hash(password)
            user = User(username=username, password=hashed_password, email=email)
            db.session.add(user)
            db.session.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash('An error occurred during registration. Please try again.', 'error')
            return render_template('register.html')
            
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password!', 'error')
            
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/predictor')
@login_required
def predictor():
    return render_template('index.html', username=current_user.username)

# @app.route('/predict', methods=['POST'])
# @login_required
# def predict():
#     if model is None:
#         load_model()  # Try to reload the model
#         if model is None:
#             flash('Error: Model not loaded. Please try again later.', 'error')
#             return redirect(url_for('home'))
    
#     try:
#         # Get form data
#         required_fields = ['CreditScore', 'Age', 'Tenure', 'Balance', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
#         features = []
#         for field in required_fields:
#             value = request.form.get(field, '').strip()
#             if not value:
#                 flash(f'Please fill in the {field} field.', 'error')
#                 return redirect(url_for('home'))
#             try:
#                 features.append(float(value))
#             except ValueError:
#                 flash(f'Invalid input for {field}. Please enter a valid number.', 'error')
#                 return redirect(url_for('home'))
        
#         # Make prediction using DataFrame with correct feature names
#         import pandas as pd
#         feature_names = ['CreditScore', 'Age', 'Tenure', 'Balance', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
#         input_df = pd.DataFrame([features], columns=feature_names)
#         prediction = model.predict(input_df)
#         result = "Customer Will Churn" if prediction[0] == 1 else "Customer Will Stay"
        
#         # Save prediction to database
#         prediction_record = Prediction(
#             user_id=current_user.id,
#             credit_score=int(features[0]),
#             age=int(features[1]),
#             tenure=int(features[2]),
#             balance=features[3],
#             has_credit_card=int(features[4]),
#             is_active_member=int(features[5]),
#             estimated_salary=features[6],
#             result=result
#         )
        
#         db.session.add(prediction_record)
#         db.session.commit()
        
#         # Show success message with prediction
#         # flash(f'Prediction Result: {result}', 'success')
#         # return redirect(url_for('history'))

#         prediction_text = f'Prediction Result: {result}'
#         return render_template('index.html', prediction_text=prediction_text, username=current_user.username)
#     except ValueError as e:
#         flash(f'Input error: {str(e)}', 'error')
#         return redirect(url_for('home'))
#     except Exception as e:
#         db.session.rollback()
#         flash(f'Error making prediction: {str(e)}', 'error')
#         return redirect(url_for('home'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if model is None:
        load_model()  # Try to reload the model
        if model is None:
            flash('Error: Model not loaded. Please try again later.', 'error')
            return redirect(url_for('home'))
    
    try:
        # Get form data
        required_fields = ['CreditScore', 'Age', 'Tenure', 'Balance', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
        features = []
        for field in required_fields:
            value = request.form.get(field, '').strip()
            if not value:
                flash(f'Please fill in the {field} field.', 'error')
                return redirect(url_for('home'))
            try:
                features.append(float(value))
            except ValueError:
                flash(f'Invalid input for {field}. Please enter a valid number.', 'error')
                return redirect(url_for('home'))

        # Make prediction using DataFrame with correct feature names
        feature_names = ['CreditScore', 'Age', 'Tenure', 'Balance', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
        input_df = pd.DataFrame([features], columns=feature_names)
        prediction = model.predict(input_df)
        result = "Customer Will Churn" if prediction[0] == 1 else "Customer Will Stay"

        # Save prediction to database
        prediction_record = Prediction(
            user_id=current_user.id,
            credit_score=int(features[0]),
            age=int(features[1]),
            tenure=int(features[2]),
            balance=features[3],
            has_credit_card=int(features[4]),
            is_active_member=int(features[5]),
            estimated_salary=features[6],
            result=result
        )

        db.session.add(prediction_record)
        db.session.commit()

        # Save prediction to CSV
        #csv_file = os.path.join('history.csv')
        csv_file = os.path.join(os.getcwd(), 'history.csv')
        prediction_data = {
            'Date': prediction_record.date.strftime('%Y-%m-%d %H:%M:%S'),
            'Username': current_user.username,
            'Credit Score': prediction_record.credit_score,
            'Age': prediction_record.age,
            'Tenure': prediction_record.tenure,
            'Balance': prediction_record.balance,
            'Has Credit Card': 'Yes' if prediction_record.has_credit_card else 'No',
            'Is Active Member': 'Yes' if prediction_record.is_active_member else 'No',
            'Estimated Salary': prediction_record.estimated_salary,
            'Result': prediction_record.result
        }

        file_exists = os.path.isfile(csv_file)
        try:
            with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
                df = pd.DataFrame([prediction_data])
                df.to_csv(file, index=False, header=not file_exists)
        except Exception as e:
            flash(f"Failed to save history to CSV: {e}", 'error')

        # Show prediction on the same page
        prediction_text = f'Prediction Result: {result}'
        return render_template('index.html', prediction_text=prediction_text, username=current_user.username)

    except ValueError as e:
        flash(f'Input error: {str(e)}', 'error')
        return redirect(url_for('home'))
    except Exception as e:
        db.session.rollback()
        flash(f'Error making prediction: {str(e)}', 'error')
        return redirect(url_for('home'))

@app.route('/history')
@login_required
def history():
    try:
        predictions = Prediction.query.filter_by(user_id=current_user.id)\
            .order_by(Prediction.date.desc())\
            .limit(10)\
            .all()
        
        # Prepare data for Jinja table rendering
        prediction_list = [
            {
                'date': pred.date.strftime('%Y-%m-%d %H:%M:%S'),
                'credit_score': pred.credit_score,
                'age': pred.age,
                'tenure': pred.tenure,
                'balance': f"{pred.balance:.2f}",
                'has_credit_card': 'Yes' if pred.has_credit_card else 'No',
                'is_active_member': 'Yes' if pred.is_active_member else 'No',
                'estimated_salary': f"{pred.estimated_salary:.2f}",
                'result': pred.result
            }
            for pred in reversed(predictions)
        ]
        return render_template('history.html', 
                             predictions=prediction_list,
                             username=current_user.username)

    
    except Exception as e:
        flash(f'Error loading history: {str(e)}', 'error')
        return render_template('history.html', 
                             username=current_user.username)

if __name__ == '__main__':
    app.run(debug=True)
