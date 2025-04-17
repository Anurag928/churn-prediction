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
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Set secret key - for production, use environment variable
app.secret_key = os.environ.get('FLASK_SECRET_KEY')
if not app.secret_key:
    # Generate a new key if environment variable is not set
    app.secret_key = secrets.token_hex(24)

# Set path to model
model_path = os.path.join('model', 'xgb_model.pkl')
try:
    if os.path.exists(model_path):
        model = pickle.load(open(model_path, 'rb'))  # Load model
    else:
        print(f"Error: Model file not found at {model_path}")
        model = None
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

# Path to the CSV file where history is saved
history_file = 'history.csv'

def init_db():
    db_path = os.environ.get('DATABASE_URL', 'predictions.db')
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # Create predictions table
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  credit_score INTEGER,
                  age INTEGER,
                  tenure INTEGER,
                  balance REAL,
                  has_credit_card INTEGER,
                  is_active_member INTEGER,
                  estimated_salary REAL,
                  result TEXT,
                  date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # Create users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def home():
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    return render_template('index.html', username=current_user.username)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        
        conn = sqlite3.connect('predictions.db')
        c = conn.cursor()
        
        try:
            hashed_password = generate_password_hash(password)
            c.execute('INSERT INTO users (username, password, email) VALUES (?, ?, ?)',
                     (username, hashed_password, email))
            conn.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists!', 'error')
        finally:
            conn.close()
            
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('predictions.db')
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = c.fetchone()
        conn.close()
        
        if user and check_password_hash(user[2], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            return redirect(url_for('predictor'))
        else:
            flash('Invalid username or password!', 'error')
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@login_required
@app.route('/predictor')
def predictor():
    return render_template('index.html', username=session.get('username'))

@login_required
@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in [
            request.form['CreditScore'],
            request.form['Age'],
            request.form['Tenure'],
            request.form['Balance'],
            request.form['HasCrCard'],
            request.form['IsActiveMember'],
            request.form['EstimatedSalary']
        ]]
        
        prediction = model.predict([features])
        result = "Will Churn" if prediction[0] == 1 else "Will Stay"
        
        conn = sqlite3.connect('predictions.db')
        c = conn.cursor()
        c.execute('''INSERT INTO predictions 
                     (user_id, credit_score, age, tenure, balance, 
                      has_credit_card, is_active_member, estimated_salary, result)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                 (session['user_id'], int(features[0]), int(features[1]), 
                  int(features[2]), features[3], int(features[4]), 
                  int(features[5]), features[6], result))
        conn.commit()
        conn.close()
        
        return render_template('index.html', 
                             prediction_text=f'Prediction: {result}',
                             username=session.get('username'))
    
    except Exception as e:
        return render_template('index.html', 
                             prediction_text=f'Error: {str(e)}',
                             username=session.get('username'))

@login_required
@app.route('/history')
def history():
    try:
        conn = sqlite3.connect('predictions.db')
        c = conn.cursor()
        c.execute('''SELECT credit_score, age, balance, result, 
                    datetime(date, 'localtime') as date
                    FROM predictions 
                    WHERE user_id = ?
                    ORDER BY date DESC LIMIT 10''', (session['user_id'],))
        
        predictions = [
            {
                'credit_score': row[0],
                'age': row[1],
                'balance': row[2],
                'result': row[3],
                'date': row[4]
            }
            for row in c.fetchall()
        ]
        
        conn.close()
        return render_template('history.html', 
                             predictions=predictions,
                             username=session.get('username'))
    
    except Exception as e:
        return render_template('history.html', 
                             error=str(e),
                             username=session.get('username'))

# Create all database tables
with app.app_context():
    db.create_all()

# Add this after your database initialization
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

if __name__ == '__main__':
    app.run(debug=True)
