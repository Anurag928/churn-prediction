import pickle
import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, flash
from datetime import datetime
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import secrets
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Set secret key from environment variable, fall back to generated key if not found
app.secret_key = os.environ.get('FLASK_SECRET_KEY')
if not app.secret_key:
    # Generate a new key if environment variable is not set
    generated_key = secrets.token_hex(24)
    app.secret_key = generated_key
    # Optionally warn about using a generated key
    print("Warning: Using generated secret key. Set FLASK_SECRET_KEY environment variable for production use.")

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
    conn = sqlite3.connect('predictions.db')
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

# Login required decorator
def login_required(f):
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('predictor'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
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

if __name__ == '__main__':
    app.run(debug=True)
