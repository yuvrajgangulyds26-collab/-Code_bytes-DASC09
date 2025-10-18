from flask import Flask, request, jsonify, send_from_directory, render_template, session, redirect, url_for
from flask_cors import CORS
import joblib
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import re  # for email validation

app = Flask(__name__)
CORS(app)
app.secret_key = "your_random_secret_key_here"

# Load ML model
model = joblib.load("cab_fare_rf_coords_model.pkl")
DB_PATH = 'users.db'

# -----------------------
# Database helpers
# -----------------------
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create users and history tables if they don't exist"""
    conn = get_db_connection()
    c = conn.cursor()
    
    # Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    ''')
    
    # History table
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            pickup_lat REAL NOT NULL,
            pickup_lng REAL NOT NULL,
            dropoff_lat REAL NOT NULL,
            dropoff_lng REAL NOT NULL,
            pickup TEXT,
            dropoff TEXT,
            distance REAL,
            duration REAL,
            fare REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );
    ''')
    
    conn.commit()
    conn.close()

init_db()

# -----------------------
# Static & HTML routes
# -----------------------
@app.route('/<path:filename>')
def serve_file(filename):
    return send_from_directory('.', filename)

@app.route('/')
def index_page():
    if 'user_id' in session:
        return redirect(url_for('dashboard_page'))
    return render_template('index.html')

@app.route('/login')
def login_page():
    if 'user_id' in session:
        return redirect(url_for('dashboard_page'))
    return render_template('login.html')

@app.route('/signup')
def signup_page():
    if 'user_id' in session:
        return redirect(url_for('dashboard_page'))
    return render_template('signup.html')

@app.route('/dashboard')
def dashboard_page():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    return render_template('dashboard.html', user_name=session.get('user_name'))

@app.route('/history')
def history_page():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    return render_template('history.html', user_name=session.get('user_name'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login_page'))

# -----------------------
# Signup/Login APIs
# -----------------------
@app.route('/api/signup', methods=['POST'])
def api_signup():
    data = request.json
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')

    if not name or not email or not password:
        return jsonify({'error': 'All fields are required'}), 400

    # Validate email
    email_regex = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    if not re.match(email_regex, email):
        return jsonify({'error': 'Invalid email address'}), 400

    hashed_password = generate_password_hash(password)

    conn = get_db_connection()
    try:
        cursor = conn.execute(
            "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
            (name, email, hashed_password)
        )
        conn.commit()
        user_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Email already exists'}), 400
    finally:
        conn.close()

    # Log in user
    session['user_id'] = user_id
    session['user_name'] = name

    return jsonify({'message': 'Signup successful'}), 200

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'error': 'All fields are required'}), 400

    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()

    if user and check_password_hash(user['password'], password):
        session['user_id'] = user['id']
        session['user_name'] = user['name']
        return jsonify({'message': 'Login successful'}), 200
    else:
        return jsonify({'error': 'Invalid email or password'}), 401

# -----------------------
# Ride History API
# -----------------------
@app.route('/api/history', methods=['GET'])
def api_history():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 403

    conn = get_db_connection()
    rows = conn.execute('''
        SELECT pickup_lat, pickup_lng, dropoff_lat, dropoff_lng,
               pickup, dropoff, distance, duration, fare, created_at
        FROM history
        WHERE user_id = ?
        ORDER BY created_at DESC
    ''', (session['user_id'],)).fetchall()
    conn.close()

    history = []
    for row in rows:
        history.append({
            'pickup_lat': row['pickup_lat'],
            'pickup_lng': row['pickup_lng'],
            'dropoff_lat': row['dropoff_lat'],
            'dropoff_lng': row['dropoff_lng'],
            'pickup': row['pickup'] or '-',
            'dropoff': row['dropoff'] or '-',
            'trip_distance': row['distance'] or 0,
            'trip_duration_min': row['duration'] or 0,
            'predicted_fare': row['fare'] or 0,
            'created_at': row['created_at']
        })
    return jsonify(history)

# -----------------------
# Fare Prediction API
# -----------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized. Please log in first.'}), 401

        data = request.get_json()
        required_fields = [
            'trip_distance', 'trip_duration_min',
            'pickup_longitude', 'pickup_latitude',
            'dropoff_longitude', 'dropoff_latitude',
            'pickup','dropoff',
            'hour_of_day', 'is_weekend_or_holiday'
        ]
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Extract fields
        trip_distance = float(data['trip_distance'])
        trip_duration_min = float(data['trip_duration_min'])
        pickup_lat = float(data['pickup_latitude'])
        pickup_lng = float(data['pickup_longitude'])
        dropoff_lat = float(data['dropoff_latitude'])
        dropoff_lng = float(data['dropoff_longitude'])
        pickup = data.get('pickup', 'Unknown')
        dropoff = data.get('dropoff', 'Unknown')
        hour_of_day = int(data['hour_of_day'])
        is_weekend_or_holiday = int(data['is_weekend_or_holiday'])

        # Validations
        if trip_distance < 1:
            return jsonify({'error': 'Trip distance too short. Must be ≥ 1 km.'}), 400
        if trip_distance > 50:
            return jsonify({'error': 'Trip distance too long. Maximum 50 km.'}), 400
        if trip_duration_min < 2:
            return jsonify({'error': 'Trip duration too short.'}), 400
        if trip_duration_min > 360:
            return jsonify({'error': 'Trip duration too long (>6 hours).'}), 400

        # Prepare features
        features = [[
            trip_distance,
            trip_duration_min,
            pickup_lng,
            pickup_lat,
            dropoff_lng,
            dropoff_lat,
            hour_of_day,
            is_weekend_or_holiday
        ]]

        # Predict fare
        fare = float(model.predict(features)[0])
        fare = max(2.0, round(fare, 2))

        # Save to history
        user_id = session['user_id']
        conn = get_db_connection()
        conn.execute('''
            INSERT INTO history (
                user_id, pickup_lat, pickup_lng, dropoff_lat, dropoff_lng,
                pickup, dropoff, distance, duration, fare, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        ''', (user_id, pickup_lat, pickup_lng, dropoff_lat, dropoff_lng,
              pickup, dropoff, trip_distance, trip_duration_min, fare))
        conn.commit()
        conn.close()

        return jsonify({'success': True, 'predicted_fare': fare}), 200

    except ValueError:
        return jsonify({'error': 'Invalid input type. Ensure numeric fields are numbers.'}), 400

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({'error': 'Internal server error.', 'details': str(e)}), 500

# -----------------------
# Run app
# -----------------------
if __name__ == '__main__':
    app.run(debug=True)
