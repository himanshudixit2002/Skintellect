import sqlite3
from flask import Flask, render_template, request, redirect, session, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

import os
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default-secret-key')
DATABASE = 'app.db'


# Create Database Tables
def create_tables():
    with sqlite3.connect(DATABASE) as connection:
        cursor = connection.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('patient', 'doctor')) DEFAULT 'patient'
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS appointment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                email TEXT,
                date TEXT,
                skin TEXT,
                phone TEXT,
                age TEXT,
                address TEXT,
                status INTEGER DEFAULT 0,  -- 0=Pending, 1=Approved, 2=Rejected
                username TEXT
            )
        ''')


def get_user(username):
    with sqlite3.connect(DATABASE) as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        return cursor.fetchone()


def insert_user(username, password, role):
    with sqlite3.connect(DATABASE) as connection:
        cursor = connection.cursor()
        cursor.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", (username, password, role))
        connection.commit()
        
# Middleware for Role-Based Authentication
def role_required(role):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if 'role' not in session or session['role'] != role:
                return "Access Denied!", 403
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Routes

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']  # Get role from form (doctor/patient)
        hashed_password = generate_password_hash(password)

        if get_user(username):
            return "Username already exists. Please choose another."

        insert_user(username, hashed_password, role)
        return redirect('/')

    return render_template('register.html')


@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = get_user(username)
        if user and check_password_hash(user[2], password):
            session['username'] = username
            session['role'] = user[3]  # Store role in session

            if user[3] == 'doctor':
                return redirect('/doctor')
            else:
                return redirect('/profile')

        return "Invalid username or password."
    
    return render_template('login.html')


@app.route('/doctor')
@role_required('doctor')
def doctor_dashboard():
    with sqlite3.connect(DATABASE) as connection:
        connection.row_factory = sqlite3.Row
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM appointment WHERE status = 0")
        appointments = [dict(row) for row in cursor.fetchall()]
    return render_template('doctor.html', appointments=appointments)


@app.route('/profile')
@role_required('patient')
def profile():
    return render_template('profile.html', name=session['username'])


@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('role', None)
    return redirect('/')

@app.route('/appointment/<int:appointment_id>', methods=['PUT'])
@role_required('doctor')
def update_appointment(appointment_id):
    data = request.get_json()
    new_status = 1 if data.get('status') == 1 else 2  # 1=Approved, 2=Rejected
    
    with sqlite3.connect(DATABASE) as connection:
        cursor = connection.cursor()
        cursor.execute("UPDATE appointment SET status = ? WHERE id = ?",
                     (new_status, appointment_id))
        connection.commit()
    
    return {'success': True}

if __name__ == '__main__':
    create_tables()
    app.run(debug=True)
