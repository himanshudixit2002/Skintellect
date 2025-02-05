import sqlite3
from flask import Flask, render_template, request, redirect, session, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

app = Flask(__name__)
app.secret_key = '4545'
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
                role TEXT NOT NULL CHECK(role IN ('patient', 'doctor'))
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
                status BOOLEAN DEFAULT 0,
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
    return render_template('doctor.html')


@app.route('/profile')
@role_required('patient')
def profile():
    return render_template('profile.html', name=session['username'])


@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('role', None)
    return redirect('/')


if __name__ == '__main__':
    create_tables()
    app.run(debug=True)
