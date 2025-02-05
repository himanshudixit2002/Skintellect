import sqlite3
from flask import Flask, render_template, request, redirect, session, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

app = Flask(__name__)
app.secret_key = 'your_secret_key'
DATABASE = 'app.db'

# Initialize database and create tables
def create_tables():
    with sqlite3.connect(DATABASE) as connection:
        cursor = connection.cursor()
        
        # Create a schema_version table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS schema_version (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version INTEGER NOT NULL
            )
        ''')

        # Get the current schema version
        cursor.execute('SELECT version FROM schema_version ORDER BY id DESC LIMIT 1')
        current_version = cursor.fetchone()

        # Apply migrations based on schema version
        if not current_version:
            # If no version exists, assume this is the first migration
            current_version = 0
            cursor.execute('INSERT INTO schema_version (version) VALUES (0)')
        else:
            current_version = current_version[0]
        
        # Define migrations
        migrations = [
            '''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'patient'
            )
            ''',
            '''
            CREATE TABLE IF NOT EXISTS appointment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                email TEXT,
                date TEXT,
                skin TEXT,
                phone TEXT,
                age TEXT,
                address TEXT,
                status BOOLEAN,
                username TEXT
            )
            '''
        ]

        # Apply any pending migrations
        for index, migration in enumerate(migrations, start=1):
            if index > current_version:
                cursor.execute(migration)
                cursor.execute('INSERT INTO schema_version (version) VALUES (?)', (index,))
                print(f"Applied migration version {index}")
        
        connection.commit()


def insert_user(username, password, role):
    with sqlite3.connect(DATABASE) as connection:
        cursor = connection.cursor()
        cursor.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", (username, password, role))
        connection.commit()

def get_user(username):
    with sqlite3.connect(DATABASE) as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        return cursor.fetchone()

def role_required(role):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if 'role' not in session or session['role'] != role:
                return "Access Denied!", 403
            return func(*args, **kwargs)
        return wrapper
    return decorator

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']
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
            session['role'] = user[3]
            if user[3] == 'doctor':
                return redirect('/doctor')
            else:
                return redirect('/profile')

        return "Invalid username or password."
    return render_template('login.html')

@app.route('/doctor')
@role_required('doctor')
def doctor_dashboard():
    appointments = []  # Replace with logic to fetch appointments from DB
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

if __name__ == '__main__':
    create_tables()
    app.run(debug=True)
