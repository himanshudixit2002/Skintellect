import sqlite3
from flask import Flask, jsonify, render_template, request, redirect, session, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from roboflow import Roboflow
import json
import os
import uuid  
import cv2
import pandas as pd
from joblib import load

df = pd.read_csv(r"dataset/updated_skincare_products.csv")

app = Flask(__name__)
app.secret_key = '4545'
DATABASE = 'app.db'

def create_tables():
    with sqlite3.connect(DATABASE) as connection:
        cursor = connection.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                          id INTEGER PRIMARY KEY AUTOINCREMENT,
                          username TEXT UNIQUE NOT NULL,
                          password TEXT NOT NULL,
                          role TEXT NOT NULL)''')  # Added role column

        cursor.execute('''CREATE TABLE IF NOT EXISTS appointment( 
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       name TEXT ,
                       email TEXT ,
                       date TEXT, 
                       skin TEXT,
                       phone TEXT,
                       age TEXT,
                       address TEXT, 
                       status BOOLEAN,
                       username TEXT)''')

def insert_user(username, password, role):
    with sqlite3.connect(DATABASE) as connection:
        cursor = connection.cursor()
        cursor.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", 
                       (username, password, role))
        connection.commit()

def get_user(username):
    with sqlite3.connect(DATABASE) as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        return cursor.fetchone()

def insert_appointment_data(name, email, date, skin, phone, age, address, status, username):
    with sqlite3.connect(DATABASE) as connection:
        cursor = connection.cursor()
        cursor.execute('''INSERT INTO appointment (name, email, date, skin, phone, age, address, status, username)
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                          (name, email, date, skin, phone, age, address, status, username))
        connection.commit()

def findallappointment():
    with sqlite3.connect(DATABASE) as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM appointment")
        return cursor.fetchall()

def update_appointment_status(appointment_id):
    with sqlite3.connect(DATABASE) as connection:
        cursor = connection.cursor()
        cursor.execute("UPDATE appointment SET status = ? WHERE id = ?", (True, appointment_id))
        connection.commit()

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = get_user(username)

        if user and check_password_hash(user[2], password):
            session['username'] = username
            session['role'] = user[3]  

            if user[3] == "doctor":  # Check role
                return redirect(url_for('allappoint'))
            else:
                return redirect(url_for('profile'))
        return "Invalid username or password"

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']  # Allow role selection
        hashed_password = generate_password_hash(password)

        if get_user(username):
            return "Username already exists."

        insert_user(username, hashed_password, role)
        return redirect('/')

    return render_template('register.html')

@app.route('/doctor')
def doctor():
    if 'username' in session and session.get('role') == "doctor":
        return render_template('doctor.html')
    return redirect('/')

@app.route('/allappointments')
def allappoint():
    if 'username' in session and session.get('role') == "doctor":
        all_appointments = findallappointment()
        return render_template('doctor.html', appointments=json.dumps(all_appointments))
    return redirect('/')

@app.route("/update_status", methods=["POST"])
def update_status():
    if 'username' in session and session.get('role') == "doctor":
        appointment_id = request.form.get("appointment_id")
        update_appointment_status(appointment_id)
        return "updated"
    return "Unauthorized", 403

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

if __name__ == '__main__':
    create_tables()
    app.run(debug=True)
