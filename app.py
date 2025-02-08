import os
import sqlite3
import uuid
import cv2
import pandas as pd
import requests
import traceback

from flask import Flask, render_template, request, redirect, session, url_for, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from roboflow import Roboflow
import supervision as sv
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# Load environment variables from .env file
load_dotenv()
GEMINIE_API_KEY = os.getenv("GEMINIE_API_KEY")
if not GEMINIE_API_KEY:
    raise Exception("GEMINIE_API_KEY not set. Please add it to your .env file.")

# Initialize Flask application
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.secret_key = os.getenv("SECRET_KEY", "4545")
DATABASE = 'app.db'

# Load the skincare products dataset (ensure the path is correct)
df = pd.read_csv(os.path.join("dataset", "updated_skincare_products.csv"))

# ---------------------------
# Database Functions
# ---------------------------
def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row  # enable accessing columns by name
    return conn

def create_tables():
    with get_db_connection() as connection:
        cursor = connection.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS survey_responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            age TEXT NOT NULL,
            gender TEXT NOT NULL,
            concerns TEXT NOT NULL,
            acne_frequency TEXT NOT NULL,
            comedones_count TEXT NOT NULL,
            first_concern TEXT NOT NULL,
            cosmetic_usage TEXT NOT NULL,
            skin_reaction TEXT NOT NULL,
            skin_type TEXT NOT NULL,
            medications TEXT NOT NULL,
            skincare_routine TEXT NOT NULL,
            stress_level TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS appointment (
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
        )''')
        connection.commit()

def insert_user(username, password):
    with get_db_connection() as conn:
        conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()

def get_user(username):
    with get_db_connection() as conn:
        user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    return user

def insert_survey_response(user_id, name, age, gender, concerns, acne_frequency, comedones_count, first_concern, cosmetics_usage, skin_reaction, skin_type, medications, skincare_routine, stress_level):
    with get_db_connection() as conn:
        conn.execute(
            """INSERT INTO survey_responses 
            (user_id, name, age, gender, concerns, acne_frequency, comedones_count, first_concern, cosmetic_usage, skin_reaction, skin_type, medications, skincare_routine, stress_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, name, age, gender, concerns, acne_frequency, comedones_count, first_concern, cosmetics_usage, skin_reaction, skin_type, medications, skincare_routine, stress_level)
        )
        conn.commit()

def get_survey_response(user_id):
    with get_db_connection() as conn:
        response = conn.execute("SELECT * FROM survey_responses WHERE user_id = ?", (user_id,)).fetchone()
    return response

def insert_appointment_data(name, email, date, skin, phone, age, address, status, username):
    with get_db_connection() as conn:
        conn.execute(
            '''INSERT INTO appointment (name, email, date, skin, phone, age, address, status, username)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (name, email, date, skin, phone, age, address, status, username)
        )
        conn.commit()

def findappointment(username):
    with get_db_connection() as conn:
        appointments = conn.execute("SELECT * FROM appointment WHERE username = ?", (username,)).fetchall()
    return appointments

def findallappointment():
    with get_db_connection() as conn:
        appointments = conn.execute("SELECT * FROM appointment").fetchall()
    return appointments

def update_appointment_status(appointment_id):
    with get_db_connection() as conn:
        conn.execute("UPDATE appointment SET status = ? WHERE id = ?", (True, appointment_id))
        conn.commit()

def delete_appointment(appointment_id):
    with get_db_connection() as conn:
        conn.execute("DELETE FROM appointment WHERE id = ?", (appointment_id,))
        conn.commit()

# Create tables at startup
create_tables()

# ---------------------------
# Chatbot Endpoint
# ---------------------------
# Route to handle chatbot requests
@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    user_input = data.get("userInput")

    if not user_input:
        return jsonify({"error": "No user input provided."}), 400

    gemini_api_key = os.environ.get("GEMINIE_API_KEY")
    if not gemini_api_key:
        return jsonify({"error": "Gemini API key not configured."}), 500

    # Construct the payload in Gemini's format
    payload = {
        "contents": [{
            "parts": [{"text": user_input}]
        }]
    }

    # Correct Gemini API endpoint
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_api_key}"

    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        response_json = response.json()

        # Extract the chatbot's response from the API response
        bot_reply = response_json.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "I'm sorry, I didn't understand that.")

        return jsonify({"botReply": bot_reply})
    
    except requests.exceptions.RequestException as e:
        print("Gemini API error:", e)
        return jsonify({"error": "Error processing request."}), 500

# ---------------------------
# User Authentication & Survey Routes
# ---------------------------
@app.route("/")
def index():
    # If the user is not logged in, redirect to the login page.
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        name = request.form.get("name", "")
        age = request.form.get("age", "")
        hashed_password = generate_password_hash(password)
        if get_user(username):
            return render_template("register.html", error="Username already exists.")
        insert_user(username, hashed_password)
        session["username"] = username
        session["name"] = name
        session["age"] = age
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = get_user(username)
        if user and check_password_hash(user["password"], password):
            session["username"] = username
            # Special redirect for the doctor account
            if username == "doctor1":
                return redirect(url_for("allappointments"))
            survey_response = get_survey_response(user["id"])
            if survey_response:
                return redirect(url_for("profile"))
            else:
                return redirect(url_for("survey"))
        return render_template("login.html", error="Invalid username or password")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/survey", methods=["GET", "POST"])
def survey():
    if "username" not in session:
        return redirect(url_for("index"))
    if request.method == "POST":
        user = get_user(session["username"])
        user_id = user["id"]
        name = session.get("name", "")
        age = session.get("age", "")
        gender = request.form["gender"]
        concerns = ",".join(request.form.getlist("concerns"))
        acne_frequency = request.form["acne_frequency"]
        comedones_count = request.form["comedones_count"]
        first_concern = request.form["first_concern"]
        cosmetics_usage = request.form["cosmetics_usage"]
        skin_reaction = request.form["skin_reaction"]
        skin_type = request.form["skin_type_details"]
        medications = request.form["medications"]
        skincare_routine = request.form["skincare_routine"]
        stress_level = request.form["stress_level"]
        insert_survey_response(user_id, name, age, gender, concerns, acne_frequency, comedones_count,
                               first_concern, cosmetics_usage, skin_reaction, skin_type,
                               medications, skincare_routine, stress_level)
        return redirect(url_for("profile"))
    return render_template("survey.html", name=session.get("name", ""), age=session.get("age", ""))

@app.route("/profile")
def profile():
    if "username" in session:
        user = get_user(session["username"])
        survey_response = get_survey_response(user["id"])
        if survey_response:
            return render_template("profile.html", survey=survey_response)
    return redirect(url_for("index"))

# ---------------------------
# Appointment Routes
# ---------------------------
@app.route("/bookappointment")
def bookappointment():
    return render_template("bookappointment.html")

@app.route("/appointment", methods=["POST"])
def appointment():
    # Retrieve fields from the form.
    name = request.form.get("name")
    email = request.form.get("email")
    date = request.form.get("date")
    skin = request.form.get("skin")
    phone = request.form.get("phone")
    age = request.form.get("age")
    # Use the "reason" field (since that's what your consultation form sends)
    address = request.form.get("reason")  # Alternatively, rename the form field to "address"
    username = session.get("username")
    status = False
    insert_appointment_data(name, email, date, skin, phone, age, address, status, username)
    
    # Redirect to the user appointments page so the user can see their appointment
    return redirect(url_for("userappoint"))


@app.route("/allappointments")
def allappointments():
    appointments = findallappointment()
    return render_template("doctor.html", appointments=appointments)

@app.route("/userappointment")
def userappoint():
    username = session.get("username")
    appointments = findappointment(username)
    return render_template("userappointment.html", all_appointments=appointments)

@app.route("/face_analysis", methods=["POST"])
def face_analysis():
    appointment_id = request.form.get("appointment_id")
    update_appointment_status(appointment_id)
    return jsonify({"message": "updated"})

@app.route("/delete_user_request", methods=["POST"])
def delete_user_request():
    appointment_id = request.form.get("id")
    delete_appointment(appointment_id)
    return jsonify({"message": "deleted successfully"})

# ---------------------------
# AI Skin Analysis & Product Recommendation
# ---------------------------
# Initialize the Roboflow skin detection model
rf_skin = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY", "9dsh7sHI8YmicqwPkdd2"))
project_skin = rf_skin.workspace().project("skin-detection-pfmbg")
model_skin = project_skin.version(2).model

# Initialize the oiliness detection model
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=os.getenv("OILINESS_API_KEY", "Gqf1hrF7jdAh8EsbOoTM")
)
# Mapping to convert oiliness model class names if needed
class_mapping = {
    "Jenis Kulit Wajah - v6 2023-06-17 11-53am": "oily skin",
    "-": "normal/dry skin"
}

def recommend_products_based_on_classes(classes):
    recommendations = []
    df_columns_lower = [col.lower() for col in df.columns]
    for skin_condition in classes:
        condition_lower = skin_condition.lower()
        if condition_lower in df_columns_lower:
            original_column = df.columns[df_columns_lower.index(condition_lower)]
            filtered = df[df[original_column] == 1][["Brand", "Name", "Price", "Ingredients"]]
            filtered["Ingredients"] = filtered["Ingredients"].apply(lambda x: ", ".join(x.split(", ")[:5]))
            products = filtered.head(5).to_dict(orient="records")
            recommendations.append((skin_condition, products))
    return recommendations

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            image_file = request.files["image"]
            image_filename = str(uuid.uuid4()) + ".jpg"
            image_path = os.path.join("static", image_filename)
            image_file.save(image_path)
            unique_classes = set()
            # Skin detection using Roboflow
            skin_result = model_skin.predict(image_path, confidence=15, overlap=30).json()
            skin_labels = [pred["class"] for pred in skin_result.get("predictions", [])]
            unique_classes.update(skin_labels)
            # Oiliness detection with custom configuration
            custom_configuration = InferenceConfiguration(confidence_threshold=0.3)
            with CLIENT.use_configuration(custom_configuration):
                oiliness_result = CLIENT.infer(image_path, model_id="oilyness-detection-kgsxz/1")
            if not oiliness_result.get("predictions"):
                unique_classes.add("dryness")
            else:
                oiliness_classes = [class_mapping.get(pred["class"], pred["class"]) for pred in oiliness_result.get("predictions", []) if pred.get("confidence", 0) >= 0.3]
                unique_classes.update(oiliness_classes)
            # Annotate image using Supervision
            image = cv2.imread(image_path)
            detections = sv.Detections.from_inference(skin_result)
            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()
            annotated_image = box_annotator.annotate(scene=image, detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
            cv2.imwrite(os.path.join("static", "annotations_0.jpg"), annotated_image)
            # Get product recommendations
            recommended_products = recommend_products_based_on_classes(list(unique_classes))
            prediction_data = {
                "classes": list(unique_classes),
                "recommendations": recommended_products
            }
            return render_template("face_analysis.html", data=prediction_data)
        except Exception as e:
            traceback.print_exc()
            return "An error occurred during analysis.", 500
    return render_template("face_analysis.html", data={})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
