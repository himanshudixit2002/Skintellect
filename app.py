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
app.config['TEMPLATES_AUTO_RELOAD'] = os.getenv("FLASK_ENV") == "development"
app.secret_key = os.environ["SECRET_KEY"]  
DATABASE = os.environ["DATABASE_URL"]  

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
            password TEXT NOT NULL,
            is_doctor BOOLEAN DEFAULT FALSE
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
        cursor.execute('''CREATE TABLE IF NOT EXISTS skincare_routines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER UNIQUE NOT NULL,
            morning_routine TEXT NOT NULL,
            night_routine TEXT NOT NULL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''')
        connection.commit()
create_tables()

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
        cursor = conn.cursor()
        cursor.execute(
            '''INSERT INTO appointment (name, email, date, skin, phone, age, address, status, username)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (name, email, date, skin, phone, age, address, status, username)
        )
        conn.commit()
        return cursor.lastrowid  # Return the ID of the newly created appointment

def find_appointments(username):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        appointments = cursor.execute("SELECT * FROM appointment WHERE username = ?", (username,)).fetchall()
        return [dict(row) for row in appointments]



def update_appointment_status(appointment_id):
    with get_db_connection() as conn:
        conn.execute("UPDATE appointment SET status = ? WHERE id = ?", (True, appointment_id))
        conn.commit()

def delete_appointment(appointment_id):
    with get_db_connection() as conn:
        conn.execute("DELETE FROM appointment WHERE id = ?", (int(appointment_id,)))
        conn.commit()


# ---------------------------
# Chatbot Endpoint
# ---------------------------
# Route to handle chatbot requests
import dateparser

@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    user_input = data.get("userInput")

    if not user_input:
        return jsonify({"error": "No user input provided."}), 400

    gemini_api_key = os.environ.get("GEMINIE_API_KEY")
    if not gemini_api_key:
        return jsonify({"error": "Gemini API key not configured."}), 500

    # Use session-based conversation tracking
    if "conversation_state" not in session:
        session["conversation_state"] = {}

    conversation_state = session["conversation_state"]

    # If waiting for a date input
    if conversation_state.get("awaiting_date"):
        parsed_date = dateparser.parse(user_input)  # Convert natural language dates
        if parsed_date:
            conversation_state["date"] = parsed_date.strftime("%Y-%m-%d %H:%M")  # Format date
            conversation_state["awaiting_date"] = False
            conversation_state["awaiting_reason"] = True  # Move to the next step
            session.modified = True
            return jsonify({
                "botReply": f"Great! Your appointment is set for {conversation_state['date']}. Now, please describe the reason for your appointment.",
                "type": "appointment_flow"
            })
        else:
            return jsonify({
                "botReply": "I couldn't understand that date format. Please try again with a valid date, such as 'Next Monday at 3 PM' or '10th Feb 2025 at 2 PM'.",
                "type": "error"
            })

    # If waiting for a reason input
    if conversation_state.get("awaiting_reason"):
        reason = user_input
        user = get_user(session.get("username"))
        survey_data = dict(get_survey_response(user["id"]))  # ✅ Convert Row to dictionary

        if not user or not survey_data:
            return jsonify({"botReply": "Please complete your profile survey first.", "type": "error"})

        # Create appointment
        appointment_id = insert_appointment_data(
            name=survey_data["name"],
            email=user["username"],
            date=conversation_state["date"],
            skin=survey_data["skin_type"],
            phone=survey_data.get("phone", ""),  # ✅ Now this works
            age=survey_data["age"],
            address=reason,
            status=False,
            username=user["username"]
        )

        # Reset conversation state
        session["conversation_state"] = {}
        session.modified = True

        return jsonify({
            "botReply": f"Your appointment has been successfully scheduled for {conversation_state['date']} with the reason: {reason}. Your reference ID is APPT-{appointment_id}.",
            "type": "appointment_confirmation",
            "appointmentId": appointment_id
        })

    # If user initiates an appointment
    if "make an appointment" in user_input.lower():
        conversation_state["awaiting_date"] = True
        session.modified = True
        return jsonify({
            "botReply": "When would you like to schedule your appointment? You can type any format (e.g., 'March 10 at 3 PM', 'Tomorrow 4 PM', 'Next Monday').",
            "type": "appointment_flow"
        })

    # Default response from Gemini AI
    payload = {"contents": [{"parts": [{"text": user_input}]}]}
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_api_key}"
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        response_json = response.json()
        bot_reply = response_json.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")

        if not bot_reply:
            return jsonify({"botReply": "I'm having trouble understanding. Could you rephrase that?", "type": "clarification_request"})

        return jsonify({"botReply": bot_reply, "type": "general_response"})

    except requests.exceptions.RequestException as e:
        print("Gemini API error:", e)
        return jsonify({"error": "Error processing request."}), 500


import requests
from langchain import PromptTemplate

def generate_skincare_routine(user_details):
    """
    Uses LangChain to generate a detailed, structured skincare routine by calling Google Gemini API.
    Ensures correct formatting with 10 structured steps for morning and night routines.
    """
    # Define a structured prompt
    prompt_template = """
    Based on the following user skin details, generate a **concise, structured, and formatted** skincare routine:

    - **Age:** {age}
    - **Gender:** {gender}
    - **Skin Type:** {skin_type}
    - **Main Concerns:** {concerns}
    - **Acne Frequency:** {acne_frequency}
    - **Current Skincare Routine:** {skincare_routine}
    - **Stress Level:** {stress_level}

    **Output Format (Strictly follow this format!):**
    
    🌞 **Morning Routine**  
    1. Step 1  
    2. Step 2  
    3. Step 3  
    4. Step 4  
    5. Step 5  
    6. Step 6  
    7. Step 7  

    🌙 **Night Routine**  
    1. Step 1  
    2. Step 2  
    3. Step 3  
    4. Step 4  
    5. Step 5  
    6. Step 6  
    7. Step 7   

    Ensure that:
    - **Each step is numbered properly**.
    - **Use bold headings** without unnecessary asterisks (`**`).
    - **Provide exactly 10 steps per routine**.
    - **Each step should be actionable and easy to follow**.
    """

    # Format the prompt with user details
    prompt = PromptTemplate(
        input_variables=["age", "gender", "skin_type", "concerns", "acne_frequency", "skincare_routine", "stress_level"],
        template=prompt_template
    ).format(
        age=user_details["age"],
        gender=user_details["gender"],
        skin_type=user_details["skin_type"],
        concerns=user_details["concerns"],
        acne_frequency=user_details["acne_frequency"],
        skincare_routine=user_details["skincare_routine"],
        stress_level=user_details["stress_level"]
    )

    # Function to call the Gemini API
    def call_gemini_api(prompt_text):
        headers = {"Content-Type": "application/json"}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINIE_API_KEY}"
        response = requests.post(url, headers=headers, json={"contents": [{"parts": [{"text": prompt_text}]}]})
        if response.status_code == 200:
            data = response.json()
            return data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        else:
            return "Failed to fetch routine from AI"

    # Call Gemini API
    bot_reply = call_gemini_api(prompt)

    # Split morning and night routines correctly
    if "🌙" in bot_reply:
        parts = bot_reply.split("🌙")
        morning_routine = parts[0].strip()
        night_routine = "🌙" + parts[1].strip()
    else:
        routines = bot_reply.split("\n\n")
        morning_routine = routines[0].strip() if routines else "No routine found"
        night_routine = routines[1].strip() if len(routines) > 1 else "No routine found"

    return {"morning_routine": morning_routine, "night_routine": night_routine}


def save_skincare_routine(user_id, morning_routine, night_routine):
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Check if the user already has a routine
        cursor.execute("SELECT id FROM skincare_routines WHERE user_id = ?", (user_id,))
        existing_routine = cursor.fetchone()

        if existing_routine:
            # If exists, UPDATE
            cursor.execute(
                """UPDATE skincare_routines 
                   SET morning_routine = ?, night_routine = ?, last_updated = CURRENT_TIMESTAMP 
                   WHERE user_id = ?""",
                (morning_routine, night_routine, user_id)
            )
        else:
            # If not, INSERT a new routine
            cursor.execute(
                """INSERT INTO skincare_routines (user_id, morning_routine, night_routine) 
                   VALUES (?, ?, ?)""",
                (user_id, morning_routine, night_routine)
            )

        conn.commit()


def get_skincare_routine(user_id):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT morning_routine, night_routine FROM skincare_routines WHERE user_id = ?", (user_id,))
        routine = cursor.fetchone()
    return routine if routine else {"morning_routine": "No routine found", "night_routine": "No routine found"}

@app.route("/generate_routine", methods=["POST"])
def generate_routine():
    if "username" not in session:
        return redirect(url_for("login"))

    user = get_user(session["username"])
    user_details = get_survey_response(user["id"])

    if not user_details:
        return jsonify({"error": "User details not found"})

    routine = generate_skincare_routine(user_details)
    save_skincare_routine(user["id"], routine["morning_routine"], routine["night_routine"])

    return jsonify({"message": "Routine Generated", "routine": routine})


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
        routine = get_skincare_routine(user["id"])  # Fetch AI-generated routine
        if survey_response:
            return render_template("profile.html", survey=survey_response, routine=routine)
    return redirect(url_for("index"))

# ---------------------------
# Appointment Routes
# ---------------------------
@app.route("/bookappointment")
def bookappointment():
    return render_template("bookappointment.html")

@app.route("/appointment", methods=["POST"])
def appointment():
    if "username" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    name = request.form.get("name")
    email = request.form.get("email")
    date = request.form.get("date")
    skin = request.form.get("skin")
    phone = request.form.get("phone")
    age = request.form.get("age")
    address = request.form.get("reason")
    username = session["username"]
    status = False  # Default status is "Pending"

    appointment_id = insert_appointment_data(name, email, date, skin, phone, age, address, status, username)

    return jsonify({"message": "Appointment successfully booked", "appointmentId": appointment_id})




@app.route("/userappointment", methods=["GET"])
def userappoint():
    if "username" not in session:
        return redirect(url_for("login"))
    username = session.get("username")
    appointments = find_appointments(username)
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify({"all_appointments": appointments})
    return render_template("userappointment.html", all_appointments=appointments)



@app.route("/delete_user_request", methods=["POST"])
def delete_user_request():
    data = request.get_json()
    try:
        appointment_id = int(data.get("id"))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid appointment ID"}), 400

    delete_appointment(appointment_id)
    return jsonify({"message": "deleted successfully"})





@app.route("/face_analysis", methods=["POST"])
def face_analysis():
    appointment_id = request.form.get("appointment_id")
    update_appointment_status(appointment_id)
    return jsonify({"message": "updated"})


# ---------------------------
# AI Skin Analysis & Product Recommendation
# ---------------------------
# Initialize the Roboflow skin detection model
rf_skin = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project_skin = rf_skin.workspace().project("skin-detection-pfmbg")
model_skin = project_skin.version(2).model

# Initialize the oiliness detection model
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=os.environ["OILINESS_API_KEY"]
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
    app.run(host="0.0.0.0", port=5000, debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")
