import os
import sqlite3
import uuid
import cv2
import pandas as pd
import requests
import traceback
import google.generativeai as genai
from flask import Flask, render_template, request, redirect, session, url_for, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from roboflow import Roboflow
import supervision as sv
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
import dateparser
from werkzeug.utils import secure_filename
import markdown
from langchain_core.prompts import PromptTemplate

# =============================================================================
# Environment & API Configuration
# =============================================================================

# Load environment variables from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise Exception("GEMINI_API_KEY not set. Please add it to your .env file.")

# Configure Gemini API for google.generativeai
genai.configure(api_key=GEMINI_API_KEY)

# =============================================================================
# Initialize Flask Application
# =============================================================================

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = os.getenv("FLASK_ENV") == "development"
app.secret_key = os.environ["SECRET_KEY"]
DATABASE = os.environ["DATABASE_URL"]

# =============================================================================
# Load Data & Models
# =============================================================================

# Load the skincare products dataset
df = pd.read_csv(os.path.join("dataset", "updated_skincare_products.csv"))

# Initialize Roboflow skin detection model
rf_skin = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project_skin = rf_skin.workspace().project("skin-detection-pfmbg")
model_skin = project_skin.version(2).model

# Initialize Oiliness Detection Client
CLIENT = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=os.environ["OILINESS_API_KEY"])
chat_prompt_template = PromptTemplate(
    input_variables=["user_input"],
    template="""
You are a helpful, friendly, and engaging AI assistant specialized in skincare advice.
Respond in a clear and structured way using Markdown. Use emojis and ask follow-up questions to keep the conversation interactive.

**User's Message:**
{user_input}

**Response:**
"""
)

# ---------------------------------------------------------------------------
# Helper Function: Generate Gemini Response
# ---------------------------------------------------------------------------
def get_gemini_response(user_input):
    """
    Generates an AI response using the Gemini API with the LangChain prompt.
    Converts the resulting Markdown into HTML for proper display.
    """
    # Format the prompt using LangChain
    formatted_prompt = chat_prompt_template.format(user_input=user_input)
    
    # Call Gemini API using google.generativeai's GenerativeModel (adjust as needed)
    response = genai.GenerativeModel("gemini-1.5-flash").generate_content(formatted_prompt)
    
    if response and response.text:
        # Convert the AI's Markdown output to HTML
        formatted_response = markdown.markdown(response.text.strip())
        return formatted_response
    else:
        return "⚠️ AI response not available."

# ---------------------------------------------------------------------------
# Chatbot Endpoint (Single definition)
# ---------------------------------------------------------------------------
def langchain_summarize(text, max_length, min_length):
    """
    Summarize the provided text using LangChain prompt templating and the Gemini API.
    """
    prompt_template = """
    Hey! You're a super-friendly **skincare expert** who gives **simple, practical, and fun** advice. 
    Your goal is to make skincare feel **easy, relatable, and stress-free**. Keep your answers **casual, engaging, and beginner-friendly**—like you're chatting with a friend! 😊

    Here’s what we need help with:

   🧴 **Skin Type:** {skin_type}  
   🌿 **Concerns:** {concerns}  
   📆 **Current Routine:** {skincare_routine}  
   🔎 **Main Issue:** {main_issue}  
    Now, based on this, **give clear, practical skincare advice** that includes:  
    - A **simple morning & night routine** (don't overcomplicate it!)  
    - Best **ingredients to look for** (and what to avoid)  
    - Some **product recommendations** (if needed)  
    - **Fun, relatable skincare tips** (keep it engaging!)  
    - If needed, **a myth-busting fact** about skincare 🚀  
    
    Here’s the text to summarize:
    {text}

    Please summarize the above text to have a maximum of {max_length} words and a minimum of {min_length} words.

    💡 **Important:**  
    - Keep it **short, casual, and easy to follow.** No jargon, no fancy science—just **real** advice!  
    - Talk **like a human, not a robot.** Add some **warmth and personality!**  
    - If the issue is serious, **politely suggest seeing a dermatologist.**  

    Go ahead and drop some **amazing** skincare tips! ✨  
    Please strictly format your response using Markdown (use '#' for headings, '-' for lists, '**' for bold, etc.), and ensure it is well-structured.
    """
   
    prompt = PromptTemplate(
        input_variables=["text", "max_length", "min_length"],
        template=prompt_template
    ).format(text=text, max_length=max_length, min_length=min_length)
    
    headers = {"Content-Type": "application/json"}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    response = requests.post(url, headers=headers, json={"contents": [{"parts": [{"text": prompt}]}]})
    
    if response.status_code == 200:
        data = response.json()
        summary_text = (
            data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
        )
        return summary_text.strip()
    else:
        return "Failed to summarize text."

# -----------------------------------------------------------------------------
# Database Functions
# -----------------------------------------------------------------------------

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row  # Enable accessing columns by name
    return conn

def create_tables():
    with get_db_connection() as connection:
        cursor = connection.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_doctor  INTEGER DEFAULT 0
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
            status INTEGER DEFAULT 0,
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
        cursor.execute('''CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            message TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )''')

        connection.commit()

create_tables()

def insert_user(username, password, is_doctor=False):
    with get_db_connection() as conn:
        conn.execute("INSERT INTO users (username, password, is_doctor) VALUES (?, ?, ?)",
                     (username, password, int(is_doctor)))
        conn.commit()

def get_user(username):
    with get_db_connection() as conn:
        return conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()

def insert_survey_response(user_id, name, age, gender, concerns, acne_frequency, comedones_count,
                           first_concern, cosmetics_usage, skin_reaction, skin_type, medications,
                           skincare_routine, stress_level):
    with get_db_connection() as conn:
        conn.execute(
            """INSERT INTO survey_responses 
            (user_id, name, age, gender, concerns, acne_frequency, comedones_count, first_concern, cosmetic_usage,
             skin_reaction, skin_type, medications, skincare_routine, stress_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, name, age, gender, concerns, acne_frequency, comedones_count, first_concern,
             cosmetics_usage, skin_reaction, skin_type, medications, skincare_routine, stress_level)
        )
        conn.commit()

def get_survey_response(user_id):
    with get_db_connection() as conn:
        return conn.execute("SELECT * FROM survey_responses WHERE user_id = ?", (user_id,)).fetchone()

def insert_appointment_data(name, email, date, skin, phone, age, address, status, username):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            '''INSERT INTO appointment (name, email, date, skin, phone, age, address, status, username)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (name, email, date, skin, phone, age, address, status, username)
        )
        conn.commit()
        return cursor.lastrowid

def find_appointments(username):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        appointments = cursor.execute("SELECT * FROM appointment WHERE username = ?", (username,)).fetchall()
        return [dict(row) for row in appointments]

def update_appointment_status(appointment_id, status):
    with get_db_connection() as conn:
        conn.execute("UPDATE appointment SET status = ? WHERE id = ?", (status, appointment_id))
        conn.commit()

def delete_appointment(appointment_id):
    with get_db_connection() as conn:
        conn.execute("DELETE FROM appointment WHERE id = ?", (int(appointment_id),))
        conn.commit()

# -----------------------------------------------------------------------------
# AI Helper Functions
# -----------------------------------------------------------------------------

def get_gemini_recommendations(skin_conditions):
    if not skin_conditions:
        return "No skin conditions detected for analysis."

    prompt = f"""
    You are an AI skincare expert. A user uploaded an image, and the detected skin conditions are: {', '.join(skin_conditions)}.

    - Explain these skin conditions in simple terms.
    - List the best skincare ingredients for these conditions.
    - Provide a basic morning and night skincare routine.
    - Suggest 3 skincare products with pros & cons.
    - Give 2 lifestyle tips for better skin health.

    Keep the response concise and well-structured.
    Please respond in an engaging and interactive manner using Markdown. Use a warm, friendly tone, include emojis
    """

    headers = {"Content-Type": "application/json"}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    response = requests.post(url, headers=headers, json={"contents": [{"parts": [{"text": prompt}]}]})
    
    print("Gemini API response status:", response.status_code)  # Debug log
    print("Gemini API raw response:", response.json())  # Debug log

    if response.status_code == 200:
        data = response.json()
        summary_text = (
            data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
        )
        return summary_text.strip()
    else:
        return "Failed to summarize text."


def recommend_products_based_on_classes(classes):
    """
    Generate product recommendations based on detected skin condition classes.
    """
    recommendations = []
    df_columns_lower = [col.lower() for col in df.columns]
    USD_TO_INR = 83  # Currency conversion rate

    for skin_condition in classes:
        condition_lower = skin_condition.lower()
        if condition_lower in df_columns_lower:
            original_column = df.columns[df_columns_lower.index(condition_lower)]
            filtered = df[df[original_column] == 1][["Brand", "Name", "Price", "Ingredients"]]

            def convert_price(price):
                try:
                    return round(float(price) * USD_TO_INR, 2)
                except ValueError:
                    return "N/A"
            filtered["Price"] = filtered["Price"].apply(convert_price)
            filtered["Ingredients"] = filtered["Ingredients"].apply(
                lambda x: ", ".join(x.split(", ")[:5])
            )
            products = filtered.head(5).to_dict(orient="records")
        else:
            products = []

        ai_analysis = get_gemini_recommendations([skin_condition])
        recommendations.append({
            "condition": skin_condition,
            "products": products,
            "ai_analysis": ai_analysis
        })
    return recommendations

def generate_skincare_routine(user_details):
    """
    Generate a structured skincare routine based on user details using LangChain and Gemini API.
    """
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
    - Each step is numbered properly.
    - Use bold headings without unnecessary asterisks.
    - Provide exactly 7 steps per routine.
    - Each step should be actionable and easy to follow.
    Please strictly format your response using Markdown (use '#' for headings, '-' for lists, '**' for bold, etc.), and ensure it is well-structured.
    """
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

    def call_gemini_api(prompt_text):
        headers = {"Content-Type": "application/json"}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
        response = requests.post(url, headers=headers, json={"contents": [{"parts": [{"text": prompt_text}]}]})
        if response.status_code == 200:
            data = response.json()
            return data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        else:
            return "Failed to fetch routine from AI"

    bot_reply = call_gemini_api(prompt)
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
        cursor.execute("SELECT id FROM skincare_routines WHERE user_id = ?", (user_id,))
        existing_routine = cursor.fetchone()
        if existing_routine:
            cursor.execute(
                """UPDATE skincare_routines 
                   SET morning_routine = ?, night_routine = ?, last_updated = CURRENT_TIMESTAMP 
                   WHERE user_id = ?""",
                (morning_routine, night_routine, user_id)
            )
        else:
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

# =============================================================================
# Flask Routes
# =============================================================================

# -----------------------------------------------------------------------------
# Chatbot Endpoint
# -----------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Helper Function to Build Conversation Prompt with History
# ------------------------------------------------------------------------------
def build_conversation_prompt(history, user_input):
    """
    Build a prompt that includes previous conversation history and the new user input.
    This helps the assistant generate a response in context.
    """
    prompt = """You are a helpful, friendly skincare assistant. Keep your tone casual and conversational.
Respond in an engaging and interactive manner using Markdown. Use a warm, friendly tone, include emojis.

Here is the conversation so far:\n"""
    for msg in history:
        role = msg.get("role")
        text = msg.get("text")
        # Capitalize the role (User/Assistant) for clarity
        prompt += f"{role.capitalize()}: {text}\n"
    prompt += f"User: {user_input}\nAssistant:"
    return prompt

# ------------------------------------------------------------------------------
# Function to Complete an Incomplete Answer (As Before)
# ------------------------------------------------------------------------------
def complete_answer_if_incomplete(answer):
    """
    Check if the answer ends with proper punctuation.
    If not, ask Gemini to continue the answer so that it conveys the full meaning.
    """
    answer = answer.strip()
    if not answer or answer[-1] not in ".!?":
        continuation_prompt = (
            "It looks like your previous answer might have been cut off. "
            "Please continue the answer in the same style and provide a complete, coherent response. "
            "Here is what you have so far:\n\n"
            f"{answer}\n\nContinue:"
        )
        headers = {"Content-Type": "application/json"}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
        response = requests.post(url, headers=headers, json={"contents": [{"parts": [{"text": continuation_prompt}]}]})
        if response.status_code == 200:
            data = response.json()
            continuation = (
                data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
            )
            full_answer = answer + " " + continuation.strip()
            return full_answer
        else:
            return answer
    else:
        return answer

# ------------------------------------------------------------------------------
# Chatbot Endpoint (Updated to Maintain Conversation History)
# ------------------------------------------------------------------------------

@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    user_input = data.get("userInput")
    if not user_input:
        return jsonify({"error": "No user input provided."}), 400

    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        return jsonify({"error": "Gemini API key not configured."}), 500

    # Initialize conversation history in session if not present
    if "conversation_history" not in session:
        session["conversation_history"] = []
    conversation_history = session["conversation_history"]

    # Initialize conversation_state in session if not present
    if "conversation_state" not in session:
        session["conversation_state"] = {}
    conversation_state = session["conversation_state"]

    # Appointment scheduling flow (unchanged)
    if conversation_state.get("awaiting_date"):
        parsed_date = dateparser.parse(user_input)
        if parsed_date:
            conversation_state["date"] = parsed_date.strftime("%Y-%m-%d %H:%M")
            conversation_state["awaiting_date"] = False
            conversation_state["awaiting_reason"] = True
            session.modified = True
            conversation_history.append({"role": "user", "text": user_input})
            conversation_history.append({"role": "assistant", "text": f"Great! Your appointment is set for {conversation_state['date']}. Now, please describe the reason for your appointment."})
            session["conversation_history"] = conversation_history
            return jsonify({
                "botReply": f"Great! Your appointment is set for {conversation_state['date']}. Now, please describe the reason for your appointment.",
                "type": "appointment_flow"
            })
        else:
            return jsonify({
                "botReply": "I couldn't understand that date format. Please try again with a valid date (e.g., 'Next Monday at 3 PM').",
                "type": "error"
            })

    if conversation_state.get("awaiting_reason"):
        reason = user_input
        user = get_user(session.get("username"))
        survey_data = dict(get_survey_response(user["id"]))
        if not user or not survey_data:
            return jsonify({"botReply": "Please complete your profile survey first.", "type": "error"})
        appointment_id = insert_appointment_data(
            name=survey_data["name"],
            email=user["username"],
            date=conversation_state["date"],
            skin=survey_data["skin_type"],
            phone=survey_data.get("phone", ""),
            age=survey_data["age"],
            address=reason,
            status=False,
            username=user["username"]
        )
        conversation_history.append({"role": "user", "text": user_input})
        conversation_history.append({"role": "assistant", "text": f"Your appointment has been successfully scheduled for {conversation_state['date']} with the reason: {reason}. Your reference ID is APPT-{appointment_id}."})
        session["conversation_history"] = conversation_history
        session["conversation_state"] = {}
        session.modified = True
        return jsonify({
            "botReply": f"Your appointment has been successfully scheduled for {conversation_state['date']} with the reason: {reason}. Your reference ID is APPT-{appointment_id}.",
            "type": "appointment_confirmation",
            "appointmentId": appointment_id
        })

    if "make an appointment" in user_input.lower():
        conversation_state["awaiting_date"] = True
        session.modified = True
        conversation_history.append({"role": "user", "text": user_input})
        conversation_history.append({"role": "assistant", "text": "When would you like to schedule your appointment? (e.g., 'March 10 at 3 PM')"})
        session["conversation_history"] = conversation_history
        return jsonify({
            "botReply": "When would you like to schedule your appointment? (e.g., 'March 10 at 3 PM')",
            "type": "appointment_flow"
        })

    # General queries – update conversation history and build the prompt
    conversation_history.append({"role": "user", "text": user_input})
    prompt = build_conversation_prompt(conversation_history, user_input)

    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_api_key}"
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        response_json = response.json()
        bot_reply = (
            response_json.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )

        # Ensure the answer is complete
        bot_reply = complete_answer_if_incomplete(bot_reply)

        # Append assistant reply to conversation history and update session
        conversation_history.append({"role": "assistant", "text": bot_reply})
        session["conversation_history"] = conversation_history

        if not bot_reply:
            return jsonify({"botReply": "I'm having trouble understanding. Could you rephrase that?", "type": "clarification_request"})
        return jsonify({"botReply": bot_reply, "type": "general_response"})

    except requests.exceptions.RequestException as e:
        print("Gemini API error:", e)
        return jsonify({"error": "Error processing request."}), 500

# -----------------------------------------------------------------------------
# Skincare Routine Generation Endpoint
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# User Authentication & Survey Routes
# -----------------------------------------------------------------------------
@app.route("/")
def index():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        is_doctor = True if request.form.get("is_doctor") == "on" else False
        name = request.form.get("name", "")
        age = request.form.get("age", "")
        hashed_password = generate_password_hash(password)
        if get_user(username):
            return render_template("register.html", error="Username already exists.")
        insert_user(username, hashed_password,is_doctor)
        session["username"] = username
        session["name"] = name
        session["age"] = age
        return redirect(url_for("login"))
    return render_template("register.html")
@app.route("/login.html")
def login_html_redirect():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = get_user(username)
        if user and check_password_hash(user["password"], password):
            session["username"] = username
            # Correct doctor check using user["is_doctor"]
            if user["is_doctor"] == 1:
                session["is_doctor"] = True
                return redirect(url_for("doctor_dashboard"))
            
            # Regular user flow
            session["is_doctor"] = False
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
        routine = get_skincare_routine(user["id"])
        if survey_response:
            return render_template("profile.html", survey=survey_response, routine=routine)
    return redirect(url_for("index"))

# -----------------------------------------------------------------------------
# Appointment Routes
# -----------------------------------------------------------------------------
from flask import abort
@app.route("/appointment/<int:appointment_id>")
def appointment_detail(appointment_id):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        appointment = cursor.execute("SELECT * FROM appointment WHERE id = ?", (appointment_id,)).fetchone()
        if not appointment:
            abort(404)
    return render_template("appointment_detail.html", appointment=appointment)

@app.route("/update_appointment", methods=["POST"])
def update_appointment():
    # Ensure that only logged-in doctors can update appointment status
    if "username" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    user = get_user(session["username"])
    if not user or user["is_doctor"] != 1:
        return jsonify({"error": "Access denied"}), 403

    data = request.get_json()
    appointment_id = data.get("appointment_id")
    action = data.get("action")  # Expected to be "confirm" or "reject"
    if not appointment_id or not action:
        return jsonify({"error": "Missing appointment id or action."}), 400

    # Set status: 1 for confirm, 2 for reject
    if action == "confirm":
        status = 1
    elif action == "reject":
        status = 2
    else:
        return jsonify({"error": "Invalid action."}), 400

    try:
        update_appointment_status(appointment_id, status)
        return jsonify({"message": f"Appointment {appointment_id} updated successfully.", "new_status": status})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/delete_appointment", methods=["POST"])
def delete_appointment_route():
    # Ensure the user is logged in.
    if "username" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    try:
        appointment_id = int(data.get("id"))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid appointment ID"}), 400

    # Check if the appointment exists and belongs to the logged-in user.
    with get_db_connection() as conn:
        cursor = conn.cursor()
        appointment = cursor.execute("SELECT * FROM appointment WHERE id = ?", (appointment_id,)).fetchone()
        if not appointment:
            return jsonify({"error": "Appointment not found."}), 404
        if appointment["username"] != session["username"]:
            return jsonify({"error": "You do not have permission to delete this appointment."}), 403

    # Delete the appointment.
    delete_appointment(appointment_id)
    return jsonify({"message": "Appointment deleted successfully."})


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
    status = False
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
    if "username" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    user = get_user(session["username"])
    # Only allow doctors to update appointment status
    if not user or user["is_doctor"] != 1:
        return jsonify({"error": "Access denied"}), 403
    appointment_id = request.form.get("appointment_id")
    update_appointment_status(appointment_id)
    return jsonify({"message": "updated"})

def get_all_appointments():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        appointments = cursor.execute("SELECT * FROM appointment").fetchall()
        return [dict(row) for row in appointments]

@app.route("/doctor_dashboard")
def doctor_dashboard():
    # Check if a user is logged in
    if "username" not in session:
        return redirect(url_for("login"))
    user = get_user(session["username"])
    # Ensure that only doctors can access this route
    if not user or user["is_doctor"] != 1:
        return redirect(url_for("login"))
    
    # Fetch all appointments (or filter by status/other criteria as needed)
    appointments = get_all_appointments()
    return render_template("doctor_dashboard.html",
                           appointments=appointments,
                           current_user=user)


# -----------------------------------------------------------------------------
# AI Skin Analysis & Product Recommendation Endpoint
# -----------------------------------------------------------------------------

UPLOAD_FOLDER = "static/uploads"
ANNOTATIONS_FOLDER = "static/annotations"
# Mapping for oiliness detection model class names
class_mapping = {
    "Jenis Kulit Wajah - v6 2023-06-17 11-53am": "oily skin",
    "-": "normal/dry skin"
}

@app.route("/predict", methods=["POST", "GET"])
def predict():
    """Handles AI Skin Analysis, detection, and AI-enhanced recommendations."""
    if request.method == "POST":
        try:
            # Ensure an image file is uploaded
            if "image" not in request.files:
                return jsonify({"error": "No image uploaded"}), 400

            image_file = request.files["image"]

            # Validate file type (Only accept JPG, JPEG, PNG)
            ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
            if not ('.' in image_file.filename and image_file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS):
                return jsonify({"error": "Invalid file type. Only JPG and PNG are allowed."}), 400

            # Save the uploaded image
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            image_filename = secure_filename(str(uuid.uuid4()) + ".jpg")
            image_path = os.path.join(UPLOAD_FOLDER, image_filename)
            image_file.save(image_path)

            # ---------------------------------------------------------------------
            # Step 1: Run Skin Detection Model
            # ---------------------------------------------------------------------
            unique_classes = set()
            skin_result = model_skin.predict(image_path, confidence=15, overlap=30).json()

            # Check if any skin/face predictions were returned.
            predictions = skin_result.get("predictions", [])
            if not predictions:
                return jsonify({
                    "error": "No face detected in the uploaded image. Please upload a clear image of your face."
                }), 400

            # Extract detected skin labels
            skin_labels = [pred["class"] for pred in predictions]
            unique_classes.update(skin_labels)

            # ---------------------------------------------------------------------
            # Step 2: Run Oiliness Detection
            # ---------------------------------------------------------------------
            custom_configuration = InferenceConfiguration(confidence_threshold=0.3)
            with CLIENT.use_configuration(custom_configuration):
                oiliness_result = CLIENT.infer(image_path, model_id="oilyness-detection-kgsxz/1")

            if not oiliness_result.get("predictions"):
                unique_classes.add("dryness")
            else:
                oiliness_classes = [
                    class_mapping.get(pred["class"], pred["class"])
                    for pred in oiliness_result.get("predictions", [])
                    if pred.get("confidence", 0) >= 0.3
                ]
                unique_classes.update(oiliness_classes)

            # ---------------------------------------------------------------------
            # Step 3: Annotate Image using Supervision
            # ---------------------------------------------------------------------
            os.makedirs(ANNOTATIONS_FOLDER, exist_ok=True)
            annotated_filename = f"annotations_{image_filename}"
            annotated_image_path = os.path.join(ANNOTATIONS_FOLDER, annotated_filename)
            image = cv2.imread(image_path)
            detections = sv.Detections.from_inference(skin_result)
            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()
            annotated_image = box_annotator.annotate(scene=image, detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
            cv2.imwrite(annotated_image_path, annotated_image)

            # ---------------------------------------------------------------------
            # Step 4: Generate AI Recommendations & Summarization
            # ---------------------------------------------------------------------
            ai_analysis_text = get_gemini_recommendations(unique_classes)

            # ---------------------------------------------------------------------
            # Step 5: Generate Product Recommendations from Dataset
            # ---------------------------------------------------------------------
            recommended_products = recommend_products_based_on_classes(list(unique_classes))

            prediction_data = {
                "classes": list(unique_classes),
                "ai_analysis": ai_analysis_text,
                "recommendations": recommended_products,
                "annotated_image": f"/{annotated_image_path}"
            }

            return jsonify(prediction_data)

        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": "An error occurred during analysis.", "details": str(e)}), 500

    # For GET requests, render the analysis page
    return render_template("face_analysis.html", data={})
# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")
