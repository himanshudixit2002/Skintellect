# Skintellect Documentation

## Project Overview

Skintellect is an intelligent skincare analysis and recommendation system designed to provide users with personalized skincare advice and product recommendations based on their individual skin conditions and concerns. The system combines computer vision techniques with skincare science to analyze user-uploaded images, identify potential skin issues, and suggest appropriate products and routines. The target audience includes individuals seeking personalized skincare solutions, as well as dermatologists and skincare professionals looking for tools to assist in diagnosis and treatment planning.

## Features

- **AI-powered skin condition analysis using YOLOv8 object detection**: This feature utilizes the YOLOv8 object detection model to identify various skin conditions from user-uploaded images. For example, it can detect acne, hyperpigmentation, and other skin lesions with high accuracy. The detected conditions are then used to provide personalized recommendations.
- **Personalized product recommendations**: Based on the identified skin conditions and user-provided survey data, the system recommends suitable skincare products from a curated database. The recommendations are tailored to address specific concerns and skin types.
- **Image-based skin assessment**: Users can upload images of their skin, and the system will analyze the images to identify potential issues and provide an overall assessment of skin health.
- **Appointment booking system**: The system allows users to book appointments with dermatologists or skincare professionals through an integrated scheduling system.
- **User authentication & profile management**: Users can create accounts, manage their profiles, and track their skincare progress over time.

## Tech Stack

- **Backend**: Python Flask (app.py): Flask is a lightweight web framework that provides the foundation for building the backend API and handling user requests.
- **ML Framework**: TensorFlow/Keras (final_model.h5): TensorFlow and Keras are used to build and train the custom CNN model for skin analysis.
- **Object Detection**: Ultralytics YOLOv8 (yolov8n.pt): YOLOv8 is a state-of-the-art object detection model used to identify skin lesions and other conditions in user-uploaded images.
- **Database**: SQLite (app.db): SQLite is a lightweight, file-based database used to store user data, survey responses, appointment information, and other application data.
- **Frontend**: HTML5/CSS3 + Jinja2 templating: HTML5 and CSS3 are used to build the user interface, while Jinja2 is used to dynamically generate HTML pages from templates.
- **Dependencies**:
    - Flask==3.0.0: A micro web framework for Python.
    - requests==2.31.0: A library for making HTTP requests.
    - python-dotenv==1.0.0: A library for loading environment variables from a .env file.
    - gunicorn==21.2.0: A production WSGI server for deploying Flask applications.
    - werkzeug==3.0.1: A comprehensive WSGI web application library.
    - opencv-python-headless==4.8.1.78: A library for computer vision tasks.
    - openai==1.12.0: The OpenAI Python library for accessing the OpenAI API.
    - numpy==1.26.3: A library for numerical computing.
    - supervision==0.18.0: A computer vision framework for automation.
    - dateparser: A library for parsing dates in various formats.
    - langchain: A framework for developing applications powered by language models.
    - sentence-transformers: A library for generating sentence embeddings.
    - rasa: A framework for building conversational AI chatbots.
    - google-generativeai: The Google Generative AI library for accessing Gemini models.
    - markdown: A library for parsing Markdown text.
    - h5py: A library for reading and writing HDF5 files.
    - gdown: A library for downloading files from Google Drive.

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/Skintellect.git
cd Skintellect

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements_updated.txt
```

## Usage

1. Start Flask development server:
```bash
python app.py
```

2. Access web interface at `http://localhost:5000`

3. Key paths:
    - `/face_analysis` - Skin image analysis
    - `/survey` - Skin questionnaire
    - `/recommendations` - Product suggestions

## Data and Model

- `dataset/cosmetics.csv`: This dataset contains over 10,000 skincare products with detailed ingredient lists, price information, and other relevant attributes.
- `dataset/updated_skincare_products.csv`: This dataset contains a curated list of product recommendations based on specific skin conditions and concerns.
- Custom CNN for skin analysis (model/final_model.h5): This is a custom Convolutional Neural Network (CNN) model trained to classify different skin conditions from user-uploaded images. The model architecture consists of convolutional layers, pooling layers, and fully connected layers. The model is trained using a dataset of labeled skin images and optimized for accuracy and performance.
- YOLOv8n for lesion detection (runs/train32/ weights): This is a pre-trained YOLOv8n object detection model fine-tuned to detect skin lesions and other anomalies in user-uploaded images. The model is trained on a large dataset of skin lesion images and can accurately identify and localize various types of lesions.

## API Endpoints

- `/`: Index page, redirects to login if not authenticated.
    - **Request**: GET
    - **Response**: Redirects to `/login` if the user is not authenticated, otherwise renders the `index.html` template.
- `/register`: User registration page.
    - **Request**: GET, POST
    - **GET Response**: Renders the `register.html` template.
    - **POST Request**:
        - **Parameters**: `username`, `password`, `is_doctor` (optional), `name` (optional), `age` (optional)
        - **Response**: Redirects to `/login` after successful registration, otherwise renders the `register.html` template with an error message.
- `/login`: User login page.
    - **Request**: GET, POST
    - **GET Response**: Renders the `login.html` template.
    - **POST Request**:
        - **Parameters**: `username`, `password`
        - **Response**: Redirects to `/profile` or `/survey` after successful login, otherwise renders the `login.html` template with an error message.
- `/logout`: User logout.
    - **Request**: GET
    - **Response**: Redirects to `/login`.
- `/survey`: Skin survey page.
    - **Request**: GET, POST
    - **GET Response**: Renders the `survey.html` template.
    - **POST Request**:
        - **Parameters**: Various survey questions related to skin type, concerns, routine, etc.
        - **Response**: Redirects to `/profile` after successful submission of the survey.
- `/profile`: User profile page.
    - **Request**: GET
    - **Response**: Renders the `profile.html` template with user data and skincare routine information.
- `/appointment/<int:appointment_id>`: Appointment detail page.
    - **Request**: GET
    - **Parameters**: `appointment_id` (integer)
    - **Response**: Renders the `appointment_detail.html` template with appointment details.
- `/update_appointment`: Updates appointment status (confirm/reject).
    - **Request**: POST
    - **Parameters**: `appointment_id`, `action` (confirm/reject)
    - **Response**: JSON response indicating success or failure.
- `/delete_appointment`: Deletes an appointment.
    - **Request**: POST
    - **Parameters**: `id` (appointment ID)
    - **Response**: JSON response indicating success or failure.
- `/bookappointment`: Renders the book appointment page.
    - **Request**: GET
    - **Response**: Renders the `bookappointment.html` template.
- `/appointment`: Handles appointment booking form submission.
    - **Request**: POST
    - **Parameters**: `name`, `email`, `date`, `skin`, `phone`, `age`, `reason`
    - **Response**: JSON response indicating success or failure.
- `/userappointment`: Displays user appointments.
    - **Request**: GET
    - **Response**: Renders the `userappointment.html` template with user appointments.
- `/delete_user_request`: Deletes a user appointment request.
    - **Request**: POST
    - **Parameters**: `id` (appointment ID)
    - **Response**: JSON response indicating success or failure.
- `/face_analysis`: Handles face analysis and skin condition prediction.
    - **Request**: POST
    - **Response**: JSON response with prediction results and recommendations.
- `/doctor_dashboard`: Doctor dashboard to view and manage appointments.
    - **Request**: GET
    - **Response**: Renders the `doctor_dashboard.html` template with appointment data.
- `/predict`: AI Skin Analysis & Product Recommendation Endpoint.
    - **Request**: POST, GET
    - **Response**: JSON response with prediction results and recommendations.
- `/skin_predict`: Skin Disease Classifier Prediction Endpoint.
    - **Request**: POST, GET
    - **Response**: JSON response with prediction results and AI analysis.
- `/privacy_policy`: Privacy policy page.
    - **Request**: GET
    - **Response**: Renders the `privacy_policy.html` template.
- `/terms_of_service`: Terms of service page.
    - **Request**: GET
    - **Response**: Renders the `terms_of_service.html` template.

## Database

The application uses SQLite as its database. The database schema includes the following tables:

- `users`: Stores user information (id, username, password, is_doctor).
- `survey_responses`: Stores user survey responses (id, user_id, name, age, gender, concerns, acne_frequency, comedones_count, first_concern, cosmetic_usage, skin_reaction, skin_type, medications, skincare_routine, stress_level).
- `appointment`: Stores appointment information (id, name, email, date, skin, phone, age, address, status, username).
- `skincare_routines`: Stores user skincare routines (id, user_id, morning_routine, night_routine, last_updated).
- `conversations`: Stores chatbot conversation history (id, user_id, role, message, timestamp).

**Entity Relationship Diagram (ERD)**

[A diagram illustrating the relationships between the tables in the database would be helpful here. The diagram should show the tables, their columns, and the primary and foreign key relationships between them.]

## AI Helper Functions

- `langchain_summarize(text, max_length, min_length)`: Uses a friendly prompt to generate a short, engaging summary of the provided text using LangChain. This function leverages the LangChain framework to interact with a language model and generate a concise summary of the input text. The function takes the text to be summarized, the maximum length of the summary, and the minimum length of the summary as input parameters.
- `get_gemini_recommendations(skin_conditions)`: Provides skincare recommendations based on detected skin conditions using the Gemini API. This function uses the Google Gemini API to generate personalized skincare recommendations based on a list of detected skin conditions. The function sends a prompt to the Gemini API with the detected skin conditions and receives a response with skincare recommendations.
- `recommend_products_based_on_classes(classes)`: Recommends skincare products based on detected skin conditions by searching a DataFrame of skincare products. This function searches a DataFrame of skincare products for products that are recommended for the detected skin conditions. The function filters the DataFrame based on the detected skin conditions and returns a list of recommended products.
- `generate_skincare_routine(user_details)`: Generates a personalized skincare routine based on user details using the Gemini API. This function uses the Google Gemini API to generate a personalized skincare routine based on user details such as age, gender, skin type, and concerns. The function sends a prompt to the Gemini API with the user details and receives a response with a personalized skincare routine.
- `save_skincare_routine(user_id, morning_routine, night_routine)`: Saves a user's skincare routine to the database. This function saves a user's skincare routine to the database. The function takes the user ID, morning routine, and night routine as input parameters and saves them to the `skincare_routines` table in the database.
- `get_skincare_routine(user_id)`: Retrieves a user's skincare routine from the database. This function retrieves a user's skincare routine from the database. The function takes the user ID as an input parameter and retrieves the morning routine and night routine from the `skincare_routines` table in the database.
- `build_conversation_prompt(history, user_input)`: Builds a conversation prompt for the chatbot, including conversation history and user input. This function builds a conversation prompt for the chatbot by combining the conversation history with the user's current input. The prompt is used to generate a response from the chatbot.
- `complete_answer_if_incomplete(answer)`: Checks if the chatbot's answer is complete and continues it if necessary. This function checks if the chatbot's answer is complete by checking if it ends with proper punctuation. If the answer is incomplete, the function sends a request to the Gemini API to continue the answer.

## Flask Routes

- `@app.route("/")`: Defines the route for the index page. This route handles GET requests to the root URL (`/`) and redirects the user to the login page if they are not authenticated. If the user is authenticated, the route renders the `index.html` template.
- `@app.route("/register", methods=["GET", "POST"])`: Defines the route for user registration. This route handles GET and POST requests to the `/register` URL. When a GET request is received, the route renders the `register.html` template. When a POST request is received, the route processes the registration form data, creates a new user account, and redirects the user to the login page.
- `@app.route("/login.html")`: Redirects to the login page. This route simply redirects the user to the `/login` URL.
- `@app.route("/login", methods=["GET", "POST"])`: Defines the route for user login. This route handles GET and POST requests to the `/login` URL. When a GET request is received, the route renders the `login.html` template. When a POST request is received, the route processes the login form data, authenticates the user, and redirects the user to the profile page or survey page.
- `@app.route("/logout")`: Defines the route for user logout. This route handles GET requests to the `/logout` URL and logs the user out by clearing the session. The route then redirects the user to the login page.
- `@app.route("/survey", methods=["GET", "POST"])`: Defines the route for the skin survey page. This route handles GET and POST requests to the `/survey` URL. When a GET request is received, the route renders the `survey.html` template. When a POST request is received, the route processes the survey form data and saves the user's responses to the database. The route then redirects the user to the profile page.
- `@app.route("/profile")`: Defines the route for the user profile page. This route handles GET requests to the `/profile` URL and renders the `profile.html` template with user data and skincare routine information.
- `@app.route("/appointment/<int:appointment_id>")`: Defines the route for displaying appointment details. This route handles GET requests to the `/appointment/<int:appointment_id>` URL and renders the `appointment_detail.html` template with appointment details.
- `@app.route("/update_appointment", methods=["POST"])`: Defines the route for updating appointment status. This route handles POST requests to the `/update_appointment` URL and updates the status of an appointment in the database.
- `@app.route("/delete_appointment", methods=["POST"])`: Defines the route for deleting an appointment. This route handles POST requests to the `/delete_appointment` URL and deletes an appointment from the database.
- `@app.route("/bookappointment")`: Defines the route for the book appointment page. This route handles GET requests to the `/bookappointment` URL and renders the `bookappointment.html` template.
- `@app.route("/appointment", methods=["POST"])`: Defines the route for handling appointment booking. This route handles POST requests to the `/appointment` URL and processes the appointment booking form data. The route then saves the appointment information to the database.
- `@app.route("/userappointment", methods=["GET"])`: Defines the route for displaying user appointments. This route handles GET requests to the `/userappointment` URL and renders the `userappointment.html` template with user appointments.
- `@app.route("/delete_user_request", methods=["POST"])`: Defines the route for deleting a user appointment request. This route handles POST requests to the `/delete_user_request` URL and deletes a user appointment request from the database.
- `@app.route("/face_analysis", methods=["POST"])`: Defines the route for performing face analysis. This route handles POST requests to the `/face_analysis` URL and performs face analysis on a user-uploaded image. The route then returns a JSON response with the analysis results.
- `@app.route("/doctor_dashboard")`: Defines the route for the doctor dashboard. This route handles GET requests to the `/doctor_dashboard` URL and renders the `doctor_dashboard.html` template with appointment data.
- `@app.route("/predict", methods=["POST", "GET"])`: Defines the route for AI skin analysis and product recommendation. This route handles POST and GET requests to the `/predict` URL and performs AI skin analysis and product recommendation based on user-uploaded images and survey data. The route then returns a JSON response with the prediction results and recommendations.
- `@app.route("/skin_predict", methods=["GET", "POST"])`: Defines the route for skin disease classification. This route handles GET and POST requests to the `/skin_predict` URL and performs skin disease classification on a user-uploaded image. The route then returns a JSON response with the prediction results and AI analysis.
- `@app.route("/privacy_policy")`: Defines the route for the privacy policy page. This route handles GET requests to the `/privacy_policy` URL and renders the `privacy_policy.html` template.
- `@app.route("/terms_of_service")`: Defines the route for the terms of service page. This route handles GET requests to the `/terms_of_service` URL and renders the `terms_of_service.html` template.