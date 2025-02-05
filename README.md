# Skintellect 🧬✨

![Project Banner](static/assets/bg.webp)

A intelligent skincare analysis and recommendation system combining computer vision with skincare science.

## Features ✨
- 🧑⚕️ AI-powered skin condition analysis using YOLOv8 object detection
- 💄 Personalized product recommendations
- 📸 Image-based skin assessment
- 📅 Appointment booking system
- 👤 User authentication & profile management

## Tech Stack 🛠️
- **Backend**: Python Flask (app.py)
- **ML Framework**: TensorFlow/Keras (final_model.h5)
- **Object Detection**: Ultralytics YOLOv8 (yolov8n.pt)
- **Database**: SQLite (app.db)
- **Frontend**: HTML5/CSS3 + Jinja2 templating

## Installation ⚙️

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

## Usage 🚀

1. Start Flask development server:
```bash
python app.py
```

2. Access web interface at `http://localhost:5000`

3. Key paths:
- `/face_analysis` - Skin image analysis
- `/survey` - Skin questionnaire
- `/recommendations` - Product suggestions

## Dataset 🔢
- `dataset/cosmetics.csv`: 10,000+ skincare products with ingredients
- `dataset/updated_skincare_products.csv`: Curated product recommendations

## Model Architecture 🧠
- Custom CNN for skin analysis (model/final_model.h5)
- YOLOv8n for lesion detection (runs/train32/ weights)

## License 📄
MIT License - See [LICENSE](LICENSE) for details