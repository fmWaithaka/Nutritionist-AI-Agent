# 🧠 AccessGen: AI Meal Planner with Gemini

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://yourapp-url.streamlit.app/)

An intelligent meal planning application powered by Google's Gemini AI, providing:
- 🖼️ Image-based food analysis
- 📆 Personalized meal plans
- 📊 Nutritional database insights

## Features

✨ **Core Functionalities**:
- **AI-Powered Food Analysis**  
  Upload food images for instant calorie and macro estimation using Gemini Vision
- **Smart Meal Planning**  
  Generate 7-day personalized meal plans based on dietary needs and goals
- **Nutrition Database**  
  Explore comprehensive nutritional information for various foods
- **Multi-Language Support**  
  Interface available in English, Spanish, French, and German

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Google API key with Gemini access

### Installation

1. **Clone Repository**
```bash
git clone https://github.com/yourusername/accessgen-meal-planner.git
cd accessgen-meal-planner
```

2. **Setup Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate  # Windows
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Configuration**
```bash
# Create .env file
echo "google_api_key=your_api_key_here" > .env

# Create data directory
mkdir -p data
# Place nutrition.csv in data folder
```

### Usage
```bash
streamlit run app.py
```

## 🛠️ Configuration

| File/Folder       | Purpose                          |
|-------------------|----------------------------------|
| `.env`            | Store Google API key             |
| `data/nutrition.csv` | Nutritional database           |
| `hereAreTheLogs.log` | Application logs (auto-created) |

## 📖 User Guide

1. **Image Analysis**  
   - Upload food images (JPEG/PNG)
   - Get instant nutritional breakdown

2. **Meal Planning**  
   - Input personal metrics (age, weight, activity level)
   - Specify dietary preferences and restrictions
   - Generate AI-powered meal plans

3. **Nutrition Database**  
   - Browse comprehensive food nutrition data
   - Filter and sort nutritional information

## 🌐 Deployment

Deploy to Streamlit Community Cloud:
1. Create `requirements.txt`
2. Push to GitHub
3. Connect repo at [Streamlit Cloud](https://share.streamlit.io/)

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

---

**Note**: This application requires a valid Google API key with Gemini API access.  
🔒 API keys should never be committed to version control.
