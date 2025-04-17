# 🧠 AccessGen: AI Meal Planner with Gemini

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://yourapp-url.streamlit.app/)

An intelligent meal planning application powered by Google's Gemini AI, providing:
- 🖼️ Image-based food analysis
- 📆 Personalized meal plans grounded in verified nutritional data
- 📊 Comprehensive nutritional insights

## Features

✨ **Core Functionalities**:
- **AI-Powered Food Analysis**
  Upload food images for instant calorie and macro estimation using Gemini Vision
- **Smart Meal Planning**
  Generate 7-day personalized meal plans based on dietary needs and goals, **leveraging the USDA FoodData Central API for accurate and verified nutritional information.**
- **Nutrition Database Insights**
  Explore comprehensive nutritional information for various foods, **ensuring data accuracy through integration with the USDA FoodData Central database.**
- **Multi-Language Support**
  Interface available in English, Spanish, French, and German

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Google API key with Gemini access
- USDA FoodData Central API key.

### Installation

1. **Clone Repository**
```bash
git clone [https://github.com/yourusername/accessgen-meal-planner.git](https://github.com/yourusername/accessgen-meal-planner.git)
cd accessgen-meal-planner
````

2.  **Setup Virtual Environment**

<!-- end list -->

```bash
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate  # Windows
```

3.  **Install Dependencies**

<!-- end list -->

```bash
pip install -r requirements.txt
```

4.  **Configuration**

<!-- end list -->

```bash
# Create .env file
echo "google_api_key=your_api_key_here" > .env
echo "usda_api_key=your_api_key_here" > .env

# Create data directory
mkdir -p data
# Place nutrition.csv in data folder
```

### Usage

```bash
streamlit run app.py
```


## 📖 User Guide

1.  **Image Analysis**

      - Upload food images (JPEG/PNG)
      - Get instant nutritional breakdown

2.  **Meal Planning**

      - Input personal metrics (age, weight, activity level)
      - Specify dietary preferences and restrictions
      - Generate AI-powered meal plans **with nutritional data sourced from the USDA FoodData Central API.**

3.  **Nutrition Database**

      - Browse comprehensive food nutrition data **verified against the USDA database.**
      - Filter and sort nutritional information

## 🌐 Deployment

Deploy to Streamlit Community Cloud:

1.  Create `requirements.txt`
2.  Push to GitHub
3.  Connect repo at [Streamlit Cloud](https://share.streamlit.io/)

## 📄 License

MIT License - See [LICENSE](https://www.google.com/search?q=LICENSE) for details

-----

## 🤖 Gen AI Capabilities Audit

| **Capability** | **Used?** | **Where? (Cell/Section)** | **How It’s Used** |
|------------------------------------|-----------|------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ✅ Structured Output / JSON         | Yes       | Gemini meal planning (Final prompt + table rendering)       | Gemini is prompted to return structured JSON with nested keys for daily nutrition and per-meal macros.                                                                                                                                                                                                                                     |
| ✅ Image Understanding             | Yes       | Image analysis with Gemini Vision (`analyze_selected_image`) | Uses Gemini Vision to analyze a selected food image and estimate calories/macros from visual data.                                                                                                                                                                                                                                            |
| ✅ Prompt Engineering               | Yes       | Meal planner prompt, calorie image prompt                  | Carefully crafted prompts instruct Gemini to reply in a specific format, include nutrients, and stay within caloric bounds.                                                                                                                                                                                                                     |
| ✅ Grounding                       | Yes   | User calorie estimation used in prompt                      | BMR + activity levels are used as grounding for calorie budget, **and the USDA FoodData Central API is used for grounding meal plan nutritional data.** |
| 🟡 Gen AI Evaluation               | No        | —                                                          | No automated evaluation metrics for Gemini output correctness or consistency.                                                                                                                                                                                                                                                                        |
| 🟡 Document Understanding           | No        | —                                                          | No PDFs or external docs parsed.                                                                                                                                                                                                                                                                                                                  |
| 🟡 Function Calling                 | No        | —                                                          | Gemini doesn’t call any external function dynamically.                                                                                                                                                                                                                                                                                            |
| 🟡 Embeddings + Vector Search       | No        | —                                                          | Could be used for food similarity or grouping in nutrition DB.                                                                                                                                                                                                                                                                                       |
| 🟡 RAG                             | No        | —                                                          | Gemini is not enhanced with retrieval-augmented generation.                                                                                                                                                                                                                                                                                          |
| 🟡 Agents                          | No        | —                                                          | Could automate workflow from intake → prediction → meal generation.                                                                                                                                                                                                                                                                                       |
| 🟡 MLOps / Gen AI Pipelines       | No        | —                                                          | No deployment, model tracking, or logging pipeline.                                                                                                                                                                                                                                                                                               |
| 🟡 Long Context / Caching          | No        | —                                                          | Not explicitly tested or handled.                                                                                                                                                                                                                                                                                                                  |

-----

**Note**: This application requires a valid Google API key with Gemini API access.
🔒 API keys should never be committed to version control.
