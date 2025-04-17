# 🧠 AccessGen: AI Meal Planner with Gemini

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://yourapp-url.streamlit.app/)

**AccessGen** is an intelligent meal planning application powered by Google's Gemini AI, designed to provide you with:

- 🖼️ **Image-Based Food Analysis:** Instantly estimate calories and macronutrients from uploaded food images.
- 📆 **Personalized Meal Plans:** Generate tailored 7-day meal plans based on your dietary needs and goals, grounded in verified nutritional data from the USDA FoodData Central API.
- 📊 **Comprehensive Nutritional Insights:** Explore detailed nutritional information for a wide range of foods, ensuring accuracy through the integration with the USDA database.

## ✨ Features

**Core Functionalities:**

- **AI-Powered Food Analysis:** Upload images of your meals (JPEG/PNG) to get an immediate estimation of their calorie and macronutrient content using Gemini Vision.
- **Smart Meal Planning:** Create personalized 7-day meal plans by specifying your dietary preferences, restrictions, and fitness goals. This feature leverages the USDA FoodData Central API to ensure the accuracy and reliability of nutritional information.
- **Nutrition Database Insights:** Access a wealth of nutritional data for various foods, sourced directly from and verified against the USDA FoodData Central database.
- **Multi-Language Support:** The application interface is available in English, Spanish, French, and German, making it accessible to a wider audience.
- **Shopping List Generator:** Extract ingredients from meal plan dishes and compile a **weekly grocery list**.

## 🚀 Getting Started

### ⚙️ Prerequisites

Before you begin, ensure you have the following:

- **Python:** Version 3.9 or higher is required.
- **Google API Key:** You'll need a valid Google API key with access to the Gemini API and Gemini Vision.
- **USDA FoodData Central API Key:** Obtain a free API key from the USDA FoodData Central website to enable accurate nutritional data retrieval.

### 🛠️ Installation

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/fmWaithaka/Nutritionist-AI-Agent.git](https://github.com/fmWaithaka/Nutritionist-AI-Agent.git)
   cd Nutritionist-AI-Agent
   ```

2. **Set Up a Virtual Environment:**
   It's recommended to use a virtual environment to manage project dependencies.

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate.bat  # Windows
   ```

3. **Install Dependencies:**
   Install the required Python packages from the `requirements.txt` file.

   ```bash
   pip install -r requirements.txt
   ```

4. **Configuration:**
   Create a `.env` file in the root directory of the project to store your API keys securely.

   ```bash
   # Create .env file
   echo "google_api_key=YOUR_GOOGLE_API_KEY_HERE" > .env
   echo "usda_api_key=YOUR_USDA_API_KEY_HERE" > .env
   ```

   Replace `YOUR_GOOGLE_API_KEY_HERE` and `YOUR_USDA_API_KEY_HERE` with your actual API keys.

### 🕹️ Usage

To run the application, use the following command in your terminal:

```bash
streamlit run app.py
```
To run evaluation, use the following command in your terminal.
```bash
python evaluate_outputs.py
```

This will open the AccessGen application in your web browser.

## 📖 User Guide

1. **📸 Image Analysis:**
   - Navigate to the image analysis section.
   - Upload a food image file (JPEG or PNG format).
   - The application will use Gemini Vision to analyze the image and provide an estimated nutritional breakdown.

2. **📅 Meal Planning:**
   - Go to the meal planning section.
   - Enter your personal metrics, including age, weight, height, gender, and activity level.
   - Specify any dietary preferences (e.g., vegetarian, vegan), allergies, and foods you like or dislike.
   - Click the "Generate Meal Plan" button.
   - AccessGen will generate a personalized 7-day meal plan with dishes and their estimated nutritional information sourced from the USDA FoodData Central API.

3. **🍎 Nutrition Database:**
   - Explore the nutrition database section.
   - Browse a comprehensive list of foods and their nutritional values, all verified against the USDA database.
   - Use the search and filtering options to find specific foods or nutritional information.

## 🌐 Deployment

You can easily deploy AccessGen to the Streamlit Community Cloud:

1. **Create `requirements.txt`:** Ensure this file lists all your project dependencies.
2. **Push to GitHub:** Upload your project repository to GitHub.
3. **Connect to Streamlit Cloud:** Go to [Streamlit Cloud](https://share.streamlit.io/) and connect your GitHub repository to deploy the application.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://opensource.org/licenses/MIT) file for details.

---

## ✅ Conclusion

AccessGen is an AI-powered meal planning system that effectively integrates **image-based food recognition** and **personalized dietary planning** using Google’s **Gemini API** and the **USDA FoodData Central API**.

This application demonstrates the following capabilities:

- **Accurate Nutritional Estimation:** Utilizes Gemini Vision to estimate nutritional values from food images.
- **Personalized Calorie Calculation:** Calculates your estimated daily caloric needs using the Mifflin-St Jeor BMR equation, considering your age, gender, weight, height, activity level, and fitness goals.
- **Dietary Preference Integration:** Incorporates your dietary preferences, including allergens, diet type, and liked/disliked foods, into the meal plan generation.
- **Grounded Meal Plan Generation:** Generates a 7-day personalized meal plan, fulfilling your daily calorie targets with dishes and their nutritional breakdowns sourced from the USDA FoodData Central API.
- **Structured Output:** Provides clear and structured tables summarizing the daily and per-meal nutritional information.
- **Multilingual Support & Voice Assistant:** Expand the multilingual support and integrate a voice assistant for hands-free interaction, improving accessibility.
-  **Shopping List Generator:** Implement a feature to automatically extract ingredients from the generated meal plan and create a weekly grocery list.

---

## 💡 Future Work and Recommendations

To further enhance AccessGen, consider the following improvements:

- 🔄 **User Image Upload Support:** Explore alternative platforms like Google Colab or a dedicated web interface to enable direct image uploads at runtime for a more seamless user experience.
- 📱 **Mobile Integration:** Develop a mobile application (iOS and Android) with camera access for real-time food analysis and nutrition logging on the go.
- 📊 **Nutrient Tracking Dashboard:** Create a dashboard to visualize your nutrient intake across the meal plan, comparing it against recommended daily allowances.
- 🤖 **AI Fine-Tuning & Optimization:** Investigate fine-tuning smaller, open-source models for specific tasks to potentially improve performance and enable edge deployment.
- 🧪 **Implement Gen AI Evaluation Metrics:** Integrate automated evaluation metrics to continuously assess the correctness and consistency of the AI-generated meal plans and image analysis results.
- 🧠 **Explore Advanced Grounding Techniques:** Investigate more advanced grounding methods, such as retrieval-augmented generation (RAG), to further enhance the accuracy and relevance of the generated meal plans.

## 🤖 Gen AI Capabilities Audit

| **Capability** | **Used?** | **Where? (Cell/Section)** | **How It’s Used** |
|------------------------------------|-----------|------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ✅ Structured Output / JSON         | Yes       | Gemini meal planning (Final prompt + table rendering)       | Gemini is prompted to return structured JSON with nested keys for daily nutrition and per-meal macros. |
| ✅ Image Understanding             | Yes       | Image analysis with Gemini Vision (`analyze_selected_image`) | Uses Gemini Vision to analyze a selected food image and estimate calories/macros from visual data. |
| ✅ Prompt Engineering               | Yes       | Meal planner prompt, calorie image prompt                  | Carefully crafted prompts instruct Gemini to reply in a specific format, include nutrients, and stay within caloric bounds. |
| ✅ Grounding                       | Yes       | Meal plan generation                                       | **The meal plan generation is grounded by using the USDA FoodData Central API to retrieve accurate and verified nutritional information for the suggested meals.** |
| ✅ Gen AI Evaluation               | Yes       | Evaluation Script                                          | Automated the process of evaluating Gemini's output correctness and consistency using a custom evaluation script, leveraging a language model as a judge and incorporating logging for monitoring and improvement. |
| 🟡 Document Understanding           | No        | —                                                          | No PDFs or external docs parsed. |
| 🟡 Function Calling                 | No        | —                                                          | Gemini doesn’t call any external function dynamically. |
| 🟡 Embeddings + Vector Search       | No        | —                                                          | Could be used for food similarity or grouping in nutrition DB. |
| 🟡 RAG                             | No        | —                                                          | Gemini is not enhanced with retrieval-augmented generation. |
| 🟡 Agents                          | No        | —                                                          | Could automate workflow from intake → prediction → meal generation. |
| 🟡 MLOps / Gen AI Pipelines       | No        | —                                                          | No deployment, model tracking, or logging pipeline. |
| 🟡 Long Context / Caching          | No        | —                                                          | Not explicitly tested or handled. |

---

**Note**: This application requires valid Google API and USDA FoodData Central API keys.
🔒 API keys should never be committed to version control.
