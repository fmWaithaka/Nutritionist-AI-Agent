import os
import re
import json
import base64
import io
import pandas as pd
import streamlit as st
from PIL import Image
import requests
import logging
from dotenv import load_dotenv

# --- Set page config FIRST ---
st.set_page_config(
    page_title="AI Meal Planner",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set up logging with both file and console output
LOG_FILE = "hereAreTheLogs.log"

logging.basicConfig(
    filename="hereAreTheLogs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a",
    force=True
)

# --- Configuration ---
load_dotenv()  # Load environment variables from .env

# Validate Google API Key
GOOGLE_API_KEY = os.getenv("google_api_key")
if not GOOGLE_API_KEY:
    st.error("❌ Google API key ('google_api_key') not found in .env file")
    st.stop()

# --- Data Loading ---


@st.cache_data
def load_nutrition_data():
    try:
        df = pd.read_csv("data/nutrition.csv")
        return df.dropna().reset_index(drop=True)
    except FileNotFoundError:
        st.error("❌ Failed to load nutrition data: `data/nutrition.csv` not found.")
        st.info("Please ensure the file exists in a 'data' subfolder.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to load nutrition data: {str(e)}")
        st.stop()


nutrition_df = load_nutrition_data()

# --- Helper Functions (Mostly from Notebook/Previous Attempts) ---

# calculate_calories function remains the same as your previous version


def calculate_calories(age, weight, height, gender, activity, goal):
    """Calculate daily caloric needs with validation"""
    # ... (Keep the implementation from your previous code) ...
    try:
        if not (18 <= age <= 100):
            raise ValueError("Age must be between 18-100")
        if weight < 30 or height < 100:
            raise ValueError("Invalid weight/height")

        # Mifflin-St Jeor Equation
        if gender == "Male":
            bmr = 10*weight + 6.25*height - 5*age + 5
        else:
            bmr = 10*weight + 6.25*height - 5*age - 161

        activity_factors = {
            "Sedentary": 1.2, "Light": 1.375, "Moderate": 1.55,
            "Active": 1.725, "Very Active": 1.9
        }
        calories = bmr * activity_factors[activity]

        # Adjust for goals
        if goal == "Lose Weight":
            calories -= 500
        elif goal == "Gain Muscle":
            calories += 500

        return max(1200, round(calories))

    except Exception as e:
        st.error(f"Calculation error: {str(e)}")
        # Returning None instead of stopping might be better in some flows
        return None  # Indicate calculation failure

# --- Helper Functions for parsing/displaying (from notebook) ---


def extract_num(val):
    try:
        if isinstance(val, (int, float)):
            return round(val, 2)
        found = re.findall(r"(\d+\.?\d*)", str(val))
        return round(float(found[0]), 2) if found else 0
    except:
        return 0


def format_number(val):
    val = extract_num(val)
    return int(val) if val == int(val) else round(val, 1)


def estimate_grams(portion):
    # Using the logic from the original notebook example
    portion = str(portion).lower()  # Ensure it's a string
    if "bowl" in portion:
        return "~200g"
    if "plate" in portion:
        return "~350g"
    if "slice" in portion:
        return "~80g"
    if "cup" in portion:
        return "~240g"
    if "egg" in portion:
        return "~50g"
    if "serving" in portion:
        return "~150g"
    # Try to extract explicit grams
    match = re.search(r"(\d+)\s*g", portion)
    if match:
        return f"{match.group(1)}g"
    # Fallback or keep original if no estimate possible
    return portion  # + " (~250g)" # Optional fallback estimate

# --- API Call Functions (using requests) ---


def analyze_image_with_rest(image_bytes, language="English"):
    """Sends image to Gemini Vision REST API using requests."""
    if not image_bytes:
        st.warning("No image bytes provided for analysis.")
        return None

    try:
        # Convert image bytes to base64
        img_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Use the model specified in the notebook
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GOOGLE_API_KEY}"
        # Match the prompt structure from the notebook for JSON output
        prompt = (
            f"Estimate the calories in this food image. "
            f"Reply ONLY in valid JSON format like: {{\"food\": \"...\", \"estimated_calories\": ..., "
            f"\"macros\": {{\"protein\": ..., \"carbs\": ..., \"fat\": ...}}, "
            # Keep micros if needed
            f"\"micros\": {{\"fiber\": ..., \"sugar\": ..., \"sodium\": ...}}, "
            f"\"portion_grams\": 100}}. Respond in {language} where appropriate (like food name)."
        )

        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                # Assume JPEG, adjust if handling PNG etc.
                                "mime_type": "image/jpeg",
                                "data": img_base64
                            }
                        }
                    ]
                }
            ]
        }

        response = requests.post(
            api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        # Extract text result (structure depends on API version, check notebook)
        # Assuming the structure from the notebook example:
        result_json = response.json()
        if not result_json.get("candidates"):
            st.error("❌ Analysis Error: No 'candidates' found in API response.")
            st.json(result_json)  # Show the full response for debugging
            return None

        text_result = result_json["candidates"][0]["content"]["parts"][0]["text"]

        # Parse JSON from the text result (using regex like notebook)
        match = re.search(r"\{.*\}", text_result, re.DOTALL)
        if not match:
            st.error("⚠️ Could not parse JSON from Gemini Vision response.")
            st.code(text_result, language="text")
            return None

        try:
            analysis_data = json.loads(match.group())
            return analysis_data
        except json.JSONDecodeError as e:
            st.error(f"⚠️ Failed to decode JSON from vision response: {e}")
            st.code(match.group())  # Show the text that failed parsing
            return None

    except requests.exceptions.RequestException as e:
        st.error(f"❌ API Request Error (Vision): {e}")
        return None
    except Exception as e:
        st.error(f"❌ An unexpected error occurred during image analysis: {e}")
        return None


def generate_meal_plan_with_rest(calorie_target, preferences, language="English"):
    """Generates meal plan using Gemini Text REST API with requests."""
    logging.info("Entering generate_meal_plan_with_rest")  # Use logging.info
    if not calorie_target or not preferences:
        # Use logging.warning
        logging.warning("Missing calorie target or preferences for meal plan.")
        return None

    try:
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GOOGLE_API_KEY}"

        # Format restrictions and preferences for the prompt
        restrictions_str = ', '.join(preferences.get(
            'restrictions', [])) if preferences.get('restrictions') else 'None'
        favorites_str = preferences.get('favorites', 'Any')
        dislikes_str = preferences.get('dislikes', 'None')

        meal_prompt = (
            f"You are a nutritionist AI assistant. The user needs **{calorie_target} kcal/day**.\n"
            f"Generate a JSON meal plan for 7 days ONLY.\n\n"
            f"User Preferences:\n"
            f"- Goal: {preferences.get('goal', 'Maintain Weight')}\n"
            f"- Diet/Restrictions: {restrictions_str}\n"
            f"- Favorite Foods: {favorites_str}\n"
            f"- Disliked Foods: {dislikes_str}\n\n"
            f"Each day MUST include:\n"
            f"- Four meals (breakfast, lunch, dinner, snacks)\n"
            f"- Each meal MUST include keys: 'dish_name' (string), 'portion_size' (string, e.g., '1 bowl', '300g'), "
            f"'nutrition' (object with keys: 'calories', 'protein', 'carbs', 'fat' as numbers or strings like '100 kcal')\n"
            f"- A 'daily_nutrition' summary object for each day (with numerical totals for calories, protein, carbs, fat)\n\n"
            # Critical instruction
            f"Respond ONLY with a valid JSON object (no introductory text, no explanations, no markdown formatting) under the top-level key: 'meal_plan'."
            f"Ensure the language used for dish names is primarily {language}."
        )

        logging.info("Sending Meal Plan Prompt:\n%s", meal_prompt)

        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": meal_prompt}]
                }
            ]
            # Add generationConfig if needed (temperature, etc.)
            # "generationConfig": { "temperature": 0.7 }
        }

        logging.info("Making API call to: %s", api_url)  # Log API URL
        response = requests.post(
            api_url, headers=headers, json=payload, timeout=120)

        logging.info("API Response Status Code: %s", response.status_code)
        logging.info("Raw API Response Text (first 500 chars):\n%s",
                     response.text[:500])
        if len(response.text) > 500:
            logging.info("... (response truncated)")

        # Raise HTTPError for bad responses (4xx or 5xx) AFTER printing debug info
        response.raise_for_status()

        result_json = response.json()
        if not result_json.get("candidates"):
            logging.error(
                "Meal Plan Error: No 'candidates' found in API response JSON.")
            logging.error("API JSON Response: %s", result_json)
            st.error(
                "❌ Meal Plan Error: No 'candidates' found in API response JSON.")
            return None

        # --- DEBUG: Check parts structure ---
        try:
            text_result = result_json["candidates"][0]["content"]["parts"][0]["text"]
            logging.info(
                "Extracted text_result from candidate (first 500 chars):\n%s", text_result[:500])
            if len(text_result) > 500:
                logging.info("... (text_result truncated)")
            # logging.debug("Full Extracted text_result:\n%s", text_result)
        except (KeyError, IndexError, TypeError) as e:
            logging.error(
                "Meal Plan Error: Could not extract text part from API response structure: %s", e)
            logging.error("Problematic API JSON Response: %s", result_json)
            st.error(
                f"❌ Meal Plan Error: Could not extract text part from API response structure: {e}")
            return None
        # --- END DEBUG ---

        match = re.search(
            r"```json\s*(\{.*?\})\s*```", text_result, re.DOTALL | re.IGNORECASE)
        if not match:
            # Fallback: Check if the *entire* string is JSON
            if text_result.strip().startswith('{') and text_result.strip().endswith('}'):
                match = re.search(r"(\{.*?\})", text_result.strip(), re.DOTALL)
            else:
                # Final fallback: find any JSON-like structure within
                match = re.search(r"(\{.*?\})", text_result, re.DOTALL)

        if not match:
            logging.error(
                "Could not parse/find JSON block within the Gemini text response.")
            st.error(
                "⚠️ Could not parse/find JSON block within the Gemini text response.")
            return None

        try:
            json_str = match.group(1)
            logging.info("Found potential JSON block to decode:\n%s", json_str)

            meal_data = json.loads(json_str)
            logging.info("Successfully decoded JSON.")

            # --- Check for 'meal_plan' key ---
            if "meal_plan" not in meal_data:
                logging.error(
                    "Meal Plan Error: JSON decoded successfully, but the required 'meal_plan' key is missing.")
                logging.error("Structure of decoded JSON: %s", meal_data)
                # st.json(meal_data)
                return None
            # --- END Check ---

            final_plan_data = meal_data.get("meal_plan")

            # --- *** MODIFY THIS CHECK *** ---
            if isinstance(final_plan_data, dict):
                logging.info("Successfully extracted meal plan dictionary with %s days.", len(
                    final_plan_data))
                return final_plan_data
            else:
                logging.error("Meal Plan Error: The value under 'meal_plan' is not a dictionary (type: %s).", type(
                    final_plan_data))
                logging.error("Full Decoded JSON: %s", meal_data)
                st.json(meal_data)
                return None

        except json.JSONDecodeError as e:
            logging.error("Failed to decode the extracted JSON block: %s", e)
            logging.error("Problematic JSON string: %s", json_str)
            st.error(f"⚠️ Failed to decode the extracted JSON block: {e}")
            return None

    except requests.exceptions.RequestException as e:
        logging.error("API Request Error (Meal Plan): %s", e)
        st.error(f"❌ API Request Error (Meal Plan): {e}")
        return None
    except Exception as e:
        logging.exception(
            "An unexpected error occurred during meal plan generation.")
        st.error(f"❌ An error occurred during meal plan generation: {e}")
        return None
    finally:
        logging.info("Exiting generate_meal_plan_with_rest")

# --- Add this function definition alongside your other functions ---


def generate_grocery_list_with_rest(meal_plan_dict, language="English"):
    """Generates a grocery list using Gemini Text REST API based on dish names."""
    logging.info("Entering generate_grocery_list_with_rest")
    if not meal_plan_dict or not isinstance(meal_plan_dict, dict):
        logging.warning(
            "Invalid or empty meal_plan_dict provided for grocery list generation.")
        return None

    all_dishes = []
    try:
        # Iterate through the dictionary structure { "day1": {...}, "day2": {...} }
        for day_key, day_content in meal_plan_dict.items():
            if not isinstance(day_content, dict):
                continue  # Skip if day data isn't a dict

            for meal_type in ["breakfast", "lunch", "dinner", "snacks"]:
                info = day_content.get(meal_type)

                if isinstance(info, dict):
                    # Handle single dish_name or multiple numbered ones (like snacks example)
                    suffix = ""
                    i = 1
                    while True:
                        dish_key = f"dish_name{suffix}"
                        if dish_key in info:
                            dish_name = info.get(dish_key)
                            # Ensure non-empty string
                            if dish_name and isinstance(dish_name, str) and dish_name.strip():
                                all_dishes.append(dish_name.strip())
                            i += 1
                            suffix = str(i)
                        # Break logic: handles cases with only dish_name, or stops after numbered keys end
                        elif suffix == "" and "dish_name" in info:
                            break  # Handled the first/only one
                        elif suffix != "":
                            break  # Stop if dish_nameN wasn't found
                        else:
                            break  # Stop if even dish_name wasn't found

                elif isinstance(info, list):  # Handle list of snacks if format changes
                    for item in info:
                        if isinstance(item, dict) and "dish_name" in item:
                            dish_name = item.get("dish_name")
                            # Ensure non-empty string
                            if dish_name and isinstance(dish_name, str) and dish_name.strip():
                                all_dishes.append(dish_name.strip())

        if not all_dishes:
            logging.warning(
                "No dish names extracted from the meal plan for grocery list.")
            st.warning(
                "Could not find any dish names in the meal plan to generate a grocery list from.")
            return None

        # Get unique dish names, filter out any potential empty strings again just in case
        unique_dishes = sorted(list(set(filter(None, all_dishes))))
        if not unique_dishes:
            logging.warning("Filtered dish list is empty.")
            st.warning("No valid dish names found to generate grocery list.")
            return None

        dishes_text = ", ".join(unique_dishes)
        logging.info(
            "Generating grocery list for unique dishes: %s", dishes_text)

        # --- Construct the Prompt ---
        prompt = f"""Act as a helpful shopping assistant. Based *only* on the following list of meal dishes planned for a week, generate a likely grocery list of ingredients needed.

            Dishes Planned:
            {dishes_text}

            Instructions for Grocery List:
            - List necessary ingredients to make these dishes.
            - Combine similar items and estimate reasonable quantities for a week for one person if possible (e.g., "Onions: 2-3", "Chicken Breast: approx 1.5 lbs / 700g", "Rice: 1 box/bag"). If quantity is uncertain, just list the item name.
            - Group the ingredients into logical grocery store categories (e.g., Produce, Meat & Poultry, Fish & Seafood, Dairy & Eggs, Pantry Staples, Frozen, Spices & Oils). Use these exact category names as Markdown H3 headings (### Category Name).
            - Exclude extremely common household items like: salt, black pepper, water, generic vegetable/canola oil (unless a specific type like olive oil or sesame oil is clearly implied by a dish).
            - Format the output *only* as a Markdown list with items under their category headings. Use bullet points (*) for each item.
            - Ensure the list is practical for someone planning to cook these meals.
            - Do not include any introductory or concluding sentences, just the categorized Markdown list.
            """

        # --- Make API Call ---
        # Or gemini-pro
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GOOGLE_API_KEY}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            # Slightly lower temp for more predictable list
            "generationConfig": {"temperature": 0.5}
        }

        logging.info("Making API call for grocery list...")
        response = requests.post(
            api_url, headers=headers, json=payload, timeout=90)
        logging.info("Grocery List API Response Status: %s",
                     response.status_code)
        # Log more text if needed for debugging, but keep it reasonable
        logging.info(
            "Grocery List Raw Response Text (first 1000 chars):\n%s", response.text[:1000])
        if len(response.text) > 1000:
            logging.info("... (response truncated)")

        response.raise_for_status()  # Check for HTTP errors

        result_json = response.json()
        if not result_json.get("candidates"):
            logging.error("Grocery List Error: No 'candidates' in response.")
            st.error(
                "❌ Grocery List Error: The AI service did not provide a valid response.")
            return None

        # Extract text - assuming simple text response for the list
        try:
            # Check response structure carefully
            if not result_json["candidates"][0].get("content") or not result_json["candidates"][0]["content"].get("parts"):
                logging.error(
                    "Grocery List Error: Unexpected content/parts structure in response.")
                st.error(
                    "❌ Grocery List Error: Unexpected response structure from AI.")
                logging.error("Problematic API JSON Response: %s", result_json)
                return None

            grocery_list_text = result_json["candidates"][0]["content"]["parts"][0]["text"]
            logging.info("Successfully extracted grocery list text.")
            # Basic cleanup (remove potential markdown code block fences if AI adds them)
            grocery_list_text = re.sub(
                r"^```markdown\s*\n?", "", grocery_list_text, flags=re.IGNORECASE | re.MULTILINE)
            grocery_list_text = re.sub(
                r"\n?```\s*$", "", grocery_list_text, flags=re.IGNORECASE | re.MULTILINE)
            return grocery_list_text.strip()
        except (KeyError, IndexError, TypeError) as e:
            logging.error(
                "Grocery List Error: Could not extract text part from API response structure: %s", e)
            logging.error("Problematic API JSON Response: %s", result_json)
            st.error(
                "❌ Grocery List Error: Could not process the response from the AI service.")
            return None

    except requests.exceptions.RequestException as e:
        logging.error("API Request Error (Grocery List): %s", e)
        st.error(
            f"❌ Network Error: Failed to connect to the AI service for the grocery list ({e})")
        return None
    except Exception as e:
        # Logs traceback
        logging.exception(
            "An unexpected error occurred during grocery list generation.")
        st.error(
            f"❌ An unexpected error occurred while creating the grocery list: {e}")
        return None
    finally:
        logging.info("Exiting generate_grocery_list_with_rest")

# --- End of function definition ---


# --- Streamlit UI ---
st.title("🧠 AccessGen: AI Meal Planner (REST API Version)")

# --- Language Selection ---
with st.sidebar:
    st.header("Settings")
    lang_options = {"English": "en", "Spanish": "es",
                    "French": "fr", "German": "de"}
    lang_name = st.selectbox("🌐 Language", list(lang_options.keys()))
    lang_code = lang_options[lang_name]

# --- Image Analysis Section ---
st.header("🖼️ Food Image Analysis")
uploaded_file = st.file_uploader(
    "Upload food image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        # Read image bytes directly from uploaded file
        image_bytes = uploaded_file.getvalue()
        # Display the image using Streamlit
        st.image(image_bytes, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Analyze Nutrition"):
            with st.spinner("Analyzing with Gemini Vision (REST)..."):
                analysis_data = analyze_image_with_rest(
                    image_bytes, language=lang_name)  # Pass name for prompt

            if analysis_data:
                try:  # Add error handling for accessing potentially missing keys
                    st.subheader(f"📊 {analysis_data.get('food', 'N/A')}")
                    cols = st.columns(4)
                    # Use .get() with defaults for safety
                    calories_val = analysis_data.get('estimated_calories', 0)
                    macros = analysis_data.get('macros', {})
                    protein_val = macros.get('protein', 0)
                    carbs_val = macros.get('carbs', 0)
                    fat_val = macros.get('fat', 0)
                    portion_val = analysis_data.get('portion_grams', 0)

                    cols[0].metric(
                        "Calories", f"{format_number(calories_val)} kcal")
                    cols[1].metric("Protein", f"{format_number(protein_val)}g")
                    cols[2].metric("Carbs", f"{format_number(carbs_val)}g")
                    cols[3].metric("Fat", f"{format_number(fat_val)}g")
                    st.write(
                        f"**Portion Size:** {format_number(portion_val)}g")

                    # Optionally display micros if they are in the response
                    micros = analysis_data.get('micros', {})
                    if micros:
                        st.write("**Micronutrients (Estimated):**")
                        st.write(
                            f"- Fiber: {format_number(micros.get('fiber', 0))}g")
                        st.write(
                            f"- Sugar: {format_number(micros.get('sugar', 0))}g")
                        st.write(
                            f"- Sodium: {format_number(micros.get('sodium', 0))}mg")

                except Exception as display_err:
                    st.error(
                        f"Error displaying analysis results: {display_err}")
                    # Show the data that caused the error
                    st.json(analysis_data)
            else:
                st.error("Analysis did not return valid data.")

    except Exception as e:
        st.error(f"Image processing failed: {str(e)}")


# --- Meal Planner Section ---
st.header("📆 Personalized Meal Plan")
with st.form("user_profile"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        weight = st.number_input(
            "Weight (kg)", min_value=30.0, value=70.0, step=0.5)
        height = st.number_input(
            "Height (cm)", min_value=100.0, value=170.0, step=0.5)
    with col2:
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        activity = st.selectbox("Activity Level",
                                ["Sedentary", "Light", "Moderate", "Active", "Very Active"], index=2)
        goal = st.selectbox(
            "Goal", ["Lose Weight", "Maintain Weight", "Gain Muscle"], index=1)

    restrictions = st.multiselect("Dietary Restrictions",
                                  ["Vegetarian", "Vegan", "Gluten-Free", "Dairy-Free", "Nut-Free", "Halal", "Kosher"])
    favorites = st.text_input("Favorite Foods (comma-separated, optional)")
    dislikes = st.text_input("Disliked Foods (comma-separated, optional)")

    submitted = st.form_submit_button("🍽️ Generate 7-Day Plan")

   # --- This block runs ONLY when the "Generate 7-Day Plan" form button is clicked ---
    if submitted:
        # 1. Clear previous results from session state first
        logging.info(
            "Form submitted. Clearing previous meal plan from session state.")
        if 'meal_plan_data' in st.session_state:
            del st.session_state['meal_plan_data']
        # Optional: Clear grocery list state too
        # if 'grocery_list_data' in st.session_state:
        #     del st.session_state['grocery_list_data']

        # 2. Calculate calories
        logging.info("Calculating calories...")
        calculated_calories = calculate_calories(
            age, weight, height, gender, activity, goal)

        # 3. Proceed only if calorie calculation is successful
        if calculated_calories:
            logging.info(
                f"Calorie calculation successful: {calculated_calories} kcal/day.")
            # User feedback
            st.info(f"Targeting approximately {calculated_calories} kcal/day.")

            # 4. Prepare user preferences
            user_prefs = {
                "goal": goal, "restrictions": restrictions,
                "favorites": favorites, "dislikes": dislikes
            }
            logging.info(f"User preferences prepared: {user_prefs}")

            # 5. Call the meal plan generation function
            meal_plan_dict_result = None
            with st.spinner("Creating your personalized meal plan (REST)..."):
                logging.info(
                    "Calling generate_meal_plan_with_rest function...")
                meal_plan_dict_result = generate_meal_plan_with_rest(
                    calculated_calories, user_prefs, language=lang_name)

            # 6. Check result and store in session state (or handle failure)
            if meal_plan_dict_result and isinstance(meal_plan_dict_result, dict):
                # STORE HERE
                st.session_state['meal_plan_data'] = meal_plan_dict_result
                logging.info(
                    "Meal plan generated and stored in session state.")
                # Feedback ONLY
                st.success(
                    "Meal plan generated successfully! See results below.")
                # NO DISPLAY CODE HERE
            else:
                logging.error(
                    "generate_meal_plan_with_rest did not return a valid dictionary.")
                # Feedback ONLY
                st.error(
                    "Meal plan generation failed. Check console logs for details.")
                # Ensure session state is clear if failure occurs after initial clear
                if 'meal_plan_data' in st.session_state:
                    del st.session_state['meal_plan_data']

        # 7. Handle calorie calculation failure
        else:
            logging.error("Calorie calculation failed.")
            st.error("Could not calculate calorie needs. Please check inputs.")
            # Ensure session state is clear
            if 'meal_plan_data' in st.session_state:
                del st.session_state['meal_plan_data']

    # --- The 'if submitted:' block MUST END HERE ---

    # --- IMPORTANT ---
    # This section is OUTSIDE the 'if submitted:' block
# --- This block runs AFTER 'if submitted:', checking session state ---
# --- Make sure it's NOT nested inside 'if submitted:' ---

    if 'meal_plan_data' in st.session_state and st.session_state['meal_plan_data']:
        # Retrieve the data stored earlier
        meal_plan_dict = st.session_state['meal_plan_data']

        # --- Display the Meal Plan ---
        st.subheader("📅 Your 7-Day Meal Plan")
        with st.expander("View Generated Meal Plan Details", expanded=True):
            # (Paste your meal plan display loop, dataframes, etc. here)
            try:
                sorted_days = sorted(meal_plan_dict.items(), key=lambda item: int(
                    re.search(r'\d+', item[0]).group()) if re.search(r'\d+', item[0]) else 0)
            except:
                sorted_days = meal_plan_dict.items()
            for day_key, day_content in sorted_days:
                # Use the key for the label (e.g., "Day 1" from "day1")
                day_num_match = re.search(r'\d+', day_key)
                # Format label
                day_label = f"Day {day_num_match.group()}" if day_num_match else day_key.capitalize(
                )

                st.markdown(f"#### 🗓️ {day_label}")  # Use H4 inside expander
                meal_rows = []

                # Inner loop to process meals
                for meal_type in ["breakfast", "lunch", "dinner", "snacks"]:
                    info = day_content.get(meal_type)  # Use .get for safety

                    # --- Handle the specific 'snacks' dictionary structure ---
                    if meal_type == "snacks" and isinstance(info, dict):
                        snack_items_text = []
                        combined_nutrition = {
                            "calories": 0, "protein": 0, "carbs": 0, "fat": 0}
                        suffix = ""
                        i = 1
                        processed_snack = False
                        while True:  # Loop for dish_name, dish_name2, dish_name3...
                            dish_key = f"dish_name{suffix}"
                            portion_key = f"portion_size{suffix}"
                            nutrition_key = f"nutrition{suffix}"

                            if dish_key in info:
                                processed_snack = True
                                dish_name = info.get(dish_key, "Snack")
                                portion_size = estimate_grams(
                                    info.get(portion_key, 'N/A'))
                                snack_items_text.append(
                                    f"{dish_name} ({portion_size})")

                                nutr = info.get(nutrition_key, {})
                                if isinstance(nutr, dict):  # Ensure nutrition is a dict
                                    combined_nutrition["calories"] += extract_num(
                                        nutr.get("calories", 0))
                                    combined_nutrition["protein"] += extract_num(
                                        nutr.get("protein", 0))
                                    combined_nutrition["carbs"] += extract_num(
                                        nutr.get("carbs", 0))
                                    combined_nutrition["fat"] += extract_num(
                                        nutr.get("fat", 0))

                                i += 1
                                suffix = str(i)
                            else:
                                break  # No more snacks found with this numbered pattern

                        if processed_snack:
                            meal_rows.append({
                                "Meal": meal_type.capitalize(),
                                "Dish": ", ".join(snack_items_text),
                                "Portion": "Multiple Items",  # Indicate multiple snacks were combined
                                "Calories (kcal)": format_number(combined_nutrition["calories"]),
                                "Protein (g)": format_number(combined_nutrition["protein"]),
                                "Carbs (g)": format_number(combined_nutrition["carbs"]),
                                "Fat (g)": format_number(combined_nutrition["fat"]),
                            })
                        elif "dish_name" in info:  # Handle if only one snack defined without numbering
                            nutrition = info.get("nutrition", {})
                            meal_rows.append({"Meal": meal_type.capitalize(), "Dish": info.get("dish_name", "N/A"), "Portion": estimate_grams(info.get("portion_size", "N/A")), "Calories (kcal)": format_number(nutrition.get(
                                "calories", 0)), "Protein (g)": format_number(nutrition.get("protein", 0)), "Carbs (g)": format_number(nutrition.get("carbs", 0)), "Fat (g)": format_number(nutrition.get("fat", 0)), })

                    # --- Handle standard meal dictionary ---
                    elif isinstance(info, dict):
                        nutrition = info.get("nutrition", {})
                        meal_rows.append({
                            "Meal": meal_type.capitalize(),
                            "Dish": info.get("dish_name", "N/A"),
                            "Portion": estimate_grams(info.get("portion_size", "N/A")),
                            "Calories (kcal)": format_number(nutrition.get("calories", 0)),
                            "Protein (g)": format_number(nutrition.get("protein", 0)),
                            "Carbs (g)": format_number(nutrition.get("carbs", 0)),
                            "Fat (g)": format_number(nutrition.get("fat", 0)),
                        })
                    # --- Handle snacks as list (fallback if API format changes) ---
                    elif isinstance(info, list) and meal_type == "snacks":
                        # Combine snacks list items (keep your previous logic if needed)
                        # This part might need adjustment if the API ever returns a list for snacks
                        st.warning(
                            f"Debug: Snacks for {day_label} were a list, processing might be basic.")
                        combined_dish = ", ".join(
                            [s.get("dish_name", "Snack") for s in info if isinstance(s, dict)])
                        # Add logic to sum nutrition from the list...
                        if combined_dish:
                            meal_rows.append({"Meal": "Snacks", "Dish": combined_dish, "Portion": "List Items",
                                             "Calories (kcal)": "N/A", "Protein (g)": "N/A", "Carbs (g)": "N/A", "Fat (g)": "N/A"})

                # Display the dataframe for the day
                if meal_rows:
                    df = pd.DataFrame(meal_rows)
                    # Use st.dataframe for better table rendering inside expander
                    st.dataframe(df, hide_index=True, use_container_width=True)
                else:
                    st.warning(f"No meal data processed for {day_label}.")

                # Display daily totals
                total = day_content.get("daily_nutrition", {})
                if total and isinstance(total, dict):
                    st.markdown("**Daily Totals (Estimated):**")
                    # Use columns for a compact layout of totals
                    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
                    with col_t1:
                        st.metric(
                            "Calories", f"{format_number(total.get('calories', 0))} kcal")
                    with col_t2:
                        st.metric(
                            "Protein", f"{format_number(total.get('protein', 0))} g")
                    with col_t3:
                        st.metric(
                            "Carbs", f"{format_number(total.get('carbs', 0))} g")
                    with col_t4:
                        st.metric(
                            "Fat", f"{format_number(total.get('fat', 0))} g")
                st.divider()  # Add a line between days

# --- Grocery List Button (Now correctly placed OUTSIDE form logic) ---
st.divider()  # Add a separator before the button
if st.button("🛒 Generate Weekly Grocery List"):
    with st.spinner("Generating grocery list based on meal plan..."):
        # Pass the meal plan dictionary retrieved from session state
        grocery_list_md = generate_grocery_list_with_rest(
            meal_plan_dict, language=lang_name)

    if grocery_list_md:
        st.subheader("🛒 Weekly Grocery List (AI Generated)")
        st.markdown(grocery_list_md)  # Render the Markdown list
        st.caption(
            "Note: This is an AI-generated estimate. Quantities may need adjustment. Please double-check against recipes if available.")
    else:
        # Error messages displayed within the function
        st.warning("Could not generate the grocery list.")
# --- End of Grocery List Section ---


# --- Nutrition Database ---
st.header("📊 Nutrition Database")
with st.expander("View Full Nutrition Data"):
    st.dataframe(nutrition_df, use_container_width=True)
