# app.py - Main Streamlit Application

import os
import re
import io
import logging
import pandas as pd
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

# Import functions from our custom modules
import utils
import gemini_api

# --- 1. Initial Page Configuration (MUST BE FIRST st command) ---
st.set_page_config(page_title="AI Meal Planner", layout="wide")

# --- 2. Setup: Logging, Environment Variables, API Key ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

load_dotenv()
GOOGLE_API_KEY = os.getenv("google_api_key")

if not GOOGLE_API_KEY:
    log.error("GOOGLE_API_KEY not found in .env file.")
    st.error("🔴 **Error:** `google_api_key` not found in environment variables.")
    st.info("Please ensure a `.env` file exists in the root directory with your API key.")
    st.stop()
else:
    log.info("Google API Key loaded successfully.")

st.title("🧠 AccessGen: AI Meal Planner ")

# --- Language Selection (Sidebar) ---
with st.sidebar:
    st.header("Settings")
    # Define language options (name: code)
    lang_options = {"English": "en", "Spanish": "es", "French": "fr", "German": "de"}
    # Display selectbox and get the chosen language name
    lang_name = st.selectbox("🌐 Language", list(lang_options.keys()), index=0) # Default to English
    # Get the corresponding language code if needed elsewhere
    lang_code = lang_options[lang_name]
    log.info(f"Language selected: {lang_name} ({lang_code})")


# --- Image Analysis Section ---
st.header("🖼️ 1. Food Image Analysis (Optional)")
uploaded_file = st.file_uploader("Upload an image of a food item...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image_bytes = uploaded_file.getvalue()
        st.image(image_bytes, caption="Uploaded Image", width=300) # Control width

        if st.button("🔍 Analyze Nutrition"):
            log.info(f"Analyze Nutrition button clicked for file: {uploaded_file.name}")
            with st.spinner("Analyzing image with Gemini Vision..."):
                # Call the API function from gemini_api module, passing the key
                analysis_data = gemini_api.analyze_image_with_rest(GOOGLE_API_KEY, image_bytes, language=lang_name)

            if analysis_data and isinstance(analysis_data, dict):
                log.info(f"Image analysis successful: {analysis_data}")
                try:
                    st.subheader(f"📊 Analysis Result: {analysis_data.get('food', 'N/A')}")
                    cols = st.columns(4)
                    calories_val = analysis_data.get('estimated_calories', 0)
                    macros = analysis_data.get('macros', {})
                    protein_val = macros.get('protein', 0)
                    carbs_val = macros.get('carbs', 0)
                    fat_val = macros.get('fat', 0)
                    portion_val = analysis_data.get('portion_grams', 0)

                    # Use formatting functions from utils
                    cols[0].metric("Calories", f"{utils.format_number(calories_val)} kcal")
                    cols[1].metric("Protein", f"{utils.format_number(protein_val)}g")
                    cols[2].metric("Carbs", f"{utils.format_number(carbs_val)}g")
                    cols[3].metric("Fat", f"{utils.format_number(fat_val)}g")
                    st.write(f"**Estimated Portion:** {utils.format_number(portion_val)}g")

                except Exception as display_err:
                    log.error(f"Error displaying analysis results: {display_err}", exc_info=True)
                    st.error(f"Error displaying analysis results: {display_err}")
                    st.json(analysis_data) # Show raw data if display fails
            else:
                # Error message should have been shown by the API function via st.error
                log.warning("Image analysis function did not return valid data.")
                # No extra error needed here, gemini_api function shows it

    except Exception as e:
        log.error(f"Error processing uploaded image: {e}", exc_info=True)
        st.error(f"Error processing image file: {e}")


# --- Meal Planner Section (Form Definition) ---
st.header("📆 2. Generate Personalized Meal Plan")
with st.form("user_profile_form"): # Changed key slightly for clarity
    st.write("Tell us about yourself to generate a 7-day plan:")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        weight = st.number_input("Weight (kg)", min_value=30.0, value=70.0, step=0.5)
        height = st.number_input("Height (cm)", min_value=100.0, value=170.0, step=0.5)
    with col2:
        gender = st.radio("Biological Sex (for BMR calculation):", ["Male", "Female"], index=0, horizontal=True)
        activity = st.selectbox("Activity Level:",
                                ["Sedentary", "Light", "Moderate", "Active", "Very Active"], index=2)
        goal = st.selectbox("Primary Goal:",
                            ["Lose Weight", "Maintain Weight", "Gain Muscle"], index=1)

    st.markdown("---") # Separator
    restrictions = st.multiselect("Dietary Restrictions / Preferences:",
                                  ["Vegetarian", "Vegan", "Gluten-Free", "Dairy-Free", "Nut-Free", "Halal", "Kosher", "Low Carb"])
    favorites = st.text_input("Favorite Foods (comma-separated, optional):", placeholder="e.g., Salmon, Avocado, Berries")
    dislikes = st.text_input("Disliked Foods (comma-separated, optional):", placeholder="e.g., Mushrooms, Olives")

    # Form submission button
    submitted = st.form_submit_button("🍽️ Generate 7-Day Plan")


# --- Form Submission Handling Block (AFTER `with st.form`) ---
# This block runs ONLY when the "Generate 7-Day Plan" form button is clicked
if submitted:
    log.info("Meal plan form submitted.")
    # 1. Clear previous results from session state first
    if 'meal_plan_data' in st.session_state:
        log.debug("Clearing previous meal_plan_data from session state.")
        del st.session_state['meal_plan_data']
    if 'grocery_list_data' in st.session_state: # Optional: Clear grocery list too
        log.debug("Clearing previous grocery_list_data from session state.")
        del st.session_state['grocery_list_data']

    # 2. Calculate calories using the function from utils.py
    log.info("Calculating calories...")
    calculated_calories = utils.calculate_calories(age, weight, height, gender, activity, goal)

    # 3. Proceed only if calorie calculation is successful
    if calculated_calories:
        log.info(f"Calorie calculation successful: {calculated_calories} kcal/day.")
        st.info(f"Targeting approximately **{calculated_calories} kcal/day**. Generating plan...") # Provide feedback

        # 4. Prepare user preferences dictionary
        user_prefs = {
            "goal": goal, "restrictions": restrictions,
            "favorites": favorites, "dislikes": dislikes
        }
        log.info(f"User preferences prepared: {user_prefs}")

        # 5. Call the API function (from gemini_api.py) to generate the meal plan
        meal_plan_dict_result = None
        with st.spinner("Creating your personalized meal plan... This may take a minute."):
            log.info("Calling generate_meal_plan_with_rest function...")
            # Pass the API key from this script's environment
            meal_plan_dict_result = gemini_api.generate_meal_plan_with_rest(
                GOOGLE_API_KEY, calculated_calories, user_prefs, language=lang_name
            )

        # 6. Check result and store in session state
        if meal_plan_dict_result and isinstance(meal_plan_dict_result, dict):
             st.session_state['meal_plan_data'] = meal_plan_dict_result # STORE SUCCESSFUL RESULT
             log.info(f"Meal plan generated and stored in session state. Days: {list(meal_plan_dict_result.keys())}")
             st.success("✅ Meal plan generated successfully! Results are displayed below.") # User feedback
        else:
             # Failure case - error message should be shown by the gemini_api function
             log.error("generate_meal_plan_with_rest did not return a valid dictionary.")
             # No redundant st.error here, function handles it. Ensure session state is clear.
             if 'meal_plan_data' in st.session_state:
                 del st.session_state['meal_plan_data']

    # 7. Handle calorie calculation failure
    else:
        log.error("Calorie calculation failed (returned None).")
        # Error message shown by calculate_calories via logging, show user message here
        st.error("❌ Could not calculate calorie needs based on the provided inputs. Please check and try again.")
        if 'meal_plan_data' in st.session_state: # Ensure state is clear
            del st.session_state['meal_plan_data']
# --- End of 'if submitted:' block ---


# This block runs on every script rerun IF meal plan data exists in session state
if 'meal_plan_data' in st.session_state and st.session_state['meal_plan_data']:
    log.info("Found meal plan data in session state. Preparing display.")
    # Retrieve the data stored earlier - it could be a list or a dict
    meal_plan_data = st.session_state['meal_plan_data']

    st.header("✅ Your Generated Meal Plan & Grocery List")

    with st.expander("📅 View 7-Day Meal Plan Details", expanded=True):

        # --- Check the format (dict or list) and process accordingly ---
        if isinstance(meal_plan_data, dict):
            # --- Logic for DICTIONARY format ---
            log.info("Displaying meal plan from DICTIONARY format.")
            if not meal_plan_data:
                st.warning("Meal plan dictionary is empty.")
            else:
                try:
                    # Sort dictionary by day number key (e.g., "day1", "day10")
                    sorted_items = sorted(
                        meal_plan_data.items(),
                        key=lambda item: int(re.search(r'\d+', item[0]).group()) if re.search(r'\d+', item[0]) else 0
                    )
                except Exception as sort_e:
                    log.warning(f"Could not sort dict keys numerically ({sort_e}), using default order.")
                    sorted_items = meal_plan_data.items() # Fallback to default order

                # Loop through the sorted dictionary items
                for day_key, day_content in sorted_items:
                    if not isinstance(day_content, dict):
                         log.warning(f"Skipping invalid day data for key {day_key}: {type(day_content)}")
                         continue

                    # Derive label from the dictionary key
                    day_num_match = re.search(r'\d+', day_key)
                    day_label = f"Day {day_num_match.group()}" if day_num_match else day_key.capitalize()

                    # --- COMMON Day Processing Logic ---
                    st.markdown(f"--- \n#### 🗓️ **{day_label}**")
                    meal_rows = []
                    # Inner loop to process meals for the table
                    for meal_type in ["breakfast", "lunch", "dinner", "snacks"]:
                        info = day_content.get(meal_type)
                        # --- Handle complex snacks dictionary structure ---
                        if meal_type == "snacks" and isinstance(info, dict):
                            snack_items_text = []; combined_nutrition = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}; suffix = ""; i_snack = 1; processed_snack = False # Renamed inner loop counter
                            while True:
                                dish_key = f"dish_name{suffix}"; portion_key = f"portion_size{suffix}"; nutrition_key = f"nutrition{suffix}"
                                if dish_key in info:
                                    processed_snack = True; dish_name = info.get(dish_key, "Snack"); portion_size = utils.estimate_grams(info.get(portion_key, 'N/A')); snack_items_text.append(f"{dish_name} ({portion_size})")
                                    nutr = info.get(nutrition_key, {})
                                    if isinstance(nutr, dict): combined_nutrition["calories"] += utils.extract_num(nutr.get("calories", 0)); combined_nutrition["protein"] += utils.extract_num(nutr.get("protein", 0)); combined_nutrition["carbs"] += utils.extract_num(nutr.get("carbs", 0)); combined_nutrition["fat"] += utils.extract_num(nutr.get("fat", 0))
                                    i_snack += 1; suffix = str(i_snack)
                                elif suffix == "" and "dish_name" in info: break
                                elif suffix != "": break
                                else: break
                            if processed_snack: meal_rows.append({ "Meal": meal_type.capitalize(), "Dish": ", ".join(snack_items_text), "Portion": "Multiple Items", "Calories (kcal)": utils.format_number(combined_nutrition["calories"]), "Protein (g)": utils.format_number(combined_nutrition["protein"]), "Carbs (g)": utils.format_number(combined_nutrition["carbs"]), "Fat (g)": utils.format_number(combined_nutrition["fat"]), })
                            elif "dish_name" in info: nutrition = info.get("nutrition", {}); meal_rows.append({ "Meal": meal_type.capitalize(), "Dish": info.get("dish_name", "N/A"), "Portion": utils.estimate_grams(info.get("portion_size", "N/A")), "Calories (kcal)": utils.format_number(nutrition.get("calories", 0)), "Protein (g)": utils.format_number(nutrition.get("protein", 0)), "Carbs (g)": utils.format_number(nutrition.get("carbs", 0)), "Fat (g)": utils.format_number(nutrition.get("fat", 0)), })
                        # --- Handle standard meal dictionary ---
                        elif isinstance(info, dict):
                            nutrition = info.get("nutrition", {})
                            meal_rows.append({ "Meal": meal_type.capitalize(), "Dish": info.get("dish_name", "N/A"), "Portion": utils.estimate_grams(info.get("portion_size", "N/A")), "Calories (kcal)": utils.format_number(nutrition.get("calories", 0)), "Protein (g)": utils.format_number(nutrition.get("protein", 0)), "Carbs (g)": utils.format_number(nutrition.get("carbs", 0)), "Fat (g)": utils.format_number(nutrition.get("fat", 0)), })
                        # --- Handle snacks as list (less common fallback) ---
                        elif isinstance(info, list) and meal_type == "snacks":
                            log.warning(f"Snacks for {day_label} is a list - using basic processing.")
                            combined_dish = ", ".join([s.get("dish_name", "Snack") for s in info if isinstance(s, dict)])
                            if combined_dish: meal_rows.append({"Meal": "Snacks", "Dish": combined_dish, "Portion": "List Items", "Calories (kcal)": "N/A", "Protein (g)": "N/A", "Carbs (g)": "N/A", "Fat (g)": "N/A"})
                    # Display table
                    if meal_rows: df = pd.DataFrame(meal_rows); st.dataframe(df, hide_index=True, use_container_width=True)
                    else: st.warning(f"No meal data processed for {day_label}.")
                    # Display totals
                    total = day_content.get("daily_nutrition", {})
                    if total and isinstance(total, dict):
                        st.markdown("**Daily Totals (Estimated):**")
                        col_t1, col_t2, col_t3, col_t4 = st.columns(4)
                        with col_t1: st.metric("Calories", f"{utils.format_number(total.get('calories', 0))} kcal")
                        with col_t2: st.metric("Protein", f"{utils.format_number(total.get('protein', 0))} g")
                        with col_t3: st.metric("Carbs", f"{utils.format_number(total.get('carbs', 0))} g")
                        with col_t4: st.metric("Fat", f"{utils.format_number(total.get('fat', 0))} g")                   

        elif isinstance(meal_plan_data, list):
            # --- Logic for LIST format ---
            log.info("Displaying meal plan from LIST format.")
            if not meal_plan_data: # Check if list is empty
                 st.warning("Meal plan list is empty.")
            else:
                # Loop through list items
                for i, day_content in enumerate(meal_plan_data):
                    if not isinstance(day_content, dict):
                         log.warning(f"Skipping invalid day data at index {i}: {type(day_content)}")
                         continue # Skip bad data

                    # Get day label from inside the day's dict, fallback to index
                    day_name = day_content.get("day", f"Day {i + 1}")
                    day_label = day_name.capitalize()

                    # --- COMMON Day Processing Logic ---
                    st.markdown(f"--- \n#### 🗓️ **{day_label}**")
                    meal_rows = []
                    # Inner loop to process meals for the table
                    for meal_type in ["breakfast", "lunch", "dinner", "snacks"]:
                        info = day_content.get(meal_type)
                        # --- Handle complex snacks dictionary structure ---
                        if meal_type == "snacks" and isinstance(info, dict):
                            snack_items_text = []; combined_nutrition = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}; suffix = ""; i_snack = 1; processed_snack = False # Renamed inner loop counter
                            while True:
                                dish_key = f"dish_name{suffix}"; portion_key = f"portion_size{suffix}"; nutrition_key = f"nutrition{suffix}"
                                if dish_key in info:
                                    processed_snack = True; dish_name = info.get(dish_key, "Snack"); portion_size = utils.estimate_grams(info.get(portion_key, 'N/A')); snack_items_text.append(f"{dish_name} ({portion_size})")
                                    nutr = info.get(nutrition_key, {})
                                    if isinstance(nutr, dict): combined_nutrition["calories"] += utils.extract_num(nutr.get("calories", 0)); combined_nutrition["protein"] += utils.extract_num(nutr.get("protein", 0)); combined_nutrition["carbs"] += utils.extract_num(nutr.get("carbs", 0)); combined_nutrition["fat"] += utils.extract_num(nutr.get("fat", 0))
                                    i_snack += 1; suffix = str(i_snack)
                                elif suffix == "" and "dish_name" in info: break
                                elif suffix != "": break
                                else: break
                            if processed_snack: meal_rows.append({ "Meal": meal_type.capitalize(), "Dish": ", ".join(snack_items_text), "Portion": "Multiple Items", "Calories (kcal)": utils.format_number(combined_nutrition["calories"]), "Protein (g)": utils.format_number(combined_nutrition["protein"]), "Carbs (g)": utils.format_number(combined_nutrition["carbs"]), "Fat (g)": utils.format_number(combined_nutrition["fat"]), })
                            elif "dish_name" in info: nutrition = info.get("nutrition", {}); meal_rows.append({ "Meal": meal_type.capitalize(), "Dish": info.get("dish_name", "N/A"), "Portion": utils.estimate_grams(info.get("portion_size", "N/A")), "Calories (kcal)": utils.format_number(nutrition.get("calories", 0)), "Protein (g)": utils.format_number(nutrition.get("protein", 0)), "Carbs (g)": utils.format_number(nutrition.get("carbs", 0)), "Fat (g)": utils.format_number(nutrition.get("fat", 0)), })
                        # --- Handle standard meal dictionary ---
                        elif isinstance(info, dict):
                            nutrition = info.get("nutrition", {})
                            meal_rows.append({ "Meal": meal_type.capitalize(), "Dish": info.get("dish_name", "N/A"), "Portion": utils.estimate_grams(info.get("portion_size", "N/A")), "Calories (kcal)": utils.format_number(nutrition.get("calories", 0)), "Protein (g)": utils.format_number(nutrition.get("protein", 0)), "Carbs (g)": utils.format_number(nutrition.get("carbs", 0)), "Fat (g)": utils.format_number(nutrition.get("fat", 0)), })
                        # --- Handle snacks as list (less common fallback) ---
                        elif isinstance(info, list) and meal_type == "snacks":
                            log.warning(f"Snacks for {day_label} is a list - using basic processing.")
                            combined_dish = ", ".join([s.get("dish_name", "Snack") for s in info if isinstance(s, dict)])
                            if combined_dish: meal_rows.append({"Meal": "Snacks", "Dish": combined_dish, "Portion": "List Items", "Calories (kcal)": "N/A", "Protein (g)": "N/A", "Carbs (g)": "N/A", "Fat (g)": "N/A"})
                    # Display table
                    if meal_rows: df = pd.DataFrame(meal_rows); st.dataframe(df, hide_index=True, use_container_width=True)
                    else: st.warning(f"No meal data processed for {day_label}.")
                    # Display totals
                    total = day_content.get("daily_nutrition", {})
                    if total and isinstance(total, dict):
                        st.markdown("**Daily Totals (Estimated):**")
                        col_t1, col_t2, col_t3, col_t4 = st.columns(4)
                        with col_t1: st.metric("Calories", f"{utils.format_number(total.get('calories', 0))} kcal")
                        with col_t2: st.metric("Protein", f"{utils.format_number(total.get('protein', 0))} g")
                        with col_t3: st.metric("Carbs", f"{utils.format_number(total.get('carbs', 0))} g")
                        with col_t4: st.metric("Fat", f"{utils.format_number(total.get('fat', 0))} g")
                    # --- END COMMON Day Processing Logic ---

        else:
            # --- Handle unexpected format ---
            st.error("Meal plan data in session state is not a recognized format (dictionary or list).")
            log.error(f"Unrecognized meal plan data type in session state: {type(meal_plan_data)}")
        # --- *** END TYPE CHECK AND CONDITIONAL LOGIC *** ---

    # --- Display the Grocery List Button ---
    st.markdown("---")
    if st.button("🛒 Generate Weekly Grocery List"):
        current_meal_plan_data = st.session_state.get('meal_plan_data') # This is now list OR dict
        if current_meal_plan_data:
             with st.spinner("Analyzing meal plan to create grocery list..."):
                 grocery_list_md = gemini_api.generate_grocery_list_with_rest(
                     GOOGLE_API_KEY, current_meal_plan_data, language=lang_name
                 )

             if grocery_list_md:
                 log.info("Grocery list generated successfully.")
                 st.subheader("🛒 Weekly Grocery List (AI Generated)")
                 st.markdown(grocery_list_md)
                 st.caption("Note: This is an AI-generated estimate. Quantities may need adjustment.")
             else:
                 # Error message should have been shown by the gemini_api function
                 log.warning("Grocery list generation function did not return valid data.")
        else:
             st.error("Cannot generate grocery list because meal plan data is missing.")

st.markdown("---")
st.caption("AI Meal Planner MVP | Powered by Google Gemini")