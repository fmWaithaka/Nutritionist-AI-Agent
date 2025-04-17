
import re
import logging
import pandas as pd
import streamlit as st # Needed only for the @st.cache_data decorator

log = logging.getLogger(__name__)

def extract_num(val):
    """Extracts the first number (integer or float) from a string."""
    try:
        if isinstance(val, (int, float)):
            return round(val, 2)
        # Regex to find integer or decimal numbers
        found = re.findall(r"(\d+\.?\d*)", str(val))
        return round(float(found[0]), 2) if found else 0
    except Exception as e:
        log.error(f"Error extracting number from '{val}': {e}")
        return 0

def format_number(val):
    """Formats a number, showing integer if whole, else rounded to 1 decimal."""
    num_val = extract_num(val)
    return int(num_val) if num_val == int(num_val) else round(num_val, 1)

def estimate_grams(portion):
    """Provides a rough gram estimate based on common portion descriptions."""
    portion_str = str(portion).lower()
    # Check for explicit grams first
    match = re.search(r"(\d+)\s*g", portion_str)
    if match:
        return f"{match.group(1)}g"
    # Common estimations
    if "bowl" in portion_str: return "~200g"
    if "plate" in portion_str: return "~350g"
    if "slice" in portion_str: return "~80g" # Highly variable (bread vs pizza vs cake)
    if "cup" in portion_str: return "~240g" # Standard cup volume approx for liquids/grains
    if "egg" in portion_str: return "~50g" # Medium/Large egg
    if "serving" in portion_str: return "~150g" # Generic serving size
    # Return original string if no estimate matched
    return str(portion) # Return original if no rule matches

def calculate_calories(age, weight, height, gender, activity, goal):
    """
    Calculates estimated daily caloric needs using Mifflin-St Jeor equation.
    Returns calculated calories as an integer, or None if inputs are invalid.
    """
    try:
        # Input validation
        if not (18 <= age <= 100): raise ValueError("Age must be between 18-100")
        if weight < 30 or height < 100: raise ValueError("Invalid weight/height provided")
        if gender not in ["Male", "Female"]: raise ValueError("Invalid gender provided")

        activity_factors = {
            "Sedentary": 1.2, "Light": 1.375, "Moderate": 1.55,
            "Active": 1.725, "Very Active": 1.9
        }
        if activity not in activity_factors: raise ValueError("Invalid activity level provided")

        # Mifflin-St Jeor Equation for BMR
        if gender == "Male":
            bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
        else: # Female
            bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161

        # Calculate TDEE (Total Daily Energy Expenditure)
        tdee = bmr * activity_factors[activity]

        # Adjust for goal
        if goal == "Lose Weight":
            calculated_calories = tdee - 500
        elif goal == "Gain Muscle":
            calculated_calories = tdee + 500
        else: # Maintain Weight
            calculated_calories = tdee

        # Ensure a minimum calorie intake and round
        final_calories = max(1200, round(calculated_calories))
        log.info(f"Calculated calories: {final_calories} (BMR: {bmr:.0f}, TDEE: {tdee:.0f}, Goal: {goal})")
        return final_calories

    except ValueError as ve:
        log.warning(f"Calorie calculation validation error: {ve}")
        return None # Indicate failure due to invalid input
    except Exception as e:
        log.error(f"Unexpected error during calorie calculation: {e}", exc_info=True)
        return None # Indicate general failure


@st.cache_data
def load_nutrition_data(filepath="data/nutrition.csv"):
    """Loads and cleans the nutrition data CSV file."""
    try:
        log.info(f"Attempting to load nutrition data from: {filepath}")
        df = pd.read_csv(filepath)
        # Basic cleaning: drop rows with any missing values
        df_cleaned = df.dropna().reset_index(drop=True)
        log.info(f"Successfully loaded and cleaned nutrition data. Shape: {df_cleaned.shape}")
        return df_cleaned
    except FileNotFoundError:
        log.error(f"Nutrition data file not found at specified path: {filepath}")
        # Let the calling script (app.py) handle how to inform the user
        return None
    except pd.errors.EmptyDataError:
        log.error(f"Nutrition data file is empty: {filepath}")
        return None
    except Exception as e:
        log.error(f"Failed to load or process nutrition data from {filepath}: {e}", exc_info=True)
        return None

def process_day_content(day_content):
    """Processes the content for a single day to extract meal rows and daily totals."""
    meal_rows = []
    daily_calories = 0
    daily_protein = 0
    daily_carbs = 0
    daily_fat = 0

    for meal_item_key, meal_item_content in day_content.items():
        if isinstance(meal_item_content, dict) and "nutrition" in meal_item_content:
            meal_type = meal_item_key.lower()
            dish_name = meal_item_content.get("dish_name", "N/A")
            portion_grams = estimate_grams(meal_item_content.get("portion_grams", "N/A"))
            nutrition = meal_item_content.get("nutrition", {})
            calories = extract_num(nutrition.get("calories", 0))
            protein = extract_num(nutrition.get("protein", 0))
            carbs = extract_num(nutrition.get("carbs", 0))
            fat = extract_num(nutrition.get("fat", 0))

            daily_calories += calories
            daily_protein += protein
            daily_carbs += carbs
            daily_fat += fat

            if meal_type == "breakfast" or meal_type == "lunch" or meal_type == "dinner":
                meal_rows.append({
                    "Meal": meal_type.capitalize(),
                    "Dish": dish_name,
                    "Portion(g)": portion_grams,
                    "Calories (kcal)": format_number(calories),
                    "Protein (g)": format_number(protein),
                    "Carbs (g)": format_number(carbs),
                    "Fat (g)": format_number(fat),
                })
            elif meal_type.startswith("snack"):
                meal_rows.append({
                    "Meal": "Snack",
                    "Dish": dish_name,
                    "Portion(g)": portion_grams,
                    "Calories (kcal)": format_number(calories),
                    "Protein (g)": format_number(protein),
                    "Carbs (g)": format_number(carbs),
                    "Fat (g)": format_number(fat),
                })
    return meal_rows, daily_calories, daily_protein, daily_carbs, daily_fat