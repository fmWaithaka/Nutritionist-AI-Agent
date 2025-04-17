import requests
import base64
import json
import re
import logging
import pandas as pd
import streamlit as st  # Required because st.error/warning are used directly here

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

from constants import GOOGLE_API_KEY, USDA_API_KEY, USDA_BASE_URL, EXAMPLE_MEAL_STRUCTURE

# Configure logger for this module
log = logging.getLogger(__name__)


if not GOOGLE_API_KEY:
    st.error("❌ Google API key ('google_api_key') not found in .env file")
    st.stop()

if not USDA_API_KEY:
    st.warning("⚠️ USDA API key ('usda_api_key') not found in .env file. Enhanced grounding will be limited.")


def validate_meal_plan_nutrition(meal_plan: dict) -> dict:
    """Cross-check generated nutrition data with USDA database"""
    validation_results = {
        "total_dishes": 0,
        "usda_verified": 0,
        "calorie_discrepancies": [],
        "macro_discrepancies": []
    }
    
    for day, meals in meal_plan.get("meal_plan", {}).items():
        for meal_type in ["breakfast", "lunch", "dinner", "snacks"]:
            meal = meals.get(meal_type, {})
            if not meal.get("dish_name"):
                continue
            
            validation_results["total_dishes"] += 1
            
            # Get USDA data
            usda_data = fetch_nutrition_data_from_usda(meal["dish_name"])
            if not usda_data:
                continue
                
            validation_results["usda_verified"] += 1
            
            # Compare values
            generated = meal.get("nutrition", {})
            discrepancies = {}
            
            for key in ["calories", "protein", "carbs", "fat"]:
                gen_val = generated.get(key, 0)
                usda_val = usda_data.get(key, 0)
                
                if usda_val > 0 and abs(gen_val - usda_val)/usda_val > 0.15:  # 15% threshold
                    discrepancies[key] = {
                        "generated": gen_val,
                        "usda": usda_val,
                        "variance": round((gen_val - usda_val)/usda_val * 100, 1)
                    }
            
            if discrepancies:
                validation_results["calorie_discrepancies"].append({
                    "dish": meal["dish_name"],
                    **discrepancies
                })
    
    return validation_results


def fetch_nutrition_data_from_usda(food_name: str) -> dict:
    """
    Fetches nutrition data for a given food name from the USDA FoodData Central API.
    Returns a dictionary with relevant nutrition information or None if not found or error.
    """
    try:
        params = {
            "api_key": USDA_API_KEY,
            "query": food_name,
            "pageSize": 3,  # Get top 3 for better matching
            "dataType": ["Survey (FNDDS)"]  # Focus on standard reference data
        }
        
        response = requests.get(USDA_BASE_URL, params=params, timeout=15)
        response.raise_for_status()
        
        best_match = None
        best_score = 0
        
        for food in response.json().get("foods", []):
            # Use fuzzy matching to find best name match
            score = fuzz.ratio(food_name.lower(), food["description"].lower())
            if score > best_score:
                best_match = food
                best_score = score
                
        if best_score < 65:  # Only use good matches
            return None
            
        # Extract nutrients with validation
        nutrients = {
            "calories": get_nutrient_value(best_match, "Energy"),
            "protein": get_nutrient_value(best_match, "Protein"),
            "carbs": get_nutrient_value(best_match, "Carbohydrate, by difference"),
            "fat": get_nutrient_value(best_match, "Total lipid (fat)")
        }
        
        # Validate required fields
        if all(v > 0 for v in nutrients.values()):
            return nutrients
            
    except Exception as e:
        log.error(f"USDA fetch error: {str(e)}")
        return None

def get_nutrient_value(food_data: dict, nutrient_name: str) -> float:
    """Safe nutrient value extraction"""
    return next(
        (n["value"] for n in food_data.get("foodNutrients", []) 
         if n.get("nutrientName") == nutrient_name and n.get("unitName") == "kcal"),
        0.0
    )

def analyze_image_with_rest(api_key: str, image_bytes: bytes, language: str = "English"):
    """
    Sends image bytes to Gemini Vision REST API for analysis using requests.
    Returns a dictionary with analysis data or None on failure.
    """
    log.info(f"Entering analyze_image_with_rest for language: {language}")
    if not api_key:
        log.error("API key is missing for analyze_image_with_rest.")
        st.error("Configuration error: API Key not provided.")
        return None
    if not image_bytes:
        log.warning("No image bytes provided for analysis.")
        # Keep user warning
        st.warning("No image bytes provided for analysis.")
        return None

    try:
        img_base64 = base64.b64encode(image_bytes).decode("utf-8")
        # Using gemini-pro-vision as it's generally preferred for vision tasks now,
        # but you can switch back to gemini-2.0-flash if needed.
        # Check latest model availability/recommendations if unsure.
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GOOGLE_API_KEY}"

        # Prompt asking for specific JSON structure
        prompt = (
            f"Analyze this food image precisely. Respond ONLY with a valid JSON object "
            f"(no extra text or markdown formatting) like this: "
            f"{{\"food\": \"Best guess name\", \"estimated_calories\": <number>, "
            f"\"macros\": {{\"protein\": <number>, \"carbs\": <number>, \"fat\": <number>}}, "
            # Include micros if desired
            # f"\"micros\": {{\"fiber\": <number>, \"sugar\": <number>, \"sodium\": <number>}}, "
            f"\"portion_grams\": <number>}}. "
            f"Ensure all numeric values are numbers, not strings. "
            f"Respond in {language} for the 'food' name if possible, keep keys in English."
        )

        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": img_base64}}
                ]
            }]
            # Add generationConfig if needed (e.g., temperature)
            # "generationConfig": { "temperature": 0.4 }
        }

        log.info(f"Calling Vision API: {api_url}")
        response = requests.post(
            api_url, headers=headers, json=payload, timeout=60)
        log.info(f"Vision API Status Code: {response.status_code}")
        log.info(
            f"Vision API Response Text (first 500): {response.text[:500]}")

        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        result_json = response.json()

        # Check response structure based on Gemini API docs
        if not result_json.get("candidates"):
            log.error("Vision Error: No 'candidates' found in API response.")
            st.error(
                "❌ Image Analysis Error: No 'candidates' found in API response.")
            log.error("Full API Response: %s", result_json)
            return None

        try:
            # Ensure the path to the text is correct
            if not result_json["candidates"][0].get("content") or not result_json["candidates"][0]["content"].get("parts"):
                log.error(
                    "Vision Error: Unexpected content/parts structure in response.")
                st.error("❌ Image Analysis Error: Unexpected response structure.")
                log.error("Full API Response: %s", result_json)
                return None

            text_result = result_json["candidates"][0]["content"]["parts"][0]["text"]
            log.info(
                f"Vision API Extracted Text (first 500): {text_result[:500]}")
            if text_result:
                log_entry = {
                    "timestamp": pd.Timestamp.now(tz='UTC').isoformat(),
                    "function_called": "analyze_image",
                    "input_context": {
                        "language": language,
                        "image_size": len(image_bytes)
                    },
                    "raw_response_text": text_result
                }
                try:
                    with open("api_log.jsonl", "a", encoding="utf-8") as f:
                        json.dump(log_entry, f)
                        f.write("\n")
                    log.info("Logged image analysis request/response")
                except Exception as log_e:
                    log.error(f"Failed to log image analysis: {log_e}")

        except (KeyError, IndexError, TypeError) as e:
            log.error(
                f"Vision Error: Could not extract text part from API response: {e}")
            st.error(
                f"❌ Image Analysis Error: Could not process API response structure: {e}")
            log.error("Full API Response: %s", result_json)
            return None

        # Parse JSON from the text result (more robust regex)
        match = re.search(
            r"```json\s*(\{.*?\})\s*```", text_result, re.DOTALL | re.IGNORECASE)
        if not match:
            # Assume whole output is JSON if no markdown
            match = re.search(r"(\{.*?\})", text_result.strip(), re.DOTALL)

        if not match:
            log.error(
                "Vision Error: Could not parse/find JSON block in response text.")
            st.error(
                "⚠️ Image Analysis Error: Could not find expected JSON data in AI response.")
            return None

        try:
            json_str = match.group(1)
            log.info(f"Vision API Found JSON block: {json_str}")
            analysis_data = json.loads(json_str)
            log.info("Vision analysis JSON decoded successfully.")

            # --- Fetch and add nutrition data from USDA ---
            food_name = analysis_data.get("food")
            if food_name:
                usda_nutrition = fetch_nutrition_data_from_usda(food_name)
                if usda_nutrition:
                    # Calculate portion-adjusted values
                    portion_grams = analysis_data.get("portion_grams", 100)
                    analysis_data["verified_nutrition"] = {
                        "calories": (usda_nutrition["calories"] / 100) * portion_grams,
                        "protein": (usda_nutrition["protein"] / 100) * portion_grams,
                        "carbs": (usda_nutrition["carbs"] / 100) * portion_grams,
                        "fat": (usda_nutrition["fat"] / 100) * portion_grams
                    }
                    analysis_data["data_source"] = "USDA"
                else:
                    analysis_data["data_source"] = "AI Estimate"

            return analysis_data
        except json.JSONDecodeError as e:
            log.error(f"Vision Error: Failed to decode JSON: {e}")
            log.error(f"Problematic JSON string: {json_str}")
            st.error(
                f"⚠️ Image Analysis Error: Failed to decode AI response data: {e}")
            return None

    except requests.exceptions.RequestException as e:
        log.error(f"API Request Error (Vision): {e}")
        st.error(
            f"❌ Network Error: Failed to connect to Image Analysis service ({e})")
        return None
    except Exception as e:
        # Logs traceback
        log.exception("Unexpected error during image analysis.")
        st.error(f"❌ An unexpected error occurred during image analysis: {e}")
        return None
    finally:
        log.info("Exiting analyze_image_with_rest")


def generate_meal_plan_with_rest(api_key: str, calorie_target: int, preferences: dict, language: str = "English"):
    """
    Generates a 7-day meal plan dictionary using Gemini Text REST API.
    Returns a dictionary { "day1": {...}, ... } or None on failure.
    """
    log.info(
        f"Entering generate_meal_plan_with_rest for {calorie_target} kcal, lang: {language}")
    if not api_key:
        log.error("API key is missing for generate_meal_plan_with_rest.")
        st.error("Configuration error: API Key not provided.")
        return None
    if not calorie_target or not preferences:
        log.warning("Missing calorie target or preferences for meal plan.")
        # No st.warning here, handled in app.py before calling
        return None

    try:
        # Using gemini-pro as it's generally better for complex JSON generation than flash
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GOOGLE_API_KEY}"

        restrictions_str = ', '.join(
            preferences.get('restrictions', [])) or 'None'
        favorites_str = preferences.get('favorites', 'Any')
        dislikes_str = preferences.get('dislikes', 'None')

        # Explicit prompt asking ONLY for JSON with 'meal_plan' key containing a dict
        meal_prompt = (
            f"You are a nutritionist AI assistant. The user needs **{calorie_target} kcal/day**.\n"
            f"Generate a JSON meal plan for 7 days ONLY.\n\n"
            f"User Preferences:\n"
            f"- Goal: {preferences.get('goal', 'Maintain Weight')}\n"
            f"- Diet/Restrictions: {restrictions_str}\n"
            f"- Favorite Foods: {favorites_str}\n"
            f"- Disliked Foods: {dislikes_str}\n\n"
            f"Critical Requirements:\n"
            f"1. Use COMMON, WELL-KNOWN FOOD ITEMS from standard nutritional databases\n"
            f"2. For each meal's nutrition data:\n"
            f"   a) First check USDA FoodData Central database values using its API\n"
            f"   b) Clearly note if values are estimates with 'estimated_' prefix\n"
            f"3. Format nutrition values as NUMBERS ONLY (no units)\n"
            f"4. Ensure portion sizes use STANDARD METRIC measurements (grams)\n"
            f"5. Include a 'data_source': 'USDA' field for each meal to indicate the source of nutrition data.\n"
            f"**6. For each day, include 1-2 realistic snack options. A typical snack should be between 100-300 kcal.**\n" # Added constraint
            f"Response MUST be valid JSON with 'meal_plan' key following this structure:\n"
            f"{EXAMPLE_MEAL_STRUCTURE}"
        )

        log.info("Sending Meal Plan Prompt (first 300 chars):\n%s",
                 meal_prompt[:300] + "...")

        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"role": "user", "parts": [{"text": meal_prompt}]}],
            "generationConfig": {"temperature": 0.6}
        }

        log.info(f"Calling Text API for meal plan: {api_url}")
        response = requests.post(
            api_url, headers=headers, json=payload, timeout=180)  # Increased timeout
        log.info(f"Meal Plan API Status Code: {response.status_code}")
        log.info(
            f"Meal Plan API Response Text (first 500): {response.text[:500]}")

        response.raise_for_status()

        result_json = response.json()
        if not result_json.get("candidates"):
            log.error(
                "Meal Plan Error: No 'candidates' found in API response JSON.")
            st.error("❌ Meal Plan Error: No 'candidates' found in AI response.")
            log.error("Full API Response: %s", result_json)
            return None

        try:
            if not result_json["candidates"][0].get("content") or not result_json["candidates"][0]["content"].get("parts"):
                log.error(
                    "Meal Plan Error: Unexpected content/parts structure in response.")
                st.error(
                    "❌ Meal Plan Error: Unexpected response structure from AI.")
                log.error("Full API Response: %s", result_json)
                return None
            text_result = result_json["candidates"][0]["content"]["parts"][0]["text"]
            log.info(
                f"Meal Plan Extracted Text (first 500): {text_result[:500]}")

            # --- ***LOGGING FOR EVALUATION*** ---
            if text_result:  # Only log if we got text back
                log_entry = {
                    # Use timezone-aware timestamp
                    "timestamp": pd.Timestamp.now(tz='UTC').isoformat(),
                    "function_called": "generate_meal_plan",
                    "input_context": {  # Log key inputs used for the prompt
                        "calorie_target": calorie_target,
                        "preferences": preferences,
                        "language": language
                    },
                    # "prompt_sent": meal_prompt, # Can be very long, log optionally or truncated
                    "raw_response_text": text_result  # Log the raw text for evaluation
                }
                try:
                    # Append to the log file (JSON Lines format)
                    with open("api_log.jsonl", "a", encoding="utf-8") as f:
                        json.dump(log_entry, f)  # Write JSON object
                        f.write("\n")  # Add newline to separate entries
                    log.info(
                        "Successfully logged meal plan request/response to api_log.jsonl")
                except Exception as log_e:
                    log.error(
                        f"Failed to write to evaluation log file api_log.jsonl: {log_e}")
            # --- *** END LOGGING FOR EVALUATION *** ---

        except (KeyError, IndexError, TypeError) as e:
            log.error(f"Meal Plan Error: Could not extract text part: {e}")
            st.error(
                f"❌ Meal Plan Error: Could not process AI response structure: {e}")
            log.error("Full API Response: %s", result_json)
            return None

        # Attempt to find JSON block (checking for ```json fence first)
        match = re.search(
            r"```json\s*(\{.*?\})\s*```", text_result, re.DOTALL | re.IGNORECASE)
        if not match:
            # Fallback: Assume the entire text is the JSON object
            if text_result.strip().startswith('{') and text_result.strip().endswith('}'):
                match = re.search(r"(\{.*?\})", text_result.strip(), re.DOTALL)

        if not match:
            log.error(
                "Could not parse/find JSON block within the meal plan text response.")
            st.error(
                "⚠️ Meal Plan Error: Could not find expected JSON data in AI response.")
            return None

        try:
            json_str = match.group(1)
            log.info(f"Meal Plan Found JSON block: {json_str[:300]}...")
            meal_data = json.loads(json_str)
            log.info("Meal plan JSON decoded successfully.")

            if "meal_plan" not in meal_data:
                log.error(
                    "Meal Plan Error: Decoded JSON missing 'meal_plan' key.")
                st.error(
                    "❌ Meal Plan Error: AI response missing 'meal_plan' data.")
                log.error("Structure of decoded JSON: %s", meal_data)
                return None

            final_plan_data = meal_data.get("meal_plan")

            if final_plan_data and (isinstance(final_plan_data, dict) or isinstance(final_plan_data, list)):
                data_type = "dictionary" if isinstance(
                    final_plan_data, dict) else "list"
                log.info(
                    f"Successfully extracted meal plan {data_type} with {len(final_plan_data)} entries.")
                return final_plan_data
            else:
                # If it's neither or is empty, something is wrong
                log.error(
                    "Meal Plan Error: Value under 'meal_plan' is not a non-empty list or dictionary (type: %s).", type(final_plan_data))
                st.error(
                    f"❌ Meal Plan Error: AI returned unexpected data format for meal plan (expected List or Dict, got {type(final_plan_data)}).")
                log.error("Full Decoded JSON: %s", meal_data)
                return None

        except json.JSONDecodeError as e:
            log.error(f"Failed to decode meal plan JSON block: {e}")
            log.error(f"Problematic JSON string: {json_str[:500]}...")
            st.error(
                f"⚠️ Meal Plan Error: Failed to decode AI response data: {e}")
            return None

    except requests.exceptions.RequestException as e:
        log.error(f"API Request Error (Meal Plan): {e}")
        st.error(
            f"❌ Network Error: Failed to connect to Meal Plan service ({e})")
        return None
    except Exception as e:
        log.exception("Unexpected error during meal plan generation.")
        st.error(
            f"❌ An unexpected error occurred during meal plan generation: {e}")
        return None
    finally:
        log.info("Exiting generate_meal_plan_with_rest")


def generate_grocery_list_with_rest(api_key: str, meal_plan_dict: dict, language: str = "English"):
    """
    Generates a grocery list string using Gemini Text REST API based on dish names.
    Returns a Markdown string or None on failure.
    """
    log.info("Entering generate_grocery_list_with_rest")
    if not api_key:
        log.error("API key is missing for generate_grocery_list_with_rest.")
        st.error("Configuration error: API Key not provided.")
        return None
    if not meal_plan_dict or not isinstance(meal_plan_dict, dict):
        log.warning(
            "Invalid or empty meal_plan_dict provided for grocery list.")
        return None  # No st.warning needed here, handled in button click

    all_dishes = []
    try:
        # Extract dish names (including handling complex snacks)
        for day_key, day_content in meal_plan_dict.items():
            if not isinstance(day_content, dict):
                continue
            for meal_type in ["breakfast", "lunch", "dinner", "snacks"]:
                info = day_content.get(meal_type)
                if isinstance(info, dict):
                    suffix = ""
                    i = 1
                    while True:
                        dish_key = f"dish_name{suffix}"
                        if dish_key in info:
                            dish_name = info.get(dish_key)
                            if dish_name and isinstance(dish_name, str) and dish_name.strip():
                                all_dishes.append(dish_name.strip())
                            i += 1
                            suffix = str(i)
                        elif suffix == "" and "dish_name" in info:
                            break
                        elif suffix != "":
                            break
                        else:
                            break
                elif isinstance(info, list):
                    for item in info:
                        if isinstance(item, dict) and "dish_name" in item:
                            dish_name = item.get("dish_name")
                            if dish_name and isinstance(dish_name, str) and dish_name.strip():
                                all_dishes.append(dish_name.strip())

        if not all_dishes:
            log.warning(
                "No dish names extracted from meal plan for grocery list.")
            # User feedback
            st.warning(
                "No dish names found in the plan to create a grocery list.")
            return None

        unique_dishes = sorted(list(set(filter(None, all_dishes))))
        if not unique_dishes:
            log.warning("Filtered dish list for grocery list is empty.")
            st.warning("No valid dish names found to generate grocery list.")
            return None

        dishes_text = ", ".join(unique_dishes)
        log.info(
            f"Generating grocery list for {len(unique_dishes)} unique dishes: {dishes_text[:200]}...")

        # Construct the Prompt for Markdown list
        prompt = f"""Act as a helpful shopping assistant. Based *only* on the following list of meal dishes planned for a week, generate a likely grocery list of ingredients needed.

                Dishes Planned:
                {dishes_text}

                Instructions for Grocery List:
                - List necessary ingredients to make these dishes in {language}.
                - Combine similar items (e.g., list 'onion' once). Estimate reasonable weekly quantities for one person (e.g., "Onions: 2-3", "Chicken Breast: 1.5 lbs / 700g"). If unsure of quantity, just list the item.
                - Group ingredients into logical categories using Markdown H3 headings (### Category Name). Common categories: Produce, Meat & Poultry, Fish & Seafood, Dairy & Eggs, Pantry Staples, Frozen, Spices & Oils.
                - Exclude: salt, black pepper, water, basic vegetable/canola oil (unless a specific type like olive oil or sesame oil is clearly needed).
                - Format the output *only* as a Markdown list with bullet points (*) under category headings.
                - Do NOT include introductory or concluding sentences. Just the list.
                """

        # Make API Call (can use flash for this less complex task)
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GOOGLE_API_KEY}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3}  # More deterministic list
        }

        log.info("Making API call for grocery list...")
        response = requests.post(
            api_url, headers=headers, json=payload, timeout=90)
        log.info(f"Grocery List API Status Code: {response.status_code}")
        log.info(
            f"Grocery List Raw Response Text (first 500): {response.text[:500]}")

        response.raise_for_status()

        result_json = response.json()
        if not result_json.get("candidates"):
            log.error("Grocery List Error: No 'candidates' in response.")
            st.error(
                "❌ Grocery List Error: AI service did not provide a valid response.")
            return None

        # Extract text
        try:
            if not result_json["candidates"][0].get("content") or not result_json["candidates"][0]["content"].get("parts"):
                log.error(
                    "Grocery List Error: Unexpected content/parts structure in response.")
                st.error(
                    "❌ Grocery List Error: Unexpected response structure from AI.")
                log.error("Full API Response: %s", result_json)
                return None

            grocery_list_text = result_json["candidates"][0]["content"]["parts"][0]["text"]
            log.info("Successfully extracted grocery list text.")

            if grocery_list_text:
                log_entry = {
                    "timestamp": pd.Timestamp.now(tz='UTC').isoformat(),
                    "function_called": "generate_grocery_list",
                    "input_context": {
                        "language": language,
                        "meal_plan_keys": list(meal_plan_dict.keys())
                    },
                    "raw_response_text": grocery_list_text
                }
                try:
                    with open("api_log.jsonl", "a", encoding="utf-8") as f:
                        json.dump(log_entry, f)
                        f.write("\n")
                    log.info("Logged grocery list request/response")
                except Exception as log_e:
                    log.error(f"Failed to log grocery list: {log_e}")

            grocery_list_text = re.sub(
                r"^```markdown\s*\n?", "", grocery_list_text, flags=re.IGNORECASE | re.MULTILINE)
            grocery_list_text = re.sub(
                r"\n?```\s*$", "", grocery_list_text, flags=re.IGNORECASE | re.MULTILINE)
            return grocery_list_text.strip()

        except (KeyError, IndexError, TypeError) as e:
            log.error(f"Grocery List Error: Could not extract text part: {e}")
            st.error("❌ Grocery List Error: Could not process response from AI.")
            log.error("Full API Response: %s", result_json)
            return None

    except requests.exceptions.RequestException as e:
        log.error(f"API Request Error (Grocery List): {e}")
        st.error(f"❌ Network Error connecting to AI for grocery list ({e})")
        return None
    except Exception as e:
        log.exception("Unexpected error during grocery list generation.")
        st.error(f"❌ Unexpected error creating grocery list: {e}")
        return None
    finally:
        log.info("Exiting generate_grocery_list_with_rest")