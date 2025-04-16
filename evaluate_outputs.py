# evaluate_outputs.py
import json
import os
from dotenv import load_dotenv
import requests
import logging
from datetime import datetime

# Configure logging to both console and file
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
log.addHandler(console_handler)

# File handler
file_handler = logging.FileHandler('evaluate_logs')
file_handler.setFormatter(log_formatter)
log.addHandler(file_handler)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    log.error("GOOGLE_API_KEY not found.")
    exit("API Key not found in .env file. Evaluation cannot proceed.")

# Define API endpoint (using gemini-1.5-flash-latest for potentially cheaper/faster evaluation)
EVALUATION_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GOOGLE_API_KEY}"

def call_gemini_api(prompt):
    """Calls the Gemini API with the given prompt."""
    headers = {
        'Content-Type': 'application/json',
    }
    data = {
        'contents': [{
            'parts': [{'text': prompt}]
        }]
    }
    try:
        response = requests.post(EVALUATION_API_URL, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        log.error(f"Error calling Gemini API: {e}")
        return None

def parse_gemini_response(response_json):
    """Extracts and cleans the evaluation text from Gemini's response."""
    if response_json and 'candidates' in response_json and response_json['candidates']:
        evaluation_text = response_json['candidates'][0]['content']['parts'][0]['text']
        # Remove markdown code block formatting if present
        evaluation_text = evaluation_text.strip().replace('```json', '').replace('```', '').strip()
        return evaluation_text
    return None

def evaluate_analyze_image(log_entry):
    """Evaluates the output of the analyze_image function."""
    input_context = log_entry.get("input_context", {})
    raw_response_text = log_entry.get("raw_response_text", "")

    prompt = f"""You are an expert AI evaluator. Given the following context and the raw response from an AI model, please evaluate the response.

Context:
Language: {input_context.get("language")}
Image Size: {input_context.get("image_size")}

Raw Response:
{raw_response_text}

Evaluate the raw response based on the following criteria:
1. **JSON Validity:** Is the raw response a valid JSON format?
2. **Reasonableness of Food Item:** Does the identified food item seem plausible given the potential context of an image analysis? Provide a brief justification.
3. **Reasonableness of Calorie Estimate:** Is the estimated calorie count reasonable for the identified food item? Provide a brief justification.
4. **Completeness of Macros:** Does the 'macros' section include protein, carbs, and fat?
5. **Reasonableness of Macros:** Are the provided macro values (protein, carbs, fat) reasonable in relation to the estimated calories for the identified food item? Provide a brief justification.

Provide your evaluation in a JSON format with the following keys: 'json_valid', 'food_item_reasonableness', 'calorie_reasonableness', 'macros_complete', 'macros_reasonableness', 'justification'. Assign a score from 1 to 5 (1=Very Poor, 5=Excellent) for each criterion (except 'json_valid' which should be a boolean)."""

    evaluation_response_json = call_gemini_api(prompt)
    evaluation_text = parse_gemini_response(evaluation_response_json)

    if evaluation_text:
        try:
            evaluation = json.loads(evaluation_text)
            return evaluation
        except json.JSONDecodeError:
            log.error(f"Failed to decode JSON from evaluation response for analyze_image: {evaluation_text}")
            return {"error": "Failed to decode evaluation JSON"}
    else:
        return {"error": "No valid evaluation response received from Gemini API"}

def evaluate_generate_meal_plan(log_entry):
    """Evaluates the output of the generate_meal_plan function for the first day."""
    input_context = log_entry.get("input_context", {})
    raw_response_text = log_entry.get("raw_response_text", "")

    try:
        meal_plan_data = json.loads(raw_response_text.strip('```json\n').strip('```'))
        day1_plan = meal_plan_data.get("meal_plan", {}).get("day1", {})
        preferences = input_context.get("preferences", {})
        calorie_target = input_context.get("calorie_target")

        prompt = f"""You are an expert AI evaluator. Given the following user preferences, calorie target, and the generated meal plan for day 1, please evaluate the meal plan.

User Preferences:
Goal: {preferences.get("goal")}
Restrictions: {preferences.get("restrictions")}
Favorites: {preferences.get("favorites")}
Dislikes: {preferences.get("dislikes")}

Calorie Target: {calorie_target}

Meal Plan for Day 1:
{json.dumps(day1_plan, indent=2)}

Evaluate the meal plan based on the following criteria:
1. **Adherence to Restrictions:** Does the meal plan for day 1 strictly adhere to the user's dietary restrictions (e.g., Vegetarian, Dairy-Free)?
2. **Inclusion of Favorites:** Does the meal plan for day 1 include any of the user's favorite foods?
3. **Exclusion of Dislikes:** Does the meal plan for day 1 exclude the user's disliked foods?
4. **Reasonableness of Calorie Distribution:** Is the distribution of calories across breakfast, lunch, dinner, and snacks reasonable for a full day?
5. **Variety of Meals:** Does the meal plan for day 1 offer a reasonable variety of different dishes?
6. **Estimated Daily Calories:** Is the 'daily_nutrition' calorie count for day 1 reasonably close to the user's calorie target?

Provide your evaluation in a JSON format with the following keys: 'adherence_to_restrictions', 'inclusion_of_favorites', 'exclusion_of_dislikes', 'calorie_distribution_reasonableness', 'meal_variety', 'daily_calories_alignment', 'justification'. Assign a score from 1 to 5 (1=Very Poor, 5=Excellent) for each criterion."""

        evaluation_response_json = call_gemini_api(prompt)
        evaluation_text = parse_gemini_response(evaluation_response_json)

        if evaluation_text:
            try:
                evaluation = json.loads(evaluation_text)
                return evaluation
            except json.JSONDecodeError:
                log.error(f"Failed to decode JSON from evaluation response for generate_meal_plan: {evaluation_text}")
                return {"error": "Failed to decode evaluation JSON"}
        else:
            return {"error": "No valid evaluation response received from Gemini API"}

    except json.JSONDecodeError:
        log.error(f"Failed to decode JSON from raw response for generate_meal_plan: {raw_response_text}")
        return {"error": "Failed to decode raw response JSON"}

def evaluate_generate_grocery_list(log_entry):
    """Evaluates the output of the generate_grocery_list function."""
    raw_response_text = log_entry.get("raw_response_text", "")

    prompt = f"""You are an expert AI evaluator. Given the following generated grocery list, please evaluate its quality.

Grocery List:
{raw_response_text}

Evaluate the grocery list based on the following criteria:
1. **Comprehensiveness:** Does the grocery list seem to include a reasonable range of ingredients that would likely be needed for a typical 7-day meal plan (assuming a diverse set of meals)?
2. **Organization:** Is the grocery list well-organized into logical categories (e.g., Produce, Pantry Staples)?
3. **Clarity:** Are the items in the list generally clear and easy to understand?
4. **Absence of Redundancy:** Does the list avoid excessive repetition of similar items?

Provide your evaluation in a JSON format with the following keys: 'comprehensiveness', 'organization', 'clarity', 'absence_of_redundancy', 'justification'. Assign a score from 1 to 5 (1=Very Poor, 5=Excellent) for each criterion."""

    evaluation_response_json = call_gemini_api(prompt)
    evaluation_text = parse_gemini_response(evaluation_response_json)

    if evaluation_text:
        try:
            evaluation = json.loads(evaluation_text)
            return evaluation
        except json.JSONDecodeError:
            log.error(f"Failed to decode JSON from evaluation response for generate_grocery_list: {evaluation_text}")
            return {"error": "Failed to decode evaluation JSON"}
    else:
        return {"error": "No valid evaluation response received from Gemini API"}

def main():
    """Reads the log file, evaluates outputs, and records results."""
    log_file_path = "api_log.jsonl"
    evaluation_results_file = "evaluation_results.jsonl"

    if not os.path.exists(log_file_path):
        log.error(f"Log file not found: {log_file_path}")
        return

    with open(log_file_path, 'r') as infile, open(evaluation_results_file, 'w') as outfile:
        for line in infile:
            try:
                log_entry = json.loads(line.strip())
                function_called = log_entry.get("function_called")
                timestamp = log_entry.get("timestamp")

                log.info(f"Evaluating entry for function: {function_called} at timestamp: {timestamp}")

                evaluation_result = {"timestamp": timestamp, "function_called": function_called, "original_log": log_entry}

                if function_called == "analyze_image":
                    evaluation = evaluate_analyze_image(log_entry)
                    evaluation_result["evaluation"] = evaluation
                elif function_called == "generate_meal_plan":
                    evaluation = evaluate_generate_meal_plan(log_entry)
                    evaluation_result["evaluation"] = evaluation
                elif function_called == "generate_grocery_list":
                    evaluation = evaluate_generate_grocery_list(log_entry)
                    evaluation_result["evaluation"] = evaluation
                else:
                    evaluation_result["evaluation"] = {"error": f"No evaluation function defined for '{function_called}'"}
                    log.warning(f"No evaluation function defined for '{function_called}'")

                outfile.write(json.dumps(evaluation_result) + '\n')
                log.info(f"Evaluation recorded for {function_called}")

            except json.JSONDecodeError:
                log.error(f"Could not decode JSON from line: {line.strip()}")
            except Exception as e:
                log.error(f"An unexpected error occurred: {e}")

    log.info(f"Evaluation process completed. Results saved to {evaluation_results_file}")

if __name__ == "__main__":
    main()