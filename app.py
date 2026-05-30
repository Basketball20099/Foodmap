from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import requests
import json
import os
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration - Replace with your actual OpenAI API key
DATASET_PATH = "Indian_FoodMap_Cleaned_v1.xlsx"

# Load the dataset
try:
    df = pd.read_excel(DATASET_PATH)
    print(f"Successfully loaded dataset with {len(df)} food items")
except Exception as e:
    print(f"Error loading dataset: {e}")
    df = None

# Cache for storing previous responses to identical queries
response_cache = {}

def query_openai(user_question, context):
    """
    Query the OpenAI API with the user's question and relevant food context
    
    Args:
        user_question (str): The user's question
        context (str): Context from the food database
        
    Returns:
        str: OpenAI's response
    """
    # Check cache first
    cache_key = f"{user_question}_{hash(context)}"
    if cache_key in response_cache:
        print("Using cached response")
        return response_cache[cache_key]
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    # Create system prompt
    system_prompt = """
    You are a nutrition expert specializing in Indian cuisine. Your role is to provide accurate nutrition information 
    about Indian foods based on the Indian FoodMap dataset. Answer questions in a friendly, concise manner.
    
    When answering:
    1. Focus only on information present in the dataset
    2. If specific nutrition data isn't available, say so
    3. Present information in an easy-to-understand format
    4. Suggest healthy alternatives when relevant
    5. Keep your answers brief but informative
    6. When providing nutrition data, use HTML formatting for better readability
    
    Remember, you have access to comprehensive nutritional information about Indian foods,
    including calories, protein, carbs, fats, cholesterol, vitamins, minerals and more.
    """
    
    # OpenAI API payload format
    payload = {
        "model": "gpt-4",  # You can use "gpt-3.5-turbo" for cost efficiency if preferred
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"""
                I need information about Indian food nutrition. Please use the following data from the Indian FoodMap dataset:
                
                {context}
                
                My question is: {user_question}
                """
            }
        ],
        "max_tokens": 1000,
        "temperature": 0.2
    }
    
    try:
        print(f"Sending request to OpenAI API")
        start_time = time.time()
        response = requests.post(
            OPENAI_API_URL, 
            headers=headers, 
            json=payload, 
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        print(f"Received response from OpenAI API in {time.time() - start_time:.2f} seconds")
        
        # OpenAI API response format
        answer = result["choices"][0]["message"]["content"]
        
        # Cache the response
        response_cache[cache_key] = answer
        
        return answer
    except Exception as e:
        print(f"Error querying OpenAI API: {e}")
        print(f"Full error details: {str(e)}")
        try:
            if response:
                print(f"Response status code: {response.status_code}")
                print(f"Response content: {response.text}")
        except:
            pass
        return "I'm having trouble accessing my nutrition database. Please try again later."

def get_relevant_context(query):
    """
    Extract relevant food data as context based on user query
    
    Args:
        query (str): The user's question
        
    Returns:
        str: JSON string containing relevant food data
    """
    if df is None:
        return json.dumps({"error": "Dataset not loaded"})
    
    query = query.lower()
    # Filter out non-vegetarian foods (meat, fish, poultry) if query includes 'vegetarian'
    df_filtered = df
    if 'vegetarian' in query:
        non_veg_keywords = ["chicken", "mutton", "fish", "egg", "meat", "pork", "beef", "prawn", "shrimp", "goat", "lamb", "turkey", "duck", "bacon"]
        pattern = '|'.join(non_veg_keywords)
        df_filtered = df[~df['food_name'].str.lower().str.contains(pattern)]
    relevant_foods = []
    food_names_mentioned = []
    
    # Match specific food names mentioned in the query
    for food_name in df_filtered['food_name'].unique():
        if food_name.lower() in query:
            food_info = df_filtered[df_filtered['food_name'] == food_name].iloc[0].to_dict()
            relevant_foods.append(clean_food_data(food_info))
            food_names_mentioned.append(food_name.lower())
    
    # Extract keywords from the query
    keywords = {
        "protein": {"column": "protein_g", "threshold": 15, "desc": "high protein foods"},
        "iron": {"column": "iron_mg", "threshold": 3, "desc": "iron-rich foods"},
        "calcium": {"column": "calcium_mg", "threshold": 200, "desc": "high calcium foods"},
        "cholesterol": {"column": "cholesterol_mg", "threshold": 0, "desc": "cholesterol content"},
        "fiber": {"column": "fibre_g", "threshold": 5, "desc": "high fiber foods"},
        "fibre": {"column": "fibre_g", "threshold": 5, "desc": "high fiber foods"},
        "calorie": {"column": "energy_kcal", "threshold": 150, "desc": "calorie content"},
        "calories": {"column": "energy_kcal", "threshold": 150, "desc": "calorie content"},
        "vitamin c": {"column": "vitc_mg", "threshold": 30, "desc": "vitamin C content"},
        "zinc": {"column": "zinc_mg", "threshold": 2, "desc": "zinc content"},
    }
    
    # Check food categories
    food_categories = {
        "breakfast": ["breakfast", "idli", "dosa", "upma", "poha", "paratha"],
        "dessert": ["sweet", "dessert", "halwa", "ladoo", "burfi", "kheer"],
        "snack": ["snack", "chaat", "pakora", "samosa", "tikki"],
        "curry": ["curry", "sabji", "dal", "sambar"],
        "biryani": ["biryani", "pulao", "rice"],
        "bread": ["roti", "naan", "paratha", "chapati"]
    }
    
    matched_categories = []
    for category, terms in food_categories.items():
        if any(term in query for term in terms):
            matched_categories.append(category)
    
    # Process health-related queries
    health_queries = {
        "weight loss": {"column": "energy_kcal", "operation": "low", "desc": "low calorie foods for weight management"},
        "diabetic": {"column": "freesugar_g", "operation": "low", "desc": "foods suitable for diabetics"},
        "heart": {"column": "cholesterol_mg", "operation": "low", "desc": "heart-healthy foods"},
        "pregnancy": {"column": "iron_mg", "operation": "high", "desc": "foods recommended during pregnancy"},
        "muscle": {"column": "protein_g", "operation": "high", "desc": "foods for muscle building"},
        "vegetarian": {"column": None, "operation": "filter", "desc": "vegetarian options"}
    }
    
    matched_health_queries = []
    for term, details in health_queries.items():
        if term in query:
            matched_health_queries.append(details["desc"])
    
    # Collect relevant food data based on matched criteria
    added_foods = set()  # Track foods we've already added
    
    # Add foods based on keywords
    for keyword, details in keywords.items():
        if keyword in query:
            # Get top 5 foods for this nutrient
            filtered_foods = df_filtered.sort_values(by=details["column"], ascending=False).head(5)
            for _, food in filtered_foods.iterrows():
                if food['food_name'] not in added_foods:
                    relevant_foods.append(clean_food_data(food.to_dict()))
                    added_foods.add(food['food_name'])
    
    # Add foods based on categories
    for category in matched_categories:
        category_foods = []
        for term in food_categories[category]:
            matched_foods = df_filtered[df_filtered['food_name'].str.lower().str.contains(term)]
            if not matched_foods.empty:
                category_foods.extend(matched_foods.head(3).to_dict('records'))
        
        for food in category_foods:
            if food['food_name'] not in added_foods:
                relevant_foods.append(clean_food_data(food))
                added_foods.add(food['food_name'])
    
    # Add foods based on health queries
    for query_desc in matched_health_queries:
        for term, details in health_queries.items():
            if details["desc"] == query_desc and details["column"] is not None:
                if details["operation"] == "low":
                    filtered_foods = df_filtered.sort_values(by=details["column"], ascending=True).head(5)
                else:  # high
                    filtered_foods = df_filtered.sort_values(by=details["column"], ascending=False).head(5)
                
                for _, food in filtered_foods.iterrows():
                    if food['food_name'] not in added_foods:
                        relevant_foods.append(clean_food_data(food.to_dict()))
                        added_foods.add(food['food_name'])
    
    # If we still don't have relevant foods, add some general recommendations
    if not relevant_foods:
        # Add some popular foods
        popular_foods = df_filtered.sample(5).to_dict('records')
        for food in popular_foods:
            relevant_foods.append(clean_food_data(food))
    
    # Add metadata to help understand the context
    metadata = {
        "query": query,
        "food_names_mentioned": food_names_mentioned,
        "matched_categories": matched_categories,
        "health_related_queries": matched_health_queries,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    full_context = {
        "metadata": metadata,
        "foods": relevant_foods
    }
    
    return json.dumps(full_context, indent=2)

def clean_food_data(food_dict):
    """
    Clean the food data to remove extraneous information and improve readability
    
    Args:
        food_dict (dict): Dictionary containing food data
        
    Returns:
        dict: Cleaned food data
    """
    # Select only the relevant columns
    important_columns = [
        'food_name', 'energy_kcal', 'carb_g', 'protein_g', 'fat_g', 
        'fibre_g', 'freesugar_g', 'cholesterol_mg', 'calcium_mg', 'iron_mg', 
        'sodium_mg', 'potassium_mg', 'zinc_mg', 'vita_ug', 'vitc_mg', 
        'folate_ug', 'servings_unit'
    ]
    
    cleaned_data = {}
    for key, value in food_dict.items():
        if key in important_columns:
            # Skip null values
            if pd.isna(value):
                continue
            # Format numeric values to 1 decimal place if they're floating point numbers
            if isinstance(value, float):
                cleaned_data[key] = round(value, 1)
            else:
                cleaned_data[key] = value
    
    return cleaned_data

@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    """API endpoint for chatbot interactions"""
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        response.headers['Access-Control-Allow-Methods'] = 'POST'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
        
    data = request.json
    user_question = data.get('query', '')
    
    if not user_question:
        return jsonify({"error": "No query provided"}), 400
    
    print(f"Received question: {user_question}")
    
    # Get relevant context for the question
    context = get_relevant_context(user_question)
    
    # Get response from OpenAI
    openai_response = query_openai(user_question, context)
    
    return jsonify({
        "response": openai_response,
        "context": json.loads(context) if isinstance(context, str) else context
    })

@app.route('/api/foods', methods=['GET'])
def get_foods():
    """API endpoint to get a list of available foods"""
    if df is None:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    search = request.args.get('search', '').lower()
    limit = int(request.args.get('limit', 20))
    
    if search:
        filtered_foods = df[df['food_name'].str.lower().str.contains(search)]
    else:
        filtered_foods = df
    
    foods = filtered_foods.head(limit)['food_name'].tolist()
    return jsonify({"foods": foods})

@app.route('/api/food/<food_name>', methods=['GET'])
def get_food_details(food_name):
    """API endpoint to get details about a specific food"""
    if df is None:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    # Find the food by name (case insensitive)
    food_data = df[df['food_name'].str.lower() == food_name.lower()]
    
    if food_data.empty:
        # Try partial match
        food_data = df[df['food_name'].str.lower().str.contains(food_name.lower())]
    
    if food_data.empty:
        return jsonify({"error": f"Food '{food_name}' not found"}), 404
    
    # Get the first matching food
    food = food_data.iloc[0].to_dict()
    return jsonify({"food": clean_food_data(food)})

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """API endpoint to get list of food categories"""
    categories = {
        "breakfast": ["idli", "dosa", "upma", "poha", "paratha"],
        "dessert": ["halwa", "ladoo", "burfi", "kheer"],
        "snack": ["chaat", "pakora", "samosa", "tikki"],
        "curry": ["curry", "sabji", "dal", "sambar"],
        "biryani": ["biryani", "pulao"],
        "bread": ["roti", "naan", "paratha", "chapati"]
    }
    
    return jsonify({"categories": categories})

@app.route('/api/nutrients', methods=['GET'])
def get_nutrients():
    """API endpoint to get list of available nutrients"""
    if df is None:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    nutrients = {
        "macronutrients": ["energy_kcal", "carb_g", "protein_g", "fat_g", "fibre_g", "freesugar_g"],
        "minerals": ["calcium_mg", "iron_mg", "sodium_mg", "potassium_mg", "zinc_mg"],
        "vitamins": ["vita_ug", "vitc_mg", "folate_ug"]
    }
    
    return jsonify({"nutrients": nutrients})

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    """Serve the frontend files"""
    if path == "" or path == "index.html":
        return send_from_directory('static', 'index.html')
    return send_from_directory('static', path)

if __name__ == '__main__':
    # Create static folder if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
    
    # Start the server
    app.run(debug=True, port=5000)
