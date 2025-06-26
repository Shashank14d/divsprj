import os
import cv2
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import json
import base64
from io import BytesIO

# Try to import TensorFlow, but make it optional
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Using simplified body type detection.")

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model (we'll create this)
MODEL_PATH = 'models/body_type_model.h5'

# Load integrated dataset if available
INTEGRATED_DATA_PATH = 'integrated_data/processed_dataset.xlsx'
user_dataset = None

def load_user_dataset():
    """Load the integrated user dataset"""
    global user_dataset
    try:
        if os.path.exists(INTEGRATED_DATA_PATH):
            user_dataset = pd.read_excel(INTEGRATED_DATA_PATH)
            print(f"Loaded user dataset with {len(user_dataset)} records")
            return True
        else:
            print("No integrated dataset found. Using default plans.")
            return False
    except Exception as e:
        print(f"Error loading user dataset: {e}")
        return False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_bmi(height_feet, height_inches, weight_kg):
    """Calculate BMI from height (feet, inches) and weight (kg)"""
    total_height_inches = (height_feet * 12) + height_inches
    height_meters = total_height_inches * 0.0254
    bmi = weight_kg / (height_meters ** 2)
    return round(bmi, 2)

def categorize_bmi(bmi):
    """Categorize BMI into health categories"""
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def preprocess_image(image_path):
    """Preprocess image for CNN model"""
    try:
        # Load and resize image
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_body_type(image_path):
    """Predict body type using CNN model or simplified analysis"""
    try:
        # For demo purposes, we'll use a simple classification
        # In a real application, you would load a trained CNN model
        body_types = ['Ectomorph', 'Mesomorph', 'Endomorph']
        
        # Simple heuristic based on image analysis
        img = cv2.imread(image_path)
        if img is None:
            return "Mesomorph"  # Default fallback
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Simple analysis based on image characteristics
        # This is a placeholder - in reality, you'd use a trained CNN
        mean_intensity = np.mean(gray)
        
        if mean_intensity < 100:
            return "Ectomorph"
        elif mean_intensity < 150:
            return "Mesomorph"
        else:
            return "Endomorph"
            
    except Exception as e:
        print(f"Error predicting body type: {e}")
        return "Mesomorph"  # Default fallback

def get_user_plan(food_preference, bmi_category, fitness_goal):
    """Get a plan from the user's dataset based on criteria"""
    global user_dataset
    
    if user_dataset is None:
        print("DEBUG: user_dataset is None")
        return None
    
    try:
        print(f"DEBUG: Looking for - Food: {food_preference}, BMI: {bmi_category}, Goal: {fitness_goal}")
        print(f"DEBUG: Available food preferences: {user_dataset['Food_Preference'].unique()}")
        print(f"DEBUG: Available fitness goals: {user_dataset['Fitness_Goal'].unique()}")
        print(f"DEBUG: Available BMI categories: {user_dataset['BMI_Category'].unique()}")
        
        # Filter dataset based on criteria
        filtered_data = user_dataset[
            (user_dataset['Food_Preference'] == food_preference) &
            (user_dataset['BMI_Category'] == bmi_category) &
            (user_dataset['Fitness_Goal'] == fitness_goal)
        ]
        
        print(f"DEBUG: Found {len(filtered_data)} exact matches")
        
        if len(filtered_data) > 0:
            # Get a random plan from matching records
            selected_plan = filtered_data.sample(1).iloc[0]
            return {
                'workout_plan': selected_plan.get('Workout_Plans', ''),
                'diet_plan': selected_plan.get('Diet_Plan', '')
            }
        else:
            # Try with just food preference and fitness goal
            filtered_data = user_dataset[
                (user_dataset['Food_Preference'] == food_preference) &
                (user_dataset['Fitness_Goal'] == fitness_goal)
            ]
            
            print(f"DEBUG: Found {len(filtered_data)} matches with just food and goal")
            
            if len(filtered_data) > 0:
                selected_plan = filtered_data.sample(1).iloc[0]
                return {
                    'workout_plan': selected_plan.get('Workout_Plans', ''),
                    'diet_plan': selected_plan.get('Diet_Plan', '')
                }
        
        print("DEBUG: No matches found")
        return None
    except Exception as e:
        print(f"Error getting user plan: {e}")
        return None

def generate_diet_plan(food_preference, bmi_category, fitness_goal, body_type):
    """Generate personalized diet plan"""
    
    # First try to get a plan from user dataset
    user_plan = get_user_plan(food_preference, bmi_category, fitness_goal)
    if user_plan and user_plan['diet_plan']:
        # Parse the user's diet plan
        diet_text = user_plan['diet_plan']
        # Convert text plan to structured format
        return parse_diet_plan_text(diet_text)
    
    # Fallback to default plans if no user plan found
    diet_plans = {
        "Pure Veg": {
            "Underweight": {
                "Weight Loss": {
                    "Breakfast": "Oatmeal with nuts and fruits, 2 boiled eggs",
                    "Snack": "Mixed nuts and dried fruits",
                    "Lunch": "Brown rice with lentils and vegetables",
                    "Evening": "Greek yogurt with honey",
                    "Dinner": "Quinoa with chickpeas and spinach"
                },
                "Muscle Gain": {
                    "Breakfast": "Protein smoothie with banana and peanut butter",
                    "Snack": "Hummus with whole grain bread",
                    "Lunch": "Paneer curry with brown rice",
                    "Evening": "Protein shake with almonds",
                    "Dinner": "Lentil soup with whole grain bread"
                },
                "Stay Fit": {
                    "Breakfast": "Whole grain toast with avocado",
                    "Snack": "Mixed fruits",
                    "Lunch": "Vegetable biryani with raita",
                    "Evening": "Green tea with nuts",
                    "Dinner": "Grilled vegetables with quinoa"
                }
            },
            "Normal": {
                "Weight Loss": {
                    "Breakfast": "Greek yogurt with berries",
                    "Snack": "Apple with almonds",
                    "Lunch": "Mixed vegetable salad with tofu",
                    "Evening": "Green tea",
                    "Dinner": "Grilled vegetables with lentils"
                },
                "Muscle Gain": {
                    "Breakfast": "Protein pancake with fruits",
                    "Snack": "Protein bar",
                    "Lunch": "Paneer tikka with brown rice",
                    "Evening": "Protein shake",
                    "Dinner": "Lentil curry with quinoa"
                },
                "Stay Fit": {
                    "Breakfast": "Oatmeal with fruits and nuts",
                    "Snack": "Mixed nuts",
                    "Lunch": "Vegetable soup with whole grain bread",
                    "Evening": "Fruit salad",
                    "Dinner": "Grilled vegetables with brown rice"
                }
            },
            "Overweight": {
                "Weight Loss": {
                    "Breakfast": "Green smoothie with spinach",
                    "Snack": "Cucumber slices",
                    "Lunch": "Mixed vegetable salad",
                    "Evening": "Green tea",
                    "Dinner": "Grilled vegetables"
                },
                "Muscle Gain": {
                    "Breakfast": "Protein smoothie",
                    "Snack": "Mixed nuts",
                    "Lunch": "Lentil soup with vegetables",
                    "Evening": "Protein shake",
                    "Dinner": "Grilled tofu with vegetables"
                },
                "Stay Fit": {
                    "Breakfast": "Oatmeal with fruits",
                    "Snack": "Mixed fruits",
                    "Lunch": "Vegetable soup",
                    "Evening": "Green tea",
                    "Dinner": "Grilled vegetables with quinoa"
                }
            },
            "Obese": {
                "Weight Loss": {
                    "Breakfast": "Green smoothie",
                    "Snack": "Cucumber water",
                    "Lunch": "Mixed vegetable salad",
                    "Evening": "Green tea",
                    "Dinner": "Grilled vegetables"
                },
                "Muscle Gain": {
                    "Breakfast": "Protein shake",
                    "Snack": "Mixed nuts",
                    "Lunch": "Lentil soup",
                    "Evening": "Protein shake",
                    "Dinner": "Grilled vegetables"
                },
                "Stay Fit": {
                    "Breakfast": "Green smoothie",
                    "Snack": "Mixed fruits",
                    "Lunch": "Vegetable soup",
                    "Evening": "Green tea",
                    "Dinner": "Grilled vegetables"
                }
            }
        },
        "Veg but Egg is ok": {
            "Underweight": {
                "Weight Loss": {
                    "Breakfast": "Oatmeal with nuts and fruits, 2 boiled eggs",
                    "Snack": "Mixed nuts and dried fruits",
                    "Lunch": "Brown rice with lentils and vegetables",
                    "Evening": "Greek yogurt with honey",
                    "Dinner": "Quinoa with chickpeas and spinach"
                },
                "Muscle Gain": {
                    "Breakfast": "Protein smoothie with banana and peanut butter",
                    "Snack": "Hummus with whole grain bread",
                    "Lunch": "Paneer curry with brown rice",
                    "Evening": "Protein shake with almonds",
                    "Dinner": "Lentil soup with whole grain bread"
                },
                "Stay Fit": {
                    "Breakfast": "Whole grain toast with avocado",
                    "Snack": "Mixed fruits",
                    "Lunch": "Vegetable biryani with raita",
                    "Evening": "Green tea with nuts",
                    "Dinner": "Grilled vegetables with quinoa"
                }
            },
            "Normal": {
                "Weight Loss": {
                    "Breakfast": "Greek yogurt with berries",
                    "Snack": "Apple with almonds",
                    "Lunch": "Mixed vegetable salad with tofu",
                    "Evening": "Green tea",
                    "Dinner": "Grilled vegetables with lentils"
                },
                "Muscle Gain": {
                    "Breakfast": "Protein pancake with fruits",
                    "Snack": "Protein bar",
                    "Lunch": "Paneer tikka with brown rice",
                    "Evening": "Protein shake",
                    "Dinner": "Lentil curry with quinoa"
                },
                "Stay Fit": {
                    "Breakfast": "Oatmeal with fruits and nuts",
                    "Snack": "Mixed nuts",
                    "Lunch": "Vegetable soup with whole grain bread",
                    "Evening": "Fruit salad",
                    "Dinner": "Grilled vegetables with brown rice"
                }
            },
            "Overweight": {
                "Weight Loss": {
                    "Breakfast": "Green smoothie with spinach",
                    "Snack": "Cucumber slices",
                    "Lunch": "Mixed vegetable salad",
                    "Evening": "Green tea",
                    "Dinner": "Grilled vegetables"
                },
                "Muscle Gain": {
                    "Breakfast": "Protein smoothie",
                    "Snack": "Mixed nuts",
                    "Lunch": "Lentil soup with vegetables",
                    "Evening": "Protein shake",
                    "Dinner": "Grilled tofu with vegetables"
                },
                "Stay Fit": {
                    "Breakfast": "Oatmeal with fruits",
                    "Snack": "Mixed fruits",
                    "Lunch": "Vegetable soup",
                    "Evening": "Green tea",
                    "Dinner": "Grilled vegetables with quinoa"
                }
            },
            "Obese": {
                "Weight Loss": {
                    "Breakfast": "Green smoothie",
                    "Snack": "Cucumber water",
                    "Lunch": "Mixed vegetable salad",
                    "Evening": "Green tea",
                    "Dinner": "Grilled vegetables"
                },
                "Muscle Gain": {
                    "Breakfast": "Protein shake",
                    "Snack": "Mixed nuts",
                    "Lunch": "Lentil soup",
                    "Evening": "Protein shake",
                    "Dinner": "Grilled vegetables"
                },
                "Stay Fit": {
                    "Breakfast": "Green smoothie",
                    "Snack": "Mixed fruits",
                    "Lunch": "Vegetable soup",
                    "Evening": "Green tea",
                    "Dinner": "Grilled vegetables"
                }
            }
        },
        "Non-Veg": {
            "Underweight": {
                "Weight Loss": {
                    "Breakfast": "Oatmeal with nuts and fruits, 2 boiled eggs",
                    "Snack": "Mixed nuts and dried fruits",
                    "Lunch": "Grilled chicken with brown rice",
                    "Evening": "Greek yogurt with honey",
                    "Dinner": "Fish curry with quinoa"
                },
                "Muscle Gain": {
                    "Breakfast": "Protein smoothie with banana and peanut butter",
                    "Snack": "Hummus with whole grain bread",
                    "Lunch": "Grilled chicken with lentils",
                    "Evening": "Protein shake with almonds",
                    "Dinner": "Fish soup with whole grain bread"
                },
                "Stay Fit": {
                    "Breakfast": "Whole grain toast with avocado",
                    "Snack": "Mixed fruits",
                    "Lunch": "Grilled chicken with vegetables",
                    "Evening": "Green tea with nuts",
                    "Dinner": "Grilled fish with quinoa"
                }
            },
            "Normal": {
                "Weight Loss": {
                    "Breakfast": "Greek yogurt with berries",
                    "Snack": "Apple with almonds",
                    "Lunch": "Grilled chicken salad",
                    "Evening": "Green tea",
                    "Dinner": "Grilled fish with vegetables"
                },
                "Muscle Gain": {
                    "Breakfast": "Protein pancake with fruits",
                    "Snack": "Protein bar",
                    "Lunch": "Grilled chicken with brown rice",
                    "Evening": "Protein shake",
                    "Dinner": "Fish curry with quinoa"
                },
                "Stay Fit": {
                    "Breakfast": "Oatmeal with fruits and nuts",
                    "Snack": "Mixed nuts",
                    "Lunch": "Chicken soup with whole grain bread",
                    "Evening": "Fruit salad",
                    "Dinner": "Grilled fish with brown rice"
                }
            },
            "Overweight": {
                "Weight Loss": {
                    "Breakfast": "Green smoothie with spinach",
                    "Snack": "Cucumber slices",
                    "Lunch": "Grilled chicken salad",
                    "Evening": "Green tea",
                    "Dinner": "Grilled fish with vegetables"
                },
                "Muscle Gain": {
                    "Breakfast": "Protein smoothie",
                    "Snack": "Mixed nuts",
                    "Lunch": "Chicken soup with vegetables",
                    "Evening": "Protein shake",
                    "Dinner": "Grilled chicken with vegetables"
                },
                "Stay Fit": {
                    "Breakfast": "Oatmeal with fruits",
                    "Snack": "Mixed fruits",
                    "Lunch": "Chicken soup",
                    "Evening": "Green tea",
                    "Dinner": "Grilled fish with quinoa"
                }
            },
            "Obese": {
                "Weight Loss": {
                    "Breakfast": "Green smoothie",
                    "Snack": "Cucumber water",
                    "Lunch": "Grilled chicken salad",
                    "Evening": "Green tea",
                    "Dinner": "Grilled fish with vegetables"
                },
                "Muscle Gain": {
                    "Breakfast": "Protein shake",
                    "Snack": "Mixed nuts",
                    "Lunch": "Chicken soup",
                    "Evening": "Protein shake",
                    "Dinner": "Grilled chicken with vegetables"
                },
                "Stay Fit": {
                    "Breakfast": "Green smoothie",
                    "Snack": "Mixed fruits",
                    "Lunch": "Chicken soup",
                    "Evening": "Green tea",
                    "Dinner": "Grilled fish with vegetables"
                }
            }
        },
        "Vegan": {
            "Underweight": {
                "Weight Loss": {
                    "Breakfast": "Oatmeal with nuts and fruits",
                    "Snack": "Mixed nuts and dried fruits",
                    "Lunch": "Brown rice with lentils and vegetables",
                    "Evening": "Almond milk with honey",
                    "Dinner": "Quinoa with chickpeas and spinach"
                },
                "Muscle Gain": {
                    "Breakfast": "Protein smoothie with banana and peanut butter",
                    "Snack": "Hummus with whole grain bread",
                    "Lunch": "Lentil curry with brown rice",
                    "Evening": "Protein shake with almonds",
                    "Dinner": "Lentil soup with whole grain bread"
                },
                "Stay Fit": {
                    "Breakfast": "Whole grain toast with avocado",
                    "Snack": "Mixed fruits",
                    "Lunch": "Vegetable biryani with coconut yogurt",
                    "Evening": "Green tea with nuts",
                    "Dinner": "Grilled vegetables with quinoa"
                }
            },
            "Normal": {
                "Weight Loss": {
                    "Breakfast": "Coconut yogurt with berries",
                    "Snack": "Apple with almonds",
                    "Lunch": "Mixed vegetable salad with tofu",
                    "Evening": "Green tea",
                    "Dinner": "Grilled vegetables with lentils"
                },
                "Muscle Gain": {
                    "Breakfast": "Protein pancake with fruits",
                    "Snack": "Protein bar",
                    "Lunch": "Tofu tikka with brown rice",
                    "Evening": "Protein shake",
                    "Dinner": "Lentil curry with quinoa"
                },
                "Stay Fit": {
                    "Breakfast": "Oatmeal with fruits and nuts",
                    "Snack": "Mixed nuts",
                    "Lunch": "Vegetable soup with whole grain bread",
                    "Evening": "Fruit salad",
                    "Dinner": "Grilled vegetables with brown rice"
                }
            },
            "Overweight": {
                "Weight Loss": {
                    "Breakfast": "Green smoothie with spinach",
                    "Snack": "Cucumber slices",
                    "Lunch": "Mixed vegetable salad",
                    "Evening": "Green tea",
                    "Dinner": "Grilled vegetables"
                },
                "Muscle Gain": {
                    "Breakfast": "Protein smoothie",
                    "Snack": "Mixed nuts",
                    "Lunch": "Lentil soup with vegetables",
                    "Evening": "Protein shake",
                    "Dinner": "Grilled tofu with vegetables"
                },
                "Stay Fit": {
                    "Breakfast": "Oatmeal with fruits",
                    "Snack": "Mixed fruits",
                    "Lunch": "Vegetable soup",
                    "Evening": "Green tea",
                    "Dinner": "Grilled vegetables with quinoa"
                }
            },
            "Obese": {
                "Weight Loss": {
                    "Breakfast": "Green smoothie",
                    "Snack": "Cucumber water",
                    "Lunch": "Mixed vegetable salad",
                    "Evening": "Green tea",
                    "Dinner": "Grilled vegetables"
                },
                "Muscle Gain": {
                    "Breakfast": "Protein shake",
                    "Snack": "Mixed nuts",
                    "Lunch": "Lentil soup",
                    "Evening": "Protein shake",
                    "Dinner": "Grilled vegetables"
                },
                "Stay Fit": {
                    "Breakfast": "Green smoothie",
                    "Snack": "Mixed fruits",
                    "Lunch": "Vegetable soup",
                    "Evening": "Green tea",
                    "Dinner": "Grilled vegetables"
                }
            }
        }
    }
    
    return diet_plans.get(food_preference, {}).get(bmi_category, {}).get(fitness_goal, {})

def parse_diet_plan_text(diet_text):
    """Parse diet plan text into structured format"""
    try:
        # Split by lines and extract meal information
        lines = diet_text.split('\n')
        meals = {}
        current_meal = None
        
        for line in lines:
            line = line.strip()
            if line and 'AM' in line or 'PM' in line:
                # Extract time and meal name
                if 'Breakfast' in line or '700' in line or '800' in line:
                    current_meal = 'Breakfast'
                elif 'Mid-morning' in line or '1100' in line:
                    current_meal = 'Snack'
                elif 'Lunch' in line or '100' in line or '1:00' in line:
                    current_meal = 'Lunch'
                elif 'Snack' in line or '400' in line or '4:00' in line:
                    current_meal = 'Evening'
                elif 'Dinner' in line or '700' in line or '7:00' in line:
                    current_meal = 'Dinner'
                elif 'Optional' in line or '900' in line or '9:00' in line:
                    current_meal = 'Optional'
                
                # Extract food items
                if current_meal and ':' in line:
                    food_part = line.split(':', 1)[1].strip()
                    if food_part:
                        meals[current_meal] = food_part
        
        # Ensure we have at least basic meals
        if not meals:
            return {
                'Breakfast': 'Custom diet plan from your dataset',
                'Snack': 'Custom diet plan from your dataset',
                'Lunch': 'Custom diet plan from your dataset',
                'Evening': 'Custom diet plan from your dataset',
                'Dinner': 'Custom diet plan from your dataset'
            }
        
        return meals
    except:
        return {
            'Breakfast': 'Custom diet plan from your dataset',
            'Snack': 'Custom diet plan from your dataset',
            'Lunch': 'Custom diet plan from your dataset',
            'Evening': 'Custom diet plan from your dataset',
            'Dinner': 'Custom diet plan from your dataset'
        }

def generate_workout_plan(bmi_category, fitness_goal, body_type):
    """Generate personalized workout plan"""
    
    # First try to get a plan from user dataset
    user_plan = get_user_plan('Non-Veg', bmi_category, fitness_goal)  # Try with any food preference
    if user_plan and user_plan['workout_plan']:
        # Parse the user's workout plan
        workout_text = user_plan['workout_plan']
        return parse_workout_plan_text(workout_text)
    
    # Fallback to default plans if no user plan found
    workout_plans = {
        "Underweight": {
            "Weight Loss": {
                "Monday": "Strength Training: Squats, Deadlifts, Bench Press (3 sets each)",
                "Tuesday": "Cardio: 20 minutes walking + HIIT intervals",
                "Wednesday": "Strength Training: Pull-ups, Rows, Shoulder Press",
                "Thursday": "Rest day with light stretching",
                "Friday": "Strength Training: Leg Press, Lunges, Calf Raises",
                "Saturday": "Cardio: 30 minutes cycling",
                "Sunday": "Rest day"
            },
            "Muscle Gain": {
                "Monday": "Chest & Triceps: Bench Press, Incline Press, Dips",
                "Tuesday": "Back & Biceps: Deadlifts, Pull-ups, Rows",
                "Wednesday": "Legs: Squats, Leg Press, Lunges",
                "Thursday": "Shoulders: Military Press, Lateral Raises",
                "Friday": "Arms: Bicep Curls, Tricep Extensions",
                "Saturday": "Full Body: Compound movements",
                "Sunday": "Rest day"
            },
            "Stay Fit": {
                "Monday": "Full Body: Squats, Push-ups, Rows (3 sets each)",
                "Tuesday": "Cardio: 30 minutes walking/jogging",
                "Wednesday": "Strength: Deadlifts, Shoulder Press, Lunges",
                "Thursday": "Rest day with yoga",
                "Friday": "Full Body: Bench Press, Pull-ups, Planks",
                "Saturday": "Cardio: 20 minutes cycling + stretching",
                "Sunday": "Rest day"
            }
        },
        "Normal": {
            "Weight Loss": {
                "Monday": "HIIT Training: 30 minutes",
                "Tuesday": "Strength Training: Upper body focus",
                "Wednesday": "Cardio: 45 minutes running/cycling",
                "Thursday": "Strength Training: Lower body focus",
                "Friday": "HIIT Training: 25 minutes",
                "Saturday": "Yoga or Pilates",
                "Sunday": "Rest day"
            },
            "Muscle Gain": {
                "Monday": "Chest & Triceps: Heavy compound movements",
                "Tuesday": "Back & Biceps: Deadlifts, Pull-ups",
                "Wednesday": "Legs: Squats, Leg Press, Deadlifts",
                "Thursday": "Shoulders & Arms: Military Press, Curls",
                "Friday": "Full Body: Compound movements",
                "Saturday": "Light cardio + stretching",
                "Sunday": "Rest day"
            },
            "Stay Fit": {
                "Monday": "Full Body: Compound movements",
                "Tuesday": "Cardio: 30 minutes moderate intensity",
                "Wednesday": "Strength: Upper body focus",
                "Thursday": "Yoga or Pilates",
                "Friday": "Full Body: Lower body focus",
                "Saturday": "Cardio: 20 minutes + stretching",
                "Sunday": "Rest day"
            }
        },
        "Overweight": {
            "Weight Loss": {
                "Monday": "Low Impact Cardio: 30 minutes walking",
                "Tuesday": "Strength Training: Bodyweight exercises",
                "Wednesday": "Cardio: 45 minutes cycling/swimming",
                "Thursday": "Yoga or Pilates",
                "Friday": "Strength Training: Light weights",
                "Saturday": "Cardio: 30 minutes walking",
                "Sunday": "Rest day"
            },
            "Muscle Gain": {
                "Monday": "Strength Training: Compound movements",
                "Tuesday": "Cardio: 20 minutes walking",
                "Wednesday": "Strength Training: Upper body",
                "Thursday": "Rest day with stretching",
                "Friday": "Strength Training: Lower body",
                "Saturday": "Light cardio + stretching",
                "Sunday": "Rest day"
            },
            "Stay Fit": {
                "Monday": "Full Body: Bodyweight exercises",
                "Tuesday": "Cardio: 30 minutes walking",
                "Wednesday": "Strength: Light weights",
                "Thursday": "Yoga or Pilates",
                "Friday": "Full Body: Compound movements",
                "Saturday": "Cardio: 20 minutes + stretching",
                "Sunday": "Rest day"
            }
        },
        "Obese": {
            "Weight Loss": {
                "Monday": "Walking: 20 minutes",
                "Tuesday": "Chair exercises and stretching",
                "Wednesday": "Walking: 25 minutes",
                "Thursday": "Rest day with light stretching",
                "Friday": "Walking: 20 minutes",
                "Saturday": "Light stretching and mobility work",
                "Sunday": "Rest day"
            },
            "Muscle Gain": {
                "Monday": "Chair exercises and light weights",
                "Tuesday": "Walking: 15 minutes",
                "Wednesday": "Bodyweight exercises",
                "Thursday": "Rest day with stretching",
                "Friday": "Light strength training",
                "Saturday": "Walking: 10 minutes + stretching",
                "Sunday": "Rest day"
            },
            "Stay Fit": {
                "Monday": "Walking: 15 minutes",
                "Tuesday": "Chair exercises",
                "Wednesday": "Walking: 20 minutes",
                "Thursday": "Rest day with stretching",
                "Friday": "Light bodyweight exercises",
                "Saturday": "Walking: 15 minutes",
                "Sunday": "Rest day"
            }
        }
    }
    
    return workout_plans.get(bmi_category, {}).get(fitness_goal, {})

def parse_workout_plan_text(workout_text):
    """Parse workout plan text into structured format"""
    try:
        # Split by lines and extract workout information
        lines = workout_text.split('\n')
        workouts = {}
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        current_day = None
        
        for line in lines:
            line = line.strip()
            if line:
                # Check if line contains day information
                for day in days:
                    if day.lower() in line.lower():
                        current_day = day
                        break
                
                # If we have a current day and line contains workout info
                if current_day and ('walking' in line.lower() or 'jogging' in line.lower() or 
                                  'cardio' in line.lower() or 'strength' in line.lower() or
                                  'yoga' in line.lower() or 'rest' in line.lower()):
                    workouts[current_day] = line
        
        # Ensure we have at least basic workouts
        if not workouts:
            return {
                'Monday': 'Custom workout plan from your dataset',
                'Tuesday': 'Custom workout plan from your dataset',
                'Wednesday': 'Custom workout plan from your dataset',
                'Thursday': 'Custom workout plan from your dataset',
                'Friday': 'Custom workout plan from your dataset',
                'Saturday': 'Custom workout plan from your dataset',
                'Sunday': 'Rest day'
            }
        
        return workouts
    except:
        return {
            'Monday': 'Custom workout plan from your dataset',
            'Tuesday': 'Custom workout plan from your dataset',
            'Wednesday': 'Custom workout plan from your dataset',
            'Thursday': 'Custom workout plan from your dataset',
            'Friday': 'Custom workout plan from your dataset',
            'Saturday': 'Custom workout plan from your dataset',
            'Sunday': 'Rest day'
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    try:
        # Get form data
        name = request.form['name']
        age = int(request.form['age'])
        gender = request.form['gender']
        height_feet_decimal = float(request.form['height_feet'])
        weight_kg = float(request.form['weight_kg'])
        food_preference = request.form['food_preference']
        fitness_goal = request.form['fitness_goal']
        health_issues = request.form.get('health_issues', '')
        
        # Convert decimal feet to feet and inches
        height_feet = int(height_feet_decimal)
        height_inches = int(round((height_feet_decimal - height_feet) * 12))
        
        # Check if image was uploaded
        if 'image' not in request.files:
            flash('No image uploaded!', 'error')
            return redirect(request.url)
        
        file = request.files['image']
        if file.filename == '':
            flash('No image selected!', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Calculate BMI
            bmi = calculate_bmi(height_feet, height_inches, weight_kg)
            bmi_category = categorize_bmi(bmi)
            
            # Predict body type
            body_type = predict_body_type(filepath)
            
            # Generate plans
            diet_plan = generate_diet_plan(food_preference, bmi_category, fitness_goal, body_type)
            workout_plan = generate_workout_plan(bmi_category, fitness_goal, body_type)
            
            # Prepare results
            results = {
                'name': name,
                'age': age,
                'gender': gender,
                'height_feet': height_feet_decimal,
                'height_inches': 0,  # Not used anymore
                'weight_kg': weight_kg,
                'food_preference': food_preference,
                'fitness_goal': fitness_goal,
                'health_issues': health_issues,
                'bmi': bmi,
                'bmi_category': bmi_category,
                'body_type': body_type,
                'diet_plan': diet_plan,
                'workout_plan': workout_plan,
                'image_path': f'uploads/{filename}'
            }
            
            return render_template('results.html', results=results)
        else:
            flash('Invalid file type! Please upload an image.', 'error')
            return redirect(request.url)
            
    except Exception as e:
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect(request.url)

@app.route('/api/calculate_bmi', methods=['POST'])
def api_calculate_bmi():
    """API endpoint for BMI calculation"""
    try:
        data = request.get_json()
        height_feet = int(data['height_feet'])
        height_inches = int(data['height_inches'])
        weight_kg = float(data['weight_kg'])
        
        bmi = calculate_bmi(height_feet, height_inches, weight_kg)
        bmi_category = categorize_bmi(bmi)
        
        return jsonify({
            'bmi': bmi,
            'bmi_category': bmi_category
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Load user dataset on startup
    load_user_dataset()
    app.run(debug=True, host='0.0.0.0', port=5000) 