from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load models dynamically
rf_model_path = os.getenv('RF_MODEL_PATH', 'random_forest_model.pkl')
svm_model_path = os.getenv('SVM_MODEL_PATH', 'svm_model.pkl')
rf_model = joblib.load(rf_model_path)
svm_model = joblib.load(svm_model_path)

# Utility function to validate inputs
def validate_input(value, min_val, max_val, value_type, field_name):
    try:
        value = value_type(value)
        if not (min_val <= value <= max_val):
            raise ValueError(f"{field_name} must be between {min_val} and {max_val}.")
        return value
    except (ValueError, TypeError):
        raise ValueError(f"Invalid input for {field_name}.")

def get_performance(percentage):
    if percentage >= 90:
        return "Excellent performance!"
    elif percentage >= 75:
        return "Good performance!"
    elif percentage >= 50:
        return "Average performance!"
    else:
        return "Poor performance!"

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/exam', methods=['GET', 'POST'])
def exam():
    try:
        if request.method == 'POST':
            correct_answers = {
                'q1': 'c', 'q2': 'c', 'q3': 'b', 'q4': 'b', 'q5': 'c',
                'q6': 'b', 'q7': 'a', 'q8': 'b', 'q9': 'b', 'q10': 'b',
                'q11': 'b', 'q12': 'b', 'q13': 'b', 'q14': 'b', 'q15': 'b',
                'q16': 'b', 'q17': 'b', 'q18': 'b', 'q19': 'a', 'q20': 'b'
            }

            scores = {'cognitive': 0, 'interpersonal': 0, 'verbal': 0, 'analytical': 0}

            # Calculate scores for each category
            for i in range(1, 6):
                if request.form.get(f'q{i}') == correct_answers[f'q{i}']:
                    scores['cognitive'] += 1
            for i in range(6, 11):
                if request.form.get(f'q{i}') == correct_answers[f'q{i}']:
                    scores['interpersonal'] += 1
            for i in range(11, 16):
                if request.form.get(f'q{i}') == correct_answers[f'q{i}']:
                    scores['verbal'] += 1
            for i in range(16, 21):
                if request.form.get(f'q{i}') == correct_answers[f'q{i}']:
                    scores['analytical'] += 1

            total_questions = 20
            percentages = {key: (value / 5) * 100 for key, value in scores.items()}
            overall_percentage = (sum(scores.values()) / total_questions) * 100

            result = {
                'cognitive': scores['cognitive'],
                'interpersonal': scores['interpersonal'],
                'verbal': scores['verbal'],
                'analytical': scores['analytical'],
                'cognitive_percentage': percentages['cognitive'],
                'interpersonal_percentage': percentages['interpersonal'],
                'verbal_percentage': percentages['verbal'],
                'analytical_percentage': percentages['analytical'],
                'overall_percentage': overall_percentage,
                'performance': get_performance(overall_percentage)
            }

            return render_template('result.html', result=result)
    except Exception as e:
        return render_template('error.html', error_message=str(e)), 500

    return render_template('exam.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        prediction = None
        if request.method == 'POST':
            # Validate inputs for overall prediction
            input_data = {
                "GPA": validate_input(request.form.get('GPA'), 0, 10, float, "GPA"),
                "SAT_Score": validate_input(request.form.get('SAT_Score'), 10000, 30000, int, "SAT Score"),
                "Study_Hours_Per_Week": validate_input(request.form.get('Study_Hours_Per_Week'), 8, 20, int, "Study Hours Per Week"),
                "Parent_Education": request.form.get('Parent_Education'),
                "Extracurricular_Participation": request.form.get('Extracurricular_Participation'),
                "Motivation_Level": request.form.get('Motivation_Level'),
                "Home_Environment": request.form.get('Home_Environment')
            }

            # Convert input data into DataFrame for RandomForest model
            input_df = pd.DataFrame([input_data])
            input_df = pd.get_dummies(input_df)
            input_df = input_df.reindex(columns=rf_model.feature_names_in_, fill_value=0)

            # Predict overall performance
            overall_prediction = rf_model.predict(input_df)[0]

            # Validate and predict individual performance
            individual_input = np.array([
                validate_input(request.form.get('cognitive'), 0, 5, float, "Cognitive"),
                validate_input(request.form.get('interpersonal'), 0, 5, float, "Interpersonal"),
                validate_input(request.form.get('verbal'), 0, 5, float, "Verbal"),
                validate_input(request.form.get('analytical'), 0, 5, float, "Analytical")
            ]).reshape(1, -1)

            individual_prediction = svm_model.predict(individual_input)[0]

            # Map individual performance
            performance_mapping = {
                1: "Low in Verbal Skills",
                2: "Low in Cognitive Skills",
                3: "All good",
                4: "Low in Analytical Skills",
                5: "Low in Interpersonal Skills",
                6: "Low in all categories",
                0: "No specific condition met"
            }

            prediction = {
                'overall': overall_prediction,
                'individual': performance_mapping.get(individual_prediction, 'Unknown label')
            }
        return render_template('predict.html', prediction=prediction)
    except Exception as e:
        return render_template('error.html', error_message=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True)
