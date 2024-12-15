import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset from the CSV file
data = pd.read_csv("student_performance.csv")

# Prepare the features (X) and target variable (y)
X = data.drop(columns=["Name", "Performance"])
y = data["Performance"]

# Convert categorical variables into dummy/indicator variables
X = pd.get_dummies(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model's accuracy
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Random Forest Model Accuracy:", accuracy)

# Save the RandomForest model using joblib
joblib.dump(rf_model, 'random_forest_model.pkl')

# Load the RandomForest model from the file
rf_model = joblib.load('random_forest_model.pkl')

# Load the saved SVM model
svm_model = joblib.load('svm_model.pkl')

# Function to collect and preprocess user input for Random Forest
def get_student_performance_input():
    print("Enter the following details for student performance prediction:")
    
    input_data = {
        "GPA": float(input("GPA: ")),
        "SAT_Score": int(input("SAT Score: ")),
        "Study_Hours_Per_Week": int(input("Study Hours Per Week: ")),
        "Parent_Education": input("Parent Education (e.g., Bachelor's Degree): "),
        "Extracurricular_Participation": input("Extracurricular Participation (e.g., Yes/No): "),
        "Motivation_Level": input("Motivation Level (e.g., High/Medium/Low): "),
        "Home_Environment": input("Home Environment (e.g., Supportive/Unsupportive): ")
    }

    return input_data

# Function to predict overall performance using the Random Forest model
def predict_student_performance():
    # Collect user input
    user_input = get_student_performance_input()
    
    # Convert input data into DataFrame
    input_df = pd.DataFrame([user_input])
    
    # Convert categorical variables into dummy/indicator variables
    input_df = pd.get_dummies(input_df)
    
    # Ensure input features match the model's expected input
    input_df = input_df.reindex(columns=X.columns, fill_value=0)
    
    # Make prediction
    prediction = rf_model.predict(input_df)
    print("\nOverall Student Performance Prediction:")
    print(f"Predicted Performance: {prediction[0]}")
    
    # Evaluate overall performance (Good/Bad)
    if prediction[0] in ['Good', 'High']:  # Adjust as needed based on your labels
        overall_performance = "Good"
    else:
        overall_performance = "Bad"
    
    print(f"Overall Performance: {overall_performance}")

# Function to get user input for individual skill marks for SVM
def get_individual_input():
    print("Please enter the marks for the individual:")
    
    cognitive = float(input("Cognitive Skills (0-5): "))
    interpersonal = float(input("Interpersonal Skills (0-5): "))
    verbal = float(input("Verbal Skills (0-5): "))
    analytical = float(input("Analytical Skills (0-5): "))
    
    return np.array([[cognitive, interpersonal, verbal, analytical]])

# Function to predict individual performance using the SVM model
def predict_individual_performance():
    # Get individual input
    new_data = get_individual_input()
    
    # Use the SVM model to predict the performance label
    prediction = svm_model.predict(new_data)
    
    # Map the predicted label to a meaningful description
    performance_mapping = {
        1: "Low in Verbal Skills",
        2: "Low in Cognitive Skills",
        3: "All good",
        4: "Low in Analytical Skills",
        5: "Low in Interpersonal Skills",
        6: "Low in all categories",
        0: "No specific condition met"
    }
    
    print("\nIndividual Performance Prediction:")
    print(f"Predicted Label: {prediction[0]}")
    print(f"Performance: {performance_mapping.get(prediction[0], 'Unknown label')}")

# Run both predictions
predict_student_performance()
predict_individual_performance()
