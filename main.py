import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#import gpa as g
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
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
joblib.dump(model, 'student_performance_model.h5')
'''
# Function to predict student performance
def predict_performance(input_data):
    # Convert input data into DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Convert categorical variables into dummy/indicator variables
    input_df = pd.get_dummies(input_df)
    
    # Ensure input features match the model's expected input
    input_df = input_df.reindex(columns=X.columns, fill_value=0)
    
    # Make prediction
    prediction = model.predict(input_df)
    
    return prediction[0]

# Example usage:
input_data = {
    "GPA": g.gpa,
    "SAT_Score": 1050,
    "Study_Hours_Per_Week": 8,
    "Parent_Education": "High School Diploma",
    "Extracurricular_Participation": "No",
    "Motivation_Level": "Moderate",
    "Home_Environment": "Chaotic"
}

predicted_performance = predict_performance(input_data)
print("Predicted Performance:", predicted_performance)
'''

