import pandas as pd
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
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save the model using joblib
joblib.dump(model, 'student_performance_model.joblib')

# Load the model from the file
loaded_model = joblib.load('student_performance_model.joblib')

# Function to collect and preprocess user input
def get_user_input():
    def get_valid_input(prompt, min_value=None, max_value=None, data_type=int):
        while True:
            try:
                value = data_type(input(prompt))
                if (min_value is not None and value < min_value) or (max_value is not None and value > max_value):
                    print(f"Please enter a value between {min_value} and {max_value}.")
                else:
                    return value
            except ValueError:
                print(f"Invalid input. Please enter a valid {data_type.__name__}.")
    
    print("Enter the following details:")

    input_data = {
        "GPA": get_valid_input("GPA (0-10): ", 0, 10, float),
        "SAT_Score": get_valid_input("SAT Score (10000-30000): ", 10000, 30000, int),
        "Study_Hours_Per_Week": get_valid_input("Study Hours Per Week (8-20): ", 8, 20, int),
        "Parent_Education": input("Parent Education (e.g., Bachelor's Degree): "),
        "Extracurricular_Participation": input("Extracurricular Participation (e.g., Yes/No): "),
        "Motivation_Level": input("Motivation Level (e.g., High/Medium/Low): "),
        "Home_Environment": input("Home Environment (e.g., Supportive/Unsupportive): ")
    }
    
    return input_data

# Example usage
user_input = get_user_input()
print("User Data Collected:", user_input)


# Convert input data into DataFrame
input_df = pd.DataFrame([user_input])

# Convert categorical variables into dummy/indicator variables
input_df = pd.get_dummies(input_df)

# Ensure input features match the model's expected input
input_df = input_df.reindex(columns=X.columns, fill_value=0)

# Make prediction
prediction = loaded_model.predict(input_df)
print("Predicted Performance:", prediction[0])
