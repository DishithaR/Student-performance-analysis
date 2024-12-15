import joblib
import numpy as np

# Load the saved models
rf_model = joblib.load('random_forest_model.pkl')  # Load the Random Forest model (optional for feature ranking)
svm_model = joblib.load('svm_model.pkl')  # Load the SVM model for predictions

# Function to get user input for the four categories
def get_individual_input():
    print("Please enter the marks for the individual:")
    
    cognitive = float(input("Cognitive Skills (0-5): "))
    interpersonal = float(input("Interpersonal Skills (0-5): "))
    verbal = float(input("Verbal Skills (0-5): "))
    analytical = float(input("Analytical Skills (0-5): "))
    
    return np.array([[cognitive, interpersonal, verbal, analytical]])

# Function to predict performance using the SVM model
def predict_performance():
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
    
    print("\nPrediction result:")
    print(f"Predicted Label: {prediction[0]}")
    print(f"Performance: {performance_mapping.get(prediction[0], 'Unknown label')}")

# Run the prediction
predict_performance()