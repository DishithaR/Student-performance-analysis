import csv
import random
import string

# Sample data for students
students = [
    {
        "Name": "Student1",
        "GPA": 3.5,
        "SAT_Score": 1200,
        "Study_Hours_Per_Week": 10,
        "Parent_Education": "Bachelor's Degree",
        "Extracurricular_Participation": "Yes",
        "Motivation_Level": "High",
        "Home_Environment": "Supportive"
    },
    {
        "Name": "Student2",
        "GPA": 2.8,
        "SAT_Score": 1050,
        "Study_Hours_Per_Week": 8,
        "Parent_Education": "High School Diploma",
        "Extracurricular_Participation": "No",
        "Motivation_Level": "Moderate",
        "Home_Environment": "Chaotic"
    },
    # Add more students as needed
]

# Define the CSV file path
csv_file = "student_performance.csv"

# Define the CSV fieldnames
fieldnames = [
    "Name",
    "GPA",
    "SAT_Score",
    "Study_Hours_Per_Week",
    "Parent_Education",
    "Extracurricular_Participation",
    "Motivation_Level",
    "Home_Environment",
    "Performance"
]

# Calculate Performance based on criteria (example)
def calculate_performance(student):
    if (student["GPA"] >= 2.5 and student["SAT_Score"] >= 1000 and 
            student["Study_Hours_Per_Week"] >= 6 and student["Parent_Education"] in ["Bachelor's Degree", "Master's Degree", "PhD"] and 
            student["Extracurricular_Participation"] == "Yes" and student["Motivation_Level"] != "Low" and 
            student["Home_Environment"] != "Chaotic"):
        return "Good"
    else:
        return "Bad"



# Function to generate random student data
def generate_random_student_data():
    name = ''.join(random.choices(string.ascii_uppercase, k=8))  # Random name
    gpa = round(random.uniform(2.0, 4.0), 1)  # Random GPA between 2.0 and 4.0
    sat_score = random.randint(900, 1600)  # Random SAT score between 900 and 1600
    study_hours = random.randint(5, 20)  # Random study hours between 5 and 20
    parent_education = random.choice(["High School Diploma", "Bachelor's Degree", "Master's Degree", "PhD"])  # Random parent education level
    extracurricular_participation = random.choice(["Yes", "No"])  # Random extracurricular participation
    motivation_level = random.choice(["Low", "Moderate", "High"])  # Random motivation level
    home_environment = random.choice(["Chaotic", "Supportive", "Neutral"])  # Random home environment
    return {
        "Name": name,
        "GPA": gpa,
        "SAT_Score": sat_score,
        "Study_Hours_Per_Week": study_hours,
        "Parent_Education": parent_education,
        "Extracurricular_Participation": extracurricular_participation,
        "Motivation_Level": motivation_level,
        "Home_Environment": home_environment
    }

# Generate 100 random student data points
random_students = [generate_random_student_data() for _ in range(1000)]

# Combine existing, additional, and new students
all_students = students + random_students

# Write all student data to CSV
with open(csv_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    for student in all_students:
        student["Performance"] = calculate_performance(student)
        writer.writerow(student)

print("CSV file created successfully with additional random data.")
