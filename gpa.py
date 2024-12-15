import random

# Function to generate exam questions along with answers
def generate_exam_questions(num_questions=5):
    questions = []
    answers = []
    for i in range(num_questions):
        question = f"Question {i+1}: "
        # Replace these with your actual questions
        if i == 0:
            question += "What is the capital of France?"
            answers.append("Paris")
        elif i == 1:
            question += "Who wrote 'To Kill a Mockingbird'?"
            answers.append("Harper Lee")
        elif i == 2:
            question += "What is the chemical symbol for water?"
            answers.append("H2O")
        elif i == 3:
            question += "Who painted the Mona Lisa?"
            answers.append("Leonardo da Vinci")
        elif i == 4:
            question += "What is the tallest mountain in the world?"
            answers.append("Mount Everest")
        questions.append(question)
    return questions, answers

# Function to ask questions and get answers from the user
def get_user_answers(questions):
    answers = []
    for question in questions:
        answer = input(question + " ")
        answers.append(answer)
    return answers

# Function to calculate GPA based on exam answers
# Function to calculate GPA based on exam answers
def calculate_gpa(answers, correct_answers):
    total_score = sum(1 for user_ans, correct_ans in zip(answers, correct_answers) if user_ans.lower() == correct_ans.lower())
    num_questions = len(answers)
    percentage_correct = total_score / num_questions  # Calculate percentage of correct answers
    gpa = 1 + (percentage_correct * 2)  # Scale GPA from 1 to 3
    return gpa


# Generate exam questions and answers
exam_questions, correct_answers = generate_exam_questions()

# Ask questions and get answers from the user
user_answers = get_user_answers(exam_questions)

# Calculate GPA based on exam answers
gpa = calculate_gpa(user_answers, correct_answers)
print("\nGPA calculated based on exam scores:", gpa)
