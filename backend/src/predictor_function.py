import pickle
import numpy as np

# Load the saved model and scaler

def load_models():
    with open('../helper/knn_model.pkl', 'rb') as model_file:
        knn_model = pickle.load(model_file)

    with open('../helper/scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    return knn_model, scaler

def grade_to_numeric(grade):
    grade_mapping = {'A': 6, 'B': 5, 'C': 4, 'D': 3, 'E': 2, 'F':1}
    return grade_mapping.get(grade.upper(), None)

'''
Based on the fund's fossil fuel exposure, it earns one of five grades.

A
No holdings flagged for our fossil fuel screens. Assigned a grade of A.

B
Fossil fuel exposure between 0% and 3%; OR, fossil fuel exposure is 0%, but fossil fuel finance and insurance exposure is above 0%. Assigned a grade of B.

C
Fossil fuel exposure between 3% and 5.5%; OR, fossil fuel exposure below 3%, with investments in higher risk companies (top carbon reserve owners and/or top coal-fired utilities). Assigned a grade of C.

D
Fossil fuel exposure between 5.5% and 9%. Assigned a grade of D.

F
Fossil fuel exposure between 9% and 100%. Assigned a grade of F.
'''

grades = ['Fossil Fuel Grade', 'Prison Industrial Complex Grade', 'Deforestation Grade',
          'Tobacco Grade', 'Military Weapon Grade', 'Relative carbon footprint']

# user_input = []
# for grade in grades[:-1]:  # Grades except the last one
#     while True:
#         value = input(f"{grade} (A, B, C, D, E): ").strip()
#         numeric_value = grade_to_numeric(value)
#         if numeric_value is not None:
#             user_input.append(numeric_value)
#             break
#         else:
#             print("Invalid grade. Please enter A, B, C, D, or E.")

def predict(user_input, knn_model, scaler):
    assert type(user_input) == list and len(user_input) == 6, 'wrong data'

    user_input = np.array(user_input).reshape(1, -1)
    user_input_rescaled = scaler.transform(user_input)

    prediction = knn_model.predict(user_input_rescaled)
    roi_end_of_year = f"{prediction[0]:.2f}"

    return roi_end_of_year

# print(f"Predicted Month-End Trailing Returns, Year 1: {prediction[0]:.2f}")
# print('This means you are likely to lose that amount of percentage in a year... So a smaller value is much better!')