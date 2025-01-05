from pulp import LpMaximize, LpProblem, LpVariable, lpSum
from supabase import create_client, Client                                        # For database adding and pulling

from dotenv import load_dotenv
from pathlib import Path

import os
import sys

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / '.env')

url: str = str(os.getenv("SUPABASE_URL")).strip()
key: str = str(os.getenv("SUPABASE_KEY")).strip()

supabase: Client = create_client(url, key)                          # Supabase client created

budget = 1000000
n = 5                           # number of companies to run optimisation on

# Weights must sum to 1
weights = {
    "fossilFuel": 0.25,
    "prison": 0.10,
    "deforestation": 0.25,
    "tobacco": 0.10,
    "militaryWeapons": 0.15,
    "carbon": 0.15
}

def calculate_green_score(esg_data, weights):
    # Extract ESG metrics
    fossil_fuel = esg_data["fossilFuel"]
    prison = esg_data["prison"]
    deforestation = esg_data["deforestation"]
    tobacco = esg_data["tobacco"]
    military_weapons = esg_data["militaryWeapons"]
    carbon_footprint = esg_data["relativeCarbonFootprint"]
    
    # Normalize grades
    ff_score = (6 - float(fossil_fuel)) / 5
    prison_score = (6 - float(prison)) / 5
    deforestation_score = (6 - float(deforestation)) / 5
    tobacco_score = (6 - tobacco) / 5
    military_weapons_score = (6 - military_weapons) / 5
    carbon_score = 1 - (float(carbon_footprint) / 100)
    
    # Weighted sum
    green_score = (
        weights["fossilFuel"] * ff_score +
        weights["prison"] * prison_score +
        weights["deforestation"] * deforestation_score +
        weights["tobacco"] * tobacco_score +
        weights["militaryWeapons"] * military_weapons_score +
        weights["carbon"] * carbon_score
    )
    return green_score

response = supabase.table('financial_data').select("*").order('id', desc=True).limit(n).execute().data

# This will generate a green score for all the companies in the supabase table
def generate_green_score_for_supabase_tables()
greenScore = []
for ind, val in enumerate(response):
    response[ind].get('user_input').pop('netAssetsRate', None)
    # basically pop it for all
    
    company_esg = response[ind].get('user_input')
    print(company_esg)
    green_score = calculate_green_score(company_esg, weights)
    greenScore.append(green_score)
    # Add the green score

print(greenScore)
# print(response[0].get('user_input'))

# roi = [0.1, 0.12, 0.08, 0.15, 0.09]  # Predicted ROI for each company
# risk = [0.3, 0.5, 0.2, 0.6, 0.4]  # Risk scores for each company
# greenScore = [0.8, 0.6, 0.9, 0.3, 0.7]  # Green scores for each company
# greenThreshold = 0.5 * budget  # Minimum green investment (50% of budget)

# # Weights for risk and green priorities
# lambda_risk = 0.5  # Risk penalty weight
# mu_green = 1.0  # Green reward weight

# # Decision Variables
# x = [LpVariable(f"x_{i}", lowBound=0) for i in range(n)]  # Investment in each company

# # Problem Definition
# model = LpProblem(name="Investment-Optimization", sense=LpMaximize)

# # Objective Function
# model += lpSum((roi[i] - lambda_risk * risk[i] + mu_green * greenScore[i]) * x[i] for i in range(n))

# # Constraints
# model += lpSum(x) <= budget, "BudgetConstraint"  # Total investment <= budget
# model += lpSum(greenScore[i] * x[i] for i in range(n)) >= greenThreshold, "GreenThreshold"  # Minimum green investment

# # Solve
# model.solve()

# # Output Results
# print("Optimal Investment Plan:")
# for i in range(n):
#     print(f"Company {i+1}: ${x[i].value():,.2f}")
# print(f"Total Investment: ${sum(x[i].value() for i in range(n)):,.2f}")
