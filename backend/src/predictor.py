from flask import Flask, request, jsonify
from flask_cors import CORS

import pulp
import pandas as pd
import numpy as np

import os
import sys
import requests

from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.generativeai as genai

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) # adding the root directory to path
from predictor_function import grade_to_numeric
from predictor_function import predict, load_models

from dotenv import load_dotenv
from pathlib import Path

from supabase import create_client, Client                                        # For database adding and pulling

load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

url: str = str(os.getenv("SUPABASE_URL")).strip()
key: str = str(os.getenv("SUPABASE_KEY")).strip()

supabase: Client = create_client(url, key)							# Supabase client created

API_KEY = str(os.getenv("API_KEY")).strip()

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

def fetch_company_data(supabase: Client, limit: int = 2) -> list:
    print('here')
    response = supabase.table('financial_data').select("*").order('id', desc=True).limit(limit).execute().data
    print(response)
    companies_data = []
    for record in response:
        try:
            print("Fetched record:", record)
            user_input = record.get('user_input', {})
            returns = record.get('returns', 0.0)

            # Validate presence of required fields
            if not isinstance(user_input, dict) or not user_input:
                raise ValueError(f"Invalid or missing 'user_input' in record: {record}")

            company_data = {
                'name': f"Company_{record['id']}",
                'user_input': user_input,
                'results': {'predicted_roi': float(returns or 0.0)},
                'risk_score': calculate_risk_score(record)
            }
            companies_data.append(company_data)
            print("Fetched Companies Data:", companies_data)

        except Exception as e:
            print(f"Error processing record: {record}, error: {e}")
            continue  # Skip this record and proceed

    return companies_data


def calculate_risk_score(company_record):
    """
    Calculate risk score based on company metrics.
    Implement your risk scoring logic here.
    """
    # Example risk scoring - modify according to your needs
    base_risk = 0.3  # Base risk score
    esg_metrics = company_record['user_input']
    
    # Risk factors based on ESG metrics
    risk_factors = {
        'fossilFuel': 0.1,
        'prison': 0.05,
        'deforestation': 0.08,
        'tobacco': 0.07,
        'militaryWeapons': 0.1
    }
    
    risk_score = base_risk
    for metric, factor in risk_factors.items():
        # Higher numerical grade (1-5) means higher risk
        risk_score += (esg_metrics[metric] / 5) * factor
    
    # Adjust based on carbon footprint
    carbon_risk = (esg_metrics['relativeCarbonFootprint'] / 100) * 0.1
    risk_score += carbon_risk
    
    return min(max(risk_score, 0.1), 0.9)  # Keeping risk score between 0.1 and 0.9

def print_portfolio_analysis(optimal_portfolio):
    output = ""

    print("\nPORTFOLIO ANALYSIS REPORT")
    output += "\nPORTFOLIO ANALYSIS REPORT"
    print("=" * 50)
    output += "="* 50
    
    print("\n1. BASIC PORTFOLIO COMPOSITION")
    print(f"Total Invested: ${optimal_portfolio['portfolio_metrics']['total_invested']:,.2f}")
    print(f"Number of Holdings: {optimal_portfolio['portfolio_metrics']['number_of_holdings']}")
    print(f"Average Position Size: ${optimal_portfolio['portfolio_metrics']['average_position_size']:,.2f}")
    
    print("\n2. ESG PROFILE")
    print(f"Portfolio ESG Score: {optimal_portfolio['portfolio_metrics']['portfolio_esg_score']:.3f}")
    print(f"Weighted Carbon Footprint: {optimal_portfolio['portfolio_metrics']['weighted_carbon_footprint']:.2f}")
    print(f"Green Companies Allocation: {optimal_portfolio['portfolio_metrics']['green_companies_allocation']*100:.1f}%")
    
    print("\n3. RISK AND RETURN METRICS")
    print(f"Expected Portfolio ROI: {optimal_portfolio['portfolio_metrics']['expected_portfolio_roi']*100:.2f}%")
    print(f"Portfolio Risk Score: {optimal_portfolio['portfolio_metrics']['portfolio_risk']:.3f}")
    print(f"Risk-Adjusted Return: {optimal_portfolio['portfolio_metrics']['risk_adjusted_return']:.3f}")
    
    print("\n4. DIVERSIFICATION METRICS")
    print(f"Diversification Score: {optimal_portfolio['portfolio_metrics']['diversification_score']:.3f}")
    print(f"Largest Position: {optimal_portfolio['portfolio_metrics']['largest_position']*100:.1f}%")
    print(f"Smallest Position: {optimal_portfolio['portfolio_metrics']['smallest_position']*100:.1f}%")
    
    print("\n5. HOLDINGS BREAKDOWN")
    output += f"""
\n1. BASIC PORTFOLIO COMPOSITION \n
Total Invested: ${optimal_portfolio['portfolio_metrics']['total_invested']:,.2f} \n
Number of Holdings: {optimal_portfolio['portfolio_metrics']['number_of_holdings']} \n
Average Position Size: ${optimal_portfolio['portfolio_metrics']['average_position_size']:,.2f}
\n2. ESG PROFILE \n
Portfolio ESG Score: {optimal_portfolio['portfolio_metrics']['portfolio_esg_score']:.3f} \n
Weighted Carbon Footprint: {optimal_portfolio['portfolio_metrics']['weighted_carbon_footprint']:.2f} \n
Green Companies Allocation: {optimal_portfolio['portfolio_metrics']['green_companies_allocation']*100:.1f}% \n
\n3. RISK AND RETURN METRICS \n
Expected Portfolio ROI: {optimal_portfolio['portfolio_metrics']['expected_portfolio_roi']*100:.2f}% \n
Portfolio Risk Score: {optimal_portfolio['portfolio_metrics']['portfolio_risk']:.3f} \n
Risk-Adjusted Return: {optimal_portfolio['portfolio_metrics']['risk_adjusted_return']:.3f} \n

4. DIVERSIFICATION METRICS \n
Diversification Score: {optimal_portfolio['portfolio_metrics']['diversification_score']:.3f} \n
Largest Position: {optimal_portfolio['portfolio_metrics']['largest_position']*100:.1f}% \n
Smallest Position: {optimal_portfolio['portfolio_metrics']['smallest_position']*100:.1f}% \n
    """
    for holding in optimal_portfolio['portfolio_metrics']['holdings_breakdown']:
        print(f"\n{holding['company']}:")
        print(f"  Weight: {holding['weight']*100:.1f}%")
        print(f"  ESG Contribution: {holding['esg_contribution']:.3f}")
        print(f"  Risk Contribution: {holding['risk_contribution']:.3f}")
        print(f"  Return Contribution: {holding['return_contribution']:.3f}")
        output += f"""
\n{holding['company']}: \n
 Weight: {holding['weight']*100:.1f}% \n
 ESG Contribution: {holding['esg_contribution']:.3f} \n
 Risk Contribution: {holding['risk_contribution']:.3f} \n
 Return Contribution: {holding['return_contribution']:.3f} \n
        """
    return output


def optimize_portfolio(companies_data, total_budget, min_esg_score=0.6):
    """
    Optimize investment portfolio considering ESG metrics and risk scores.
    Modified to work with numerical grades (1-5, where 1 is best)
    """
    prob = pulp.LpProblem("ESG_Investment_Optimization", pulp.LpMaximize)
    
    investments = pulp.LpVariable.dicts("invest",
                                      [company['name'] for company in companies_data],
                                      lowBound=0)
    
    invest_binary = pulp.LpVariable.dicts("invest_binary",
                                         [company['name'] for company in companies_data],
                                         cat='Binary')
    
    # Calculate ESG scores (normalized to 0-1 scale, where 1 is best)
    for company in companies_data:
        esg_metrics = company['user_input']
        # Convert numerical grades (1-5) to 0-1 scale where 1 is best
        company['esg_score'] = 1 - (
            (esg_metrics['fossilFuel'] / 5) * 0.25 +
            (esg_metrics['prison'] / 5) * 0.15 +
            (esg_metrics['deforestation'] / 5) * 0.20 +
            (esg_metrics['tobacco'] / 5) * 0.15 +
            (esg_metrics['militaryWeapons'] / 5) * 0.25
        )
    
    # Objective function
    prob += pulp.lpSum([
    investments[company['name']] * (company['results']['predicted_roi'] + 10) / (1 + company['risk_score'])
    for company in companies_data
	])

    
    # Constraints
    
    # Budget constraint
    prob += pulp.lpSum([investments[company['name']] for company in companies_data]) <= total_budget
    
    # Minimum investment amounts if we choose to invest
    min_investment = total_budget * 0.05  # 5% minimum investment
    for company in companies_data:
        prob += investments[company['name']] >= min_investment * invest_binary[company['name']]
        prob += investments[company['name']] <= total_budget * invest_binary[company['name']]
    
    # ESG score constraint - portfolio average must meet minimum
    prob += pulp.lpSum([
        investments[company['name']] * company['esg_score']
        for company in companies_data
    ]) >= min_esg_score * pulp.lpSum([investments[company['name']] for company in companies_data])
    
    # Risk diversification - no more than 40% in any single company
    for company in companies_data:
        prob += investments[company['name']] <= 0.4 * total_budget
    
    # Solve the problem
    prob.solve()
    
    # Extract results
    results = {
        'status': pulp.LpStatus[prob.status],
        'optimal_investments': {},
        'portfolio_metrics': {}
    }
    
    if prob.status == 1:  # Optimal solution found
        total_invested = 0
        weighted_esg_score = 0
        weighted_roi = 0
        weighted_risk = 0
        portfolio_investments = []
        esg_scores = []
        carbon_footprints = []
        risk_scores = []
        rois = []
        
        for company in companies_data:
            amount = investments[company['name']].value()
            if amount > 0:
                results['optimal_investments'][company['name']] = amount
                weight = amount / total_budget
                portfolio_investments.append({
                    'company': company['name'],
                    'amount': amount,
                    'weight': weight,
                    'esg_score': company['esg_score'],
                    'carbon_footprint': company['user_input']['relativeCarbonFootprint'],
                    'risk_score': company['risk_score'],
                    'roi': company['results']['predicted_roi']
                })
                
                total_invested += amount
                weighted_esg_score += weight * company['esg_score']
                weighted_roi += weight * company['results']['predicted_roi']
                weighted_risk += weight * company['risk_score']
                
                esg_scores.append(company['esg_score'])
                carbon_footprints.append(company['user_input']['relativeCarbonFootprint'])
                risk_scores.append(company['risk_score'])
                rois.append(company['results']['predicted_roi'])
        
        # Calculate portfolio diversification metrics
        diversification_score = 1 - np.sqrt(sum([(amt/total_invested)**2 for amt in results['optimal_investments'].values()]))
        
        # Calculate sector concentration (green companies are those with carbon footprint < 30)
        green_weight = sum(weight for inv in portfolio_investments 
                         if inv['carbon_footprint'] < 30)
        
        results['portfolio_metrics'] = {
            # Basic Metrics
            'total_invested': total_invested,
            'number_of_holdings': len(portfolio_investments),
            'average_position_size': total_invested / len(portfolio_investments) if portfolio_investments else 0,
            
            # ESG Metrics
            'portfolio_esg_score': weighted_esg_score,
            'esg_score_range': {
                'min': min(esg_scores) if esg_scores else 0,
                'max': max(esg_scores) if esg_scores else 0,
                'std_dev': np.std(esg_scores) if esg_scores else 0
            },
            'weighted_carbon_footprint': sum(inv['weight'] * inv['carbon_footprint'] 
                                          for inv in portfolio_investments),
            'green_companies_allocation': green_weight,
            
            # Risk Metrics
            'portfolio_risk': weighted_risk,
            'risk_adjusted_return': weighted_roi / weighted_risk if weighted_risk > 0 else 0,
            'risk_score_range': {
                'min': min(risk_scores) if risk_scores else 0,
                'max': max(risk_scores) if risk_scores else 0,
                'std_dev': np.std(risk_scores) if risk_scores else 0
            },
            
            # Return Metrics
            'expected_portfolio_roi': weighted_roi,
            'roi_range': {
                'min': min(rois) if rois else 0,
                'max': max(rois) if rois else 0,
                'std_dev': np.std(rois) if rois else 0
            },
            
            # Diversification Metrics
            'diversification_score': diversification_score,
            'largest_position': max(amt/total_invested for amt in results['optimal_investments'].values()) if results['optimal_investments'] else 0,
            'smallest_position': min(amt/total_invested for amt in results['optimal_investments'].values()) if results['optimal_investments'] else 0,
            
            # Composition Analysis
            'holdings_breakdown': [{
                'company': inv['company'],
                'weight': inv['weight'],
                'esg_contribution': inv['weight'] * inv['esg_score'],
                'risk_contribution': inv['weight'] * inv['risk_score'],
                'return_contribution': inv['weight'] * inv['roi']
            } for inv in portfolio_investments]
        }
    return results

def extract_list(user_input_raw):
	prompt_init = """ 

	you will be given the raw user input. You are required to parse the following values from it (the prompt will necessarily contain all these parameters and their associated values)

	1. Fossil Fuel Grade
	2. Prison Grade
	3. Deforestation Grade
    4. Tobacco Grade
    5. Military Weapon Grade
    6. Relative carbon footprint
    7. Assets

    Note that you must follow this exact order always

    ---
    Example:

    user input: Relative carbon footprint is around 56.1, Fossil Fuel Grade is A, Deforestation D, Prison Grade is C, assets worth 1 million dollars ,Military Grade is F, Tobacco is A
    Gemini output: 

    fossilFuel, A
	prison, C
	deforestation, D
    tobacco, A
    militaryWeapons, F
    relativeCarbonFootprint, 56.1
    netAssetsRate, 1000000

    ---

    You must output in that exact order!

	"""

	prompt = prompt_init + f"This is the user's input: {user_input_raw}"

	try:
		output = ''
		response = chat.send_message(prompt, stream=False, safety_settings={
				HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE, 
				HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
				HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
				HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE
		})

		# Sometimes the model acts up...
		if not response:
			raise ValueError("No response received")

		for chunk in response:
			if chunk.text:
				output += str(chunk.text)

		values = []
		dic = {}

		try:
			for _ in output.split('\n')[:-2]:			# All except last 2
				values.append(grade_to_numeric(_.split(',')[1].strip()))
				dic[_.split(',')[0].strip()] = grade_to_numeric(_.split(',')[1].strip())
		
			for _ in output.split('\n')[-2:]:
				if 'assets' in _.split(',')[0].strip().lower():
					dic[_.split(',')[0].strip()] = float(_.split(',')[1].strip())
				else:
					values.append(float(_.split(',')[1].strip()))
					dic[_.split(',')[0].strip()] = float(_.split(',')[1].strip())

		except Exception as e:
			return "enter the right grades please! (A to F)"

		return values, dic

	except Exception as e:
		print(f"Error generating response: {e}")
		return 'Try again'

def wrap_it(roi_end_of_year):
	prompt_init = """ 

	You will be given the month end trailing returns, year 1 of a fund. This has been calculated by a KNN model trained on an extensive dataset of ESG parameters for the company.

	Your job is to tell the user how the fund will likely perform after the year (you have the value, just add some english around it). 

	Explain to them what this means. Think of the user as a potential investor and explain to them what this value means given the company's ESG data.

	"""

	prompt = prompt_init + f"This is the month end trailing returns, year 1: {roi_end_of_year}"

	try:
		output = ''
		response = chat.send_message(prompt, stream=False, safety_settings={
				HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE, 
				HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
				HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
				HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE
		})

		# Sometimes the model acts up...
		if not response:
			raise ValueError("No response received")

		for chunk in response:
			if chunk.text:
				output += str(chunk.text)

		return output

	except Exception as e:
		print(f"Error generating response: {e}")
		return 'Try again'

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/process-input', methods=['POST'])
def predictor():
	user_input_raw = request.json.get('message')

	if not user_input_raw:
		return jsonify({"error": "No input provided"}), 400

	try:
		extracted_list, dic = extract_list(user_input_raw)

		# print(dic)
		# print(extracted_list)

		knn_model, scaler = load_models()
		roi_end_of_year = predict(extracted_list, knn_model, scaler)

		processed_output = wrap_it(roi_end_of_year)

		print(processed_output)
		info = {'user_input': dic, 'processed_data': processed_output, 'returns': float(roi_end_of_year)}
		response = supabase.table('financial_data').insert(info).execute()

		# print(response)
		return jsonify({"response": processed_output + f"\n Data has been added to database!"})
		# return processed_output

	except Exception as e:
		# return e
		return jsonify({"response": f"Something went wrong: {e}"})
@app.route('/optimise', methods=['POST'])
def MIP_programmer():
    should_we_optimise = request.json.get('message')

    if not should_we_optimise:
        return jsonify({"error": 'No input provided'}), 400

    try:
        if 'True' in should_we_optimise:
            # Fetch company data
            companies_data = fetch_company_data(supabase)
            print(companies_data)
            if not companies_data:
                print("No company data available for optimization")
                return jsonify({"error": "No company data available for optimization"}), 404

            # Run optimization
            total_budget = 1000000  # Example budget
            optimal_portfolio = optimize_portfolio(companies_data, total_budget)

            # Print analysis
            output = print_portfolio_analysis(optimal_portfolio)
            return jsonify({"response": output})
        else:
            return jsonify({"error": "Invalid message content"}), 400
    except Exception as e:
        return jsonify({"error": f'Error: {e} happened'}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use PORT environment variable if available
    app.run(host="0.0.0.0", port=port, debug=True)

# print(predictor('Relative carbon footprint is around 56.1, Fossil Fuel Grade is A, Deforestation D, Prison Grade is C, assets worth 1 million dollars ,Military Grade is F, Tobacco is F'))