from flask import Flask, request, jsonify
from flask_cors import CORS

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use PORT environment variable if available
    app.run(host="0.0.0.0", port=port, debug=True)

# print(predictor('Relative carbon footprint is around 56.1, Fossil Fuel Grade is A, Deforestation D, Prison Grade is C, assets worth 1 million dollars ,Military Grade is F, Tobacco is F'))