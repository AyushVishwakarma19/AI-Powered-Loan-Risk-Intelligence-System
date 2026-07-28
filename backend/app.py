from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from groq import Groq
import json
import psycopg2
import os

# with open("../models/loan_risk_model.pkl", "rb") as file:
#     model = pickle.load(file)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "loan_risk_model.pkl"
)

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# print(model.feature_names_in_)

app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static"
)

from dotenv import load_dotenv
load_dotenv()
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    conn = psycopg2.connect(DATABASE_URL)
else:
    conn = psycopg2.connect(
        host="localhost",
        database="loan_risk_db",
        user="postgres",
        password="zeblo@n/m"
    )

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS prediction_history (

    id SERIAL PRIMARY KEY,

    age INTEGER,
    income NUMERIC,
    loan_amount NUMERIC,
    credit_score INTEGER,
    loan_term INTEGER,
    employment_type VARCHAR(50),

    risk_score INTEGER,
    risk_category VARCHAR(50),
    default_probability NUMERIC,

    ai_decision VARCHAR(50),
    ai_recommendation TEXT,

    prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()


def create_features(
    age,
    income,
    loan_amount,
    credit_score,
    loan_term,
    employment_type
):

    loan_to_income_ratio = loan_amount / income

    feature_dict = {}

    feature_dict["Age"] = age
    feature_dict["Income"] = income
    feature_dict["LoanAmount"] = loan_amount
    feature_dict["CreditScore"] = credit_score
    feature_dict["LoanTerm"] = loan_term
    feature_dict["LoanToIncomeRatio"] = loan_to_income_ratio

    # Employment Type
    feature_dict["EmploymentType_Part-time"] = 1 if employment_type == "Part-time" else 0
    feature_dict["EmploymentType_Self-employed"] = 1 if employment_type == "Self-employed" else 0
    feature_dict["EmploymentType_Unemployed"] = 1 if employment_type == "Unemployed" else 0

    # AgeGroup
    feature_dict["AgeGroup_26-35"] = 1 if 26 <= age <= 35 else 0
    feature_dict["AgeGroup_36-45"] = 1 if 36 <= age <= 45 else 0
    feature_dict["AgeGroup_46-55"] = 1 if 46 <= age <= 55 else 0
    feature_dict["AgeGroup_56+"] = 1 if age > 55 else 0

    # IncomeGroup
    feature_dict["IncomeGroup_Low Income"] = 1 if income < 40000 else 0

    feature_dict["IncomeGroup_Medium Income"] = (
        1 if 40000 <= income < 80000 else 0
)

    feature_dict["IncomeGroup_Very High Income"] = (
        1 if income >= 120000 else 0
)

    # CreditRiskBand
    feature_dict["CreditRiskBand_Fair"] = 1 if 580 <= credit_score <= 669 else 0
    feature_dict["CreditRiskBand_Good"] = 1 if 670 <= credit_score <= 739 else 0
    feature_dict["CreditRiskBand_Poor"] = 1 if credit_score < 580 else 0

    # Loan Size
    # Loan Size

    feature_dict["LoanSizeCategory_Small"] = (
        1 if loan_amount < 50000 else 0
)

    feature_dict["LoanSizeCategory_Medium"] = (
        1 if 50000 <= loan_amount < 100000 else 0
)

    feature_dict["LoanSizeCategory_Very Large"] = (
        1 if loan_amount >= 150000 else 0
)

    # Risk Tier
    feature_dict["RiskTier_High Risk"] = (
        1 if credit_score < 580 or loan_to_income_ratio > 0.5 else 0
    )

    feature_dict["RiskTier_Low Risk"] = (
        1 if credit_score > 670 and loan_to_income_ratio <= 0.5 else 0
    )

    feature_dict["RiskTier_Medium Risk"] = (
        1 if 580 <= credit_score <= 670 else 0
    )

    return pd.DataFrame([feature_dict])

def generate_ai_insight(
    age,
    income,
    loan_amount,
    credit_score,
    risk_score,
    risk_category,
    default_probability
):

    prompt = f"""
You are a senior credit risk analyst.

Return ONLY valid JSON:

{{
  "explanation": "short risk explanation",
  "decision": "Approve or Review or Reject",
  "recommendation": "one line recommendation"
}}

Customer:

Age: {age}
Income: {income}
Loan Amount: {loan_amount}
Credit Score: {credit_score}

Risk Score: {risk_score}
Risk Category: {risk_category}
Default Probability: {default_probability}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return json.loads(
        response.choices[0].message.content
    )

def save_prediction(
    age,
    income,
    loan_amount,
    credit_score,
    loan_term,
    employment_type,
    risk_score,
    risk_category,
    default_probability,
    ai_decision,
    ai_recommendation
):

    cursor.execute(
        """
        INSERT INTO prediction_history (

            age,
            income,
            loan_amount,
            credit_score,
            loan_term,
            employment_type,

            risk_score,
            risk_category,
            default_probability,

            ai_decision,
            ai_recommendation

        )

        VALUES (
            %s,%s,%s,%s,%s,%s,
            %s,%s,%s,
            %s,%s
        )
        """,

        (
            age,
            income,
            loan_amount,
            credit_score,
            loan_term,
            employment_type,

            risk_score,
            risk_category,
            default_probability,

            ai_decision,
            ai_recommendation
        )
    )

    conn.commit()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/history")
def history():

    cursor.execute("""
        SELECT
            id,
            age,
            income,
            loan_amount,
            credit_score,
            risk_score,
            risk_category,
            default_probability,
            ai_decision,
            prediction_time
        FROM prediction_history
        ORDER BY id DESC
        LIMIT 20
    """)

    rows = cursor.fetchall()

    print(rows)

    return render_template(
        "history.html",
        predictions=rows
    )

@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        age = int(request.form["age"])
        income = float(request.form["income"])
        loan_amount = float(request.form["loan_amount"])
        loan_term = int(request.form["loan_term"])
        credit_score = int(request.form["credit_score"])
        employment_type = request.form["employment_type"]

        X = create_features(
            age,
            income,
            loan_amount,
            credit_score,
            loan_term,
            employment_type
        )

        # print(X.columns.tolist())
        X = X[model.feature_names_in_]

        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]

        # Risk Category
        # Risk categories calibrated using model prediction distribution
        if probability < 0.35:
            risk_category = "Low Risk"
        elif probability < 0.50:
            risk_category = "Medium Risk"
        else:
            risk_category = "High Risk"

        # Risk Card Color
        if risk_category == "Low Risk":
            risk_color = "low"

        elif risk_category == "Medium Risk":
            risk_color = "medium"

        else:
            risk_color = "high"    

        # Risk Score
        risk_score = int((1 - probability) * 1000)
        progress_width = risk_score /10

        # Default Probability %
        default_probability = round(probability * 100, 2)

        ai_insight = generate_ai_insight(
            age,
            income,
            loan_amount,
            credit_score,
            risk_score,
            risk_category,
            default_probability
        )

        print("AI INSIGHT:")
        print(ai_insight)

        # AI Recommendation
        risk_factors = []
        if risk_category == "Low Risk":
            
            risk_factors = [
                "Excellent credit history",
                "Strong income relative to loan amount",
                "Low predicted default probability"
            ]    

            # ai_recommendation = (
            #     "Applicant demonstrates strong repayment potential. "
            #     "Loan can proceed through standard approval workflow."
            # )

        elif risk_category == "Medium Risk":

            risk_factors = []

            if credit_score < 700:
                risk_factors.append("Credit score below preferred range")

            if loan_amount / income > 0.30:
                risk_factors.append("Loan-to-income ratio is elevated")

                risk_factors.append(
                "Additional verification of income and employment recommended" )
            
            # ai_recommendation = (
            #     "Additional verification is recommended. "
            #     "Review employment stability and repayment capacity."
            # )

        else:

            risk_factors = []

            if credit_score < 580:
                risk_factors.append(f"Credit score of {credit_score} falls within the Poor range")

            if loan_amount / income > 0.50:
                risk_factors.append(
                    f"Loan amount is {round(loan_amount/income,1)}x annual income"
                )

            risk_factors.append(
                f"Predicted default probability reached {default_probability}%"
            )

            # ai_recommendation = (
            #     "High default risk detected. "
            #     "Manual underwriting review is strongly recommended."
            # )

        if len(risk_factors) == 0:
           risk_factors.append(
               "Borrower profile does not exhibit any major risk indicators requiring additional review."
            )
        
        print(X)
        print("Prediction:", prediction)
        print("Probability:", probability)

        print("Risk Category:", risk_category)
        print("Risk Score:", risk_score)
        print("Default Probability:", default_probability)
        # print("AI Recommendation:", ai_recommendation)
        print("AI Explanation:", ai_insight["explanation"])
        print("AI Decision:", ai_insight["decision"])
        print("AI Recommendation:", ai_insight["recommendation"])

        print("Risk Factors:", risk_factors)

        save_prediction(
            age,
            income,
            loan_amount,
            credit_score,
            loan_term,
            employment_type,
            risk_score,
            risk_category,
            default_probability,
            ai_insight["decision"],
            ai_insight["recommendation"]
        )

        return render_template(
            "predict.html",
            prediction=prediction,
            probability=probability,
            risk_category=risk_category,
            risk_score=risk_score,
            default_probability=default_probability,
            # ai_recommendation=ai_recommendation,
            progress_width=progress_width,
            risk_color=risk_color,
            risk_factors=risk_factors,
            ai_explanation=ai_insight["explanation"],
            ai_decision=ai_insight["decision"],
            ai_llm_recommendation=ai_insight["recommendation"]
        )


    return render_template("predict.html")

if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host="0.0.0.0", port=5000)