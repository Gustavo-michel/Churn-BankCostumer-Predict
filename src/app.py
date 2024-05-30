from cgitb import text
from flask import Flask, render_template,request
import joblib
# import pickle
from matplotlib import texmanager
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.model_utils import *

app = Flask(__name__)
# secret_key = '7829'
# app.config['SECRET_KEY'] = secret_key
model = joblib.load('notebooks/model/churn_detection_clf.sav')

@app.route("/", methods=['POST', 'GET'])
def form():
    if request.method == 'POST':
        data = {
            'CreditScore': request.form.get('CreditScore', type=int),
            'Tenure': request.form.get('Tenure', type=int),
            'Age': request.form.get('Age', type=int),
            'Balance': request.form.get('Balance', type=float),
            'NumOfProducts': request.form.get('NumOfProducts', type=int),
            'HasCrCard': 1 if request.form.get('HasCrCard') == 'on' else 0,
            'IsActiveMember': 1 if request.form.get('IsActiveMember') == 'on' else 0,
            'EstimatedSalary': request.form.get('EstimatedSalary', type=float),
            'Complain': request.form.get('Complain', type=int),
            'Satisfaction_Score': request.form.get('Satisfaction_Score', type=int),
            'Point_Earned': request.form.get('Point_Earned', type=int),
            'Geography': request.form.get('Geography', type=str),
            'Card_Type': request.form.get('Card_Type', type=str),
        }

        X = pd.DataFrame([data])
        X = scaler_norm(X)
        
        prediction = model.predict(X)
        prediction = 'Fraude' if prediction[0] == 1 else 'Não Fraude'
        return render_template('form.html', result=prediction)
    return render_template('form.html', result='')


if __name__ in '__main__':
    app.run(debug=True)