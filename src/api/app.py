from flask import Flask, render_template,request
# import joblib
import pickle
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.model_utils import *

app = Flask(__name__, template_folder='templates')
model = pickle.load(open('model/churn_detection_clf.sav', 'rb'))

@app.route("/", methods=['GET', 'POST'])
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
            'Complain': 1 if request.form.get('Complain', type=int) == 'on' else 0,
            'Satisfaction_Score': request.form.get('Satisfaction_Score', type=int),
            'Point_Earned': request.form.get('Point_Earned', type=int),
            'Geography': request.form.get('Geography', type=str),
            'Card_Type': request.form.get('Card_Type', type=str),
        }
        
        X = pd.DataFrame([data])
        X = scaler_norm(X)
        
        prediction = model.predict(X)
        prediction = 'Ativo' if prediction[0] == 1 else 'Cancelado'
        return render_template('form.html', result=prediction)
    return render_template('form.html', result='')


if __name__ in '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))