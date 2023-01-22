import numpy as np
import pandas as pd
import pickle
import json
# from IPython.display import display
from sklearn.neighbors import KNeighborsClassifier
from flask import Flask
from flask import request
from flask import jsonify

FEATURES = ['is_male','num_inters','late_on_payment','age','years_in_contract']
# Reading datafiles
with open('churn_model.pkl', 'rb') as file:
    clf = pickle.load(file)

features = FEATURES

app = Flask(__name__)

# Individual prediction
@app.route('/')
def get_prediction():
    observation = pd.DataFrame()
    for feature in features:
        observation[feature] = [float(request.args.get(feature))]
    prediction = str(clf.predict(observation)[0])
    return f"Predicted value of churn: {prediction}"

#Bulk prediction
@app.route('/predict_churn_bulk',methods=['POST'])
def get_bulk_predictions():
    observations = pd.DataFrame(json.loads(request.get_json()))
    result = {'Prediction' : list(clf.predict(observations).astype(str))}
    return jsonify(result)


app.run()


