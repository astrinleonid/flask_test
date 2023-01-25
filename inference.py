import numpy as np
import pandas as pd
import pickle
import json
# from IPython.display import display
from sklearn.neighbors import KNeighborsClassifier
from flask import Flask
from flask import request
from flask import jsonify
from flask.ext.cors import CORS, cross_origin

FEATURES = ['is_male','num_inters','late_on_payment','age','years_in_contract']
# Reading datafiles
with open('churn_model.pkl', 'rb') as file:
    clf = pickle.load(file)

features = FEATURES

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'application/json'

# Individual prediction
@app.route('/')
def get_prediction():
    try:
        observation = pd.DataFrame()
        for feature in features:
            observation[feature] = [float(request.args.get(feature))]
        prediction = str(clf.predict(observation)[0])
        return f"Predicted value of churn: {prediction}"
    except:
        return

#Bulk prediction
@app.route('/predict_churn_bulk',methods=['POST'])
def get_bulk_predictions():
    try:
        observations = pd.DataFrame(json.loads(request.get_json()))
        result = {'Prediction' : list(clf.predict(observations).astype(str))}
        return jsonify(result)
    except:
        return

@app.route("/get_event", methods=['POST', 'OPTIONS'])
def get_event():

    try:
        event_id = request.args.get("event_id")
        city = request.args.get("city")

        # db.table_get_value_with_ID(self, table, event_id, columns)
        data = {
            "Header": "Sample Event Header",
            "Description": "Sample Event Description",
            "Date": "Sample Event Date",
            "Genre": "Sample Event Genre",
            "Price": "Sample Event Price",
            "Location": "Sample Event Location",
        }
        print(f"recived: event_id={event_id}")
        data = [1,2,3,4]
        return jsonify(data)
    except:
        return



app.run(host='0.0.0.0', port=3000)


