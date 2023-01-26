import numpy as np
import pandas as pd
import pickle
import json
# from IPython.display import display
from sklearn.neighbors import KNeighborsClassifier
from flask import Flask
from flask import request
from flask import jsonify
from db_class import StorageDatabase
from prediction import model

# db = StorageDatabase("sql11.freemysqlhosting.net","3306","sql11593194","sql11593194","2CPjwjQHDQ")
# df_NYC = pd.read_sql("SELECT * FROM NYC", db.__connection__)
# df_MSK = pd.read_sql("SELECT * FROM MSK", db.__connection__)
# df_TLV = pd.read_sql("SELECT * FROM TLV", db.__connection__)
# db.__connection__.close()
df_NYC =

df_TLV =

city_data = {'TLV' : df_TLV, 'MSK' : df_MSK, 'NYC' : df_NYC}


FEATURES = ['is_male','num_inters','late_on_payment','age','years_in_contract']
# Reading datafiles
with open('churn_model.pkl', 'rb') as file:
    clf = pickle.load(file)

features = FEATURES

app = Flask(__name__)


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

@app.route("/get_event")
def get_event():

    try:
        event_id = int(request.args.get("event_id"))
        home_city = request.args.get("home_city")
        target_city = request.args.get("target_city")
        print(f"Home {home_city} Target {target_city} ID {event_id}")

        dfr = city_data[home_city]
        sample_text = dfr.loc[event_id, 'description']

        print(sample_text[:10])
        result = model(city_data[target_city], sample_text, 20)[['id', 'similarity_percentage']]
        data = result.sort_values('similarity_percentage', ascending=False)
        data = data.to_dict(orient='records')
        print(data)
        return jsonify(data)
    except:
        return

# app.run(host='0.0.0.0', port=3000)
app.run()


