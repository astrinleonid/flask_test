import numpy as np
import pandas as pd
import json
import requests


def main():
    URL = 'http://34.238.42.93:3000'
    BULK_PATH = 'predict_churn_bulk'

    # Reading datafiles
    X_test = pd.read_csv('X_test.csv')
    with open('y_pred.csv', 'r') as file:
        y_saved = np.loadtxt(file)

    # Getting 5 individual predictions (random choice)
    sample = np.random.randint(0,7500,size=5)
    for i in sample:
        args = X_test.iloc[i].to_dict()
        print(requests.get(URL, args).content, f"Stored prediction: {y_saved[i]:.0f}")

    #Getting full prediction of the test sample
    json_body = json.dumps(X_test.to_dict())
    y_pred = json.loads(requests.post(f"{URL}/{BULK_PATH}", json=json_body).content)
    y_pred = np.array(y_pred['Prediction']).astype(int)
    assert (y_pred != y_saved).sum() == 0

if __name__ == '__main__' :
    main()