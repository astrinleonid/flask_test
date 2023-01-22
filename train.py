import numpy as np
import pandas as pd
import pickle
# from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def main():

    # reading data
    df = pd.read_csv('cellular_churn_greece.csv')
    # display(df)

    # Splitting to train and test sets
    target = 'churned'
    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state = 57)
    print(len(X_train), len(X_test), len(y_train), len(y_test))

    # Training k nearest neighbors classifier
    clf = KNeighborsClassifier(3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test,y_pred):.3f}")
    # display(pd.DataFrame({'Test':y_test,'Predict':y_pred}))

    #Writing datafiles
    with open('churn_model.pkl', 'wb') as file:
        pickle.dump(clf, file)
    with open('X_test.csv', 'w') as file:
        X_test.to_csv(file, index = False)
    with open('y_pred.csv', 'w') as file:
        np.savetxt(file, y_pred)


if __name__ == '__main__' :
    main()
