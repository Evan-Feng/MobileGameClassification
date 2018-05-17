from sklearn.preprocessing import Imputer
from sklearn import tree
import numpy as np
import json


def load_dataset(in_path):
    with open(in_path, 'r', encoding='utf-8') as fin:
        return json.load(fin)


def main():
    x_train, y_train, x_test, y_test = load_dataset('toy.json')

    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(x_train)
    x_train = imp.transform(x_train)  
    x_test = imp.transform(x_test)

    clf = tree.DecisionTreeClassifier(max_depth=12)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    accur_res = [i for i in range(len(pred)) if pred[i] == y_test[i]]
    accuracy = len(accur_res) / len(pred)
    print(accuracy)

if __name__ == '__main__':
    main()
