import numpy as np
import pandas
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def loadDataSet(filePath):
    data_set = pandas.read_csv(filePath, sep='\t', header=None)

    x = data_set.iloc[:, [0, 1]].values
    y = data_set.iloc[:, -1].values

    data_mat = x.tolist()
    label_mat = y.tolist()
    return data_mat, label_mat

if __name__ == '__main__':

    dataMat, classLabels = loadDataSet('testSetRBF.txt')

    model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=20)
    model.fit(dataMat, classLabels)

    testDataMat, testLabelMat = loadDataSet('testSetRBF2.txt')

    predictResult = model.predict(testDataMat)

    print(model.score(testDataMat, testLabelMat))
    count = 0.0
    for i in range(len(predictResult)):
        if predictResult[i] == testLabelMat[i]:
            count += 1

    print(count / len(predictResult))
