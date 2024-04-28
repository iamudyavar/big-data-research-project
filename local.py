from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import time
import numpy as np
def nonSpark_heart():
    print("heart dataset")
    start = time.time() * 1000
    # print(start)
    x = np.loadtxt('resources/heart/heart.dat')
    x = pd.DataFrame(x)
    y = x.iloc[:, -1:]
    x = x.iloc[:, :-1]
    # print(x)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y.values.ravel(), test_size=.3)
    gnb = GaussianNB()
    # print(xtrain)
    y_pred = gnb.fit(xtrain, ytrain).predict(xtest)
    miss = (ytest != y_pred).sum()
    print("Number of mislabled points out of a total %d points: %d" % (xtest.shape[0], miss))
    accuracy = 1 - (miss / xtest.shape[0])
    print("Accuracy: %d", accuracy)
    elapsed = time.time() * 1000 - start
    print("took %d ms" % elapsed)
    return [elapsed, accuracy]


def nonSpark_airline():
    print('airline dataset')
    start = time.time() * 1000
    # print(start)
    x = pd.read_csv('resources/airline/train.csv')
    x = x.replace('Male', 0.0)
    x = x.replace('Female', 1.0)
    x = x.replace('Loyal Customer', 1.0)
    x = x.replace('disloyal Customer', 0.0)
    x = x.replace('Business travel', 1.0)
    x = x.replace('Personal Travel', 0.0)
    x = x.replace('Eco', 0.0)
    x = x.replace('Eco Plus', 1.0)
    x = x.replace('Business', 2.0)
    x = x.replace('neutral or dissatisfied', 0.0)
    x = x.replace('satisfied', 1.0)
    y = x.iloc[:, -1:]
    x = x.iloc[:, :-1]
    x = x.fillna(0)
    # print(x, y)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y.values.ravel(), test_size=.3)
    gnb = GaussianNB()
    # print(xtrain)
    # print(np.any(np.isnan(xtrain)))
    # print(np.any(np.isfinite(xtrain)))
    # print(xtrain.size)
    y_pred = gnb.fit(xtrain, ytrain).predict(xtest)
    miss = (ytest != y_pred).sum()
    print("Number of mislabled points out of a total %d points: %d" % (xtest.shape[0], miss))
    accuracy = 1 - (miss / xtest.shape[0])
    print("Accuracy: %d", accuracy)
    elapsed = time.time() * 1000 - start
    print("took %d ms" % elapsed)
    return [elapsed, accuracy]

def nonSpark_fraud():
    print('fraud dataset')
    start = time.time() * 1000
    fileLocation = 'resources/fraud/fraudTrain.csv'
    x = pd.read_csv(fileLocation)
    # print(x.columns)
    y = x.iloc[::, -1:]
    # print(y)
    x = x.iloc[::, 1:-1]
    # print(x)
    # print(x['trans_date_trans_time'].nunique())
    # print(x.groupby('trans_date_trans_time').ngroup())
    cols_to_encode = ['trans_date_trans_time', 'merchant', 'category', 'first', 'last', 'gender', 'street', 'city',
                      'state', 'job', 'dob', 'trans_num']
    for val in cols_to_encode:
        x[val] = x.groupby(val).ngroup()

    xtrain, xtest, ytrain, ytest = train_test_split(x, y.values.ravel(), test_size=.3)
    gnb = GaussianNB()
    # print(xtrain)

    y_pred = gnb.fit(xtrain, ytrain).predict(xtest)
    miss = (ytest != y_pred).sum()
    print("Number of mislabled points out of a total %d points: %d" % (xtest.shape[0], miss))
    accuracy = 1 - (miss/xtest.shape[0])
    print("Accuracy: %d", accuracy)
    elapsed = time.time() * 1000 - start
    print("took %d ms" % elapsed)
    return [elapsed, accuracy]

def nonSpark_particle():
    fileLocation = 'resources/particle/SUSY.csv'
    start = time.time() * 1000
    columns = ['class', 'lepton 1 pT', 'lepton 1 eta', 'lepton 1 phi', 'lepton 2 pT', 'lepton 2 eta', 'lepton 2 phi',
               'missing energy magnitude', 'missing energy phi', 'MET_rel', 'axial MET', 'M_R', 'M_TR_2', 'R', 'MT2',
               'S_R', 'M_Delta_R', 'dPhi_r_b', 'cos(theta_r1)']
    x = pd.read_csv(fileLocation, names=columns)
    y = x.iloc[::, 0:1]
    x = x.iloc[::, 1:]
    # print(y.iloc[0])
    # print(x.iloc[0])
    xtrain, xtest, ytrain, ytest = train_test_split(x, y.values.ravel(), test_size=.3)
    gnb = GaussianNB()
    # print(xtrain)

    y_pred = gnb.fit(xtrain, ytrain).predict(xtest)
    miss = (ytest != y_pred).sum()
    print("Number of mislabled points out of a total %d points: %d" % (xtest.shape[0], miss))
    accuracy = 1 - (miss / xtest.shape[0])
    print("Accuracy: %d", accuracy)
    elapsed = time.time() * 1000 - start
    print("took %d ms" % elapsed)
    return [elapsed, accuracy]

averages = [[0,0],[0,0],[0,0],[0,0]]
iter_count = 3

for i in range(iter_count):
    holder = nonSpark_heart()
    averages[0][0] = averages[0][0] + holder[0]
    averages[0][1] = averages[0][1] + holder[1]
    print(averages[0])
    holder = nonSpark_airline()
    averages[1][0] = averages[1][0] + holder[0]
    averages[1][1] = averages[1][1] + holder[1]
    holder = nonSpark_fraud()
    averages[2][0] = averages[2][0] + holder[0]
    averages[2][1] = averages[2][1] + holder[1]
    holder= nonSpark_particle()
    averages[3][0] = averages[3][0] + holder[0]
    averages[3][1] = averages[3][1] + holder[1]

print("Averages for execution time and accuracy for Heart, Airline, Fraud and Particle datasets:")

for index in range(len(averages)):
    averages[index][0] = averages[index][0] / iter_count
    averages[index][1] = averages[index][1] / iter_count
    print(averages[index][0], averages[index][1])
