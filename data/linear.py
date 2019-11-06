import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import LassoLars

if(sys.argv[1]=="a"):
    trainDf= pd.read_csv(sys.argv[2],header=None)
    testDf= pd.read_csv(sys.argv[3],header=None)
    testDf[245] = np.ones(len(testDf))

    trainDf[246] = trainDf[245]
    trainDf[245] = np.ones(len(trainDf))


    X_train = trainDf.iloc[:,:-1].values
    Y_train = trainDf.iloc[:,246].values
    X_test = testDf.iloc[:,:].values
    XtXinv = np.linalg.inv(np.matmul(X_train.transpose(),X_train))
    XtY = np.matmul(X_train.transpose(),Y_train)
    Wopt = np.matmul(XtXinv,XtY)
    Y_pred = np.matmul(X_test,Wopt)
    Wopt = np.roll(Wopt,1)
    np.savetxt(sys.argv[4],Y_pred,delimiter=',')
    np.savetxt(sys.argv[5],Wopt,delimiter=',')
if(sys.argv[1]=="b"):
    trainDf= pd.read_csv(sys.argv[2],header=None)
    testDf= pd.read_csv(sys.argv[3],header=None)
    testDf[245] = np.ones(len(testDf))
    trainDf[246] = trainDf[245]
    trainDf[245] = np.ones(len(trainDf))
    lamda = np.loadtxt(sys.argv[4],delimiter="\n")
    k = 10
    features = trainDf.iloc[:,:-1].values
    labels = trainDf.iloc[:,246].values
    trainLen = len(trainDf)
    lossmin = np.inf
    hyperParam = lamda[0]
    j=0
    for l in lamda:
        j=j+1
        # print(j)
        for i in range(0,k):
            X_test = features[int(i*trainLen/k):int((i+1)*trainLen/k)]
            Y_test = labels[int(i*trainLen/k):int((i+1)*trainLen/k)]
            X_train = np.concatenate((features[:int(i*trainLen/k)],features[int((i+1)*trainLen/k):]),axis=0)
            Y_train = np.concatenate((labels[:int(i*trainLen/k)],labels[int((i+1)*trainLen/k):]),axis=0)
            XtXinv = np.linalg.inv(np.matmul(X_train.transpose(),X_train)+l*np.eye(246))
            XtY = np.matmul(X_train.transpose(),Y_train)
            W = np.matmul(XtXinv,XtY)
            Y_predicted = np.matmul(X_test,W)
            diff = Y_predicted-Y_test
            normError = ((np.linalg.norm(diff)**2)/np.matmul(Y_test.T,Y_test))
            if(normError<lossmin):
                lossmin = normError
                hyperParam = l
    print(hyperParam)
    X_train = features
    Y_train = labels
    X_test = testDf.iloc[:,:].values
    XtXinv = np.linalg.inv(np.matmul(X_train.transpose(),X_train)+l*np.eye(246))
    XtY = np.matmul(X_train.transpose(),Y_train)
    Wopt = np.matmul(XtXinv,XtY)
    Y_pred = np.matmul(testDf.iloc[:,:].values,Wopt)
    Wopt = np.roll(Wopt,1)
    np.savetxt(sys.argv[5],Y_pred,delimiter=',')
    np.savetxt(sys.argv[6],Wopt,delimiter=',')
if(sys.argv[1] == "c"):
    trainDf = pd.read_csv(sys.argv[2], header=None)
    testDf = pd.read_csv(sys.argv[3], header=None)
    labels = trainDf.iloc[:, 245].values
    trainDf = trainDf.drop(columns=[245])
    features = trainDf.iloc[:, :].values
    trainDf[245] = np.log(features[:, 52]+1)
    trainDf[246] = np.log(features[:, 45]+1)
    trainDf[247] = np.log(features[:, 21]+1)
    trainDf[248] = np.log(features[:, 46]+1)
    trainDf[249] = np.log(features[:, 233]+1)
    trainDf[250] = np.log(features[:, 58]+1)
    features = trainDf.iloc[:, :].values
    # feature4 = np.multiply(features[:, 248].reshape(len(trainDf), 1), features)
    # features = np.concatenate((features, feature4), axis=1)
    feature1 = np.multiply(features[:, 247].reshape(len(trainDf), 1), features)
    features = np.concatenate((features, feature1), axis=1)
    feature2 = np.multiply(features[:, 246].reshape(len(trainDf), 1), features)
    features = np.concatenate((features, feature2), axis=1)
    feature3 = np.multiply(features[:, 245].reshape(len(trainDf), 1), features)
    features = np.concatenate((features, feature3), axis=1)
    X_train = features
    Y_train = labels
    reg = LassoLars(alpha=.0003)
    reg.fit(X_train,Y_train)
    Wopt = reg.coef_
    features = testDf.iloc[:, :].values
    testDf[245] = np.log(features[:, 52]+1)
    testDf[246] = np.log(features[:, 45]+1)
    testDf[247] = np.log(features[:, 21]+1)
    testDf[248] = np.log(features[:, 46]+1)
    testDf[249] = np.log(features[:, 233]+1)
    testDf[250] = np.log(features[:, 58]+1)
    # testDf[251] = np.log(features[:, 100]+1)
    features = testDf.iloc[:, :].values
    # feature4 = np.multiply(features[:, 248].reshape(len(testDf), 1), features)
    # features = np.concatenate((features, feature4), axis=1)
    feature1 = np.multiply(features[:, 247].reshape(len(testDf), 1), features)
    features = np.concatenate((features, feature1), axis=1)
    feature2 = np.multiply(features[:, 246].reshape(len(testDf), 1), features)
    features = np.concatenate((features, feature2), axis=1)
    feature3 = np.multiply(features[:, 245].reshape(len(testDf), 1), features)
    features = np.concatenate((features, feature3), axis=1)
    X_test = features
    Y_pred = np.matmul(features, Wopt)
    np.savetxt(sys.argv[4], Y_pred, delimiter=',')    
