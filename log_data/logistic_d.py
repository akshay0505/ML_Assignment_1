import pandas as pd
import numpy as np
import os
import sys
def hotencode_train(data):
    train = pd.DataFrame()
    parents = pd.get_dummies(data[0], prefix="parents")
    has_nurs = pd.get_dummies(data[1], prefix="has_nurs")
    form = pd.get_dummies(data[2], prefix="form")
    children = pd.get_dummies(data[3], prefix="children")
    housing = pd.get_dummies(data[4], prefix="housing")
    finance = pd.get_dummies(data[5], prefix="finance")
    social = pd.get_dummies(data[6], prefix="social")
    health = pd.get_dummies(data[7], prefix="health")
    classDistribution = pd.get_dummies(data[8], prefix="class")
    train = pd.concat([parents, has_nurs, form, children, housing,
                    finance, social, health, classDistribution], axis=1)
    cols = ['parents_usual', 'parents_pretentious', 'parents_great_pret', 'has_nurs_proper', 'has_nurs_less_proper', 'has_nurs_improper',
            'has_nurs_critical', 'has_nurs_very_crit',
            'form_complete', 'form_completed', 'form_incomplete', 'form_foster',
            'children_1', 'children_2', 'children_3', 'children_more',
            'housing_convenient', 'housing_less_conv', 'housing_critical',
            'finance_convenient', 'finance_inconv',
            'social_nonprob', 'social_slightly_prob', 'social_problematic',
            'health_recommended', 'health_priority', 'health_not_recom',
            'class_not_recom', 'class_recommend', 'class_very_recom', 'class_priority', 'class_spec_prior']

    return train[pd.Index(cols)]

def hotencode_test(data):
    train = pd.DataFrame()
    parents = pd.get_dummies(data[0], prefix="parents")
    has_nurs = pd.get_dummies(data[1], prefix="has_nurs")
    form = pd.get_dummies(data[2], prefix="form")
    children = pd.get_dummies(data[3], prefix="children")
    housing = pd.get_dummies(data[4], prefix="housing")
    finance = pd.get_dummies(data[5], prefix="finance")
    social = pd.get_dummies(data[6], prefix="social")
    health = pd.get_dummies(data[7], prefix="health")
    train = pd.concat([parents, has_nurs, form, children, housing,
                    finance, social, health], axis=1)
    cols = ['parents_usual', 'parents_pretentious', 'parents_great_pret', 'has_nurs_proper', 'has_nurs_less_proper', 'has_nurs_improper',
            'has_nurs_critical', 'has_nurs_very_crit',
            'form_complete', 'form_completed', 'form_incomplete', 'form_foster',
            'children_1', 'children_2', 'children_3', 'children_more',
            'housing_convenient', 'housing_less_conv', 'housing_critical',
            'finance_convenient', 'finance_inconv',
            'social_nonprob', 'social_slightly_prob', 'social_problematic',
            'health_recommended', 'health_priority', 'health_not_recom']
    return train[pd.Index(cols)]
train_data = pd.read_csv(sys.argv[1], header=None)
test_data = pd.read_csv(sys.argv[2], header=None)
train = hotencode_train(train_data)
test = hotencode_test(test_data)
classes = train.iloc[:, -5:].values
features = train.iloc[:, :-5].values
features = np.c_[np.ones(len(train)), features]
# print(X_train.shape, Y_train.shape,X_test.shape)
def gradient(X,Y,W):
    XW = np.exp(np.matmul(X, W))
    denom = np.sum(XW, axis=1)
    Y_predict = np.divide(XW, denom.reshape(X.shape[0], 1))
    return np.matmul(X.transpose(), Y - Y_predict)/X.shape[0]
def cost(W,X,Y):
    XW = np.exp(np.matmul(X,W))
    logTerm = np.log(np.sum(XW,axis=1))
    weightedTerm = np.sum(np.multiply(np.matmul(Y,W.T),X),axis=1)
    error = np.sum(logTerm-weightedTerm)/X.shape[0]
    return error
X_train = np.copy(features[:, :])
Y_train = np.copy(classes[:, :])
X_test = np.copy(features[:, :])
Y_test = np.copy(classes[:, :])
# lamIndex, batchIndex =  np.where(a==np.max(a))[0][0], np.where(a==np.max(a))[1][0]
w_initial = np.zeros(28*5).reshape(28, 5)
costAr = []
# k = batch[batchIndex]
k=300
l = X_train.shape[0]
costAr.append(0)
# lr = alpha[lamIndex]
lr = 0.1
# print(X_train.shape,Y_train.shape,lr,k,a[lamIndex][batchIndex])

for j in range(1, 15000):
    for i in range(0,k):
        grad  = gradient(X_train[int((l/k)*i):int((l/k)*(i+1)),:] , Y_train[int((l/k)*i):int((l/k)*(i+1)),:], w_initial)
        # w_initial = w_initial+lr*(grad-4*w_initial/X_train[int((l/k)*i):int((l/k)*(i+1)),:].shape[0])
        w_initial = w_initial+lr*grad
    c = cost(w_initial,X_train,Y_train)
    print((j,c), end="\r", flush=True)
    costAr.append(c)
    if(costAr[j-1]-costAr[j]<.0000001 and j>1):
        print(costAr[j],costAr[j-1])
        break
print(j)
features = test.iloc[:, :].values
features = np.c_[np.ones(len(test)), features]
X_test = features[:, :]
WmulX = np.exp(np.matmul(X_test, w_initial))
denom = np.sum(WmulX, axis=1)
Y_predict = np.divide(WmulX, denom.reshape(len(X_test), 1))
b = np.zeros_like(Y_predict)
b[np.arange(len(Y_predict)), Y_predict.argmax(1)] = 1
Y_predict = b
Y_predict[:, 0] = 1*Y_predict[:, 0]
Y_predict[:, 1] = 2*Y_predict[:, 1]
Y_predict[:, 2] = 3*Y_predict[:, 2]
Y_predict[:, 3] = 4*Y_predict[:, 3]
Y_predict[:, 4] = 5*Y_predict[:, 4]
Y_predict = np.sum(Y_predict, axis=1).tolist()
predict = []
for i in range(0, len(Y_predict)):
    if(Y_predict[i] == 1):
        predict.append("not_recom")
    elif(Y_predict[i] == 2):
        predict.append("recommend")
    elif(Y_predict[i] == 3):
        predict.append("very_recom")
    elif(Y_predict[i] == 4):
        predict.append("priority")
    elif(Y_predict[i] == 5):
        predict.append("spec_prior")
Y_predict = pd.DataFrame(predict)
Y_predict.to_csv(sys.argv[3], header=False, index=False)
w_initial = pd.DataFrame(w_initial)
w_initial.to_csv(sys.argv[4], header=False, index=False)