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

alpha = [0.000001,0.00001]
# alpha = [0.1,.02]
batch = [1080,540]
lamIndex=0
batchIndex=0
a = np.zeros((len(alpha),len(batch)))
f1= 0
for m in range(0,len(batch)):
    for n in range(0,len(alpha)):
        X_train = np.copy(features[:5400, :])
        Y_train = np.copy(classes[:5400, :])
        X_test = np.copy(features[5400:, :])
        Y_test = np.copy(classes[5400:, :])
        cl = np.sum(Y_test,axis=0)
        w_initial = np.zeros(28*5).reshape(28, 5)
        # w_initial = (w_initial.T - w_initial.T[0]).T
        costAr = []
        k = batch[m]
        l = X_train.shape[0]
        costAr.append(0)
        for j in range(1, 10000):
            for i in range(0,k):
                grad  = gradient(X_train[int((l/k)*i):int((l/k)*(i+1)),:] , Y_train[int((l/k)*i):int((l/k)*(i+1)),:], w_initial)
                w_initial = w_initial+0.01*(grad-alpha[n]*w_initial)
                # w_initial = (w_initial.T - w_initial.T[0]).T
            c = cost(w_initial,X_train,Y_train)
            print((j,c), end="\r", flush=True)
            costAr.append(c)
            if(costAr[j-1]-costAr[j]<.00001 and j>10):
                print(costAr[j],costAr[j-1])
                break
        print(j)
        w_initial = (w_initial.T - w_initial.T[0]).T
        WmulX = np.exp(np.matmul(X_test, w_initial))
        denom = np.sum(WmulX, axis=1)
        Y_predict = np.divide(WmulX, denom.reshape(len(X_test), 1))
        b = np.zeros_like(Y_predict)
        b[np.arange(len(Y_predict)), Y_predict.argmax(1)] = 1
        Y_predict = b
        accuracy = np.trace(np.matmul(Y_predict, Y_test.T))/len(X_test)
        Y_predict[:,0] =1*Y_predict[:,0]
        Y_predict[:,1] =2*Y_predict[:,1]
        Y_predict[:,2] =3*Y_predict[:,2]
        Y_predict[:,3] =4*Y_predict[:,3]
        Y_predict[:,4] =5*Y_predict[:,4]
        Y_predict = np.sum(Y_predict,axis=1).tolist()
        Y_test[:,0] =1*Y_test[:,0]
        Y_test[:,1] =2*Y_test[:,1]
        Y_test[:,2] =3*Y_test[:,2]
        Y_test[:,3] =4*Y_test[:,3]
        Y_test[:,4] =5*Y_test[:,4]
        Y_test = np.sum(Y_test,axis=1).tolist()
        mat = np.zeros((5,5))
        f1Score = []
        for i in range(0,len(Y_predict)):
            mat[int(Y_predict[i])-1][Y_test[i]-1]=mat[int(Y_predict[i])-1][Y_test[i]-1]+1
        for i in range(len(mat)):
            TP = mat[i,i]
            FP = np.sum(mat[:,i])-mat[i,i]
            FN = np.sum(mat[i,:])-mat[i,i]
            if(TP==0 and FP==0 and FN==0):
                precision=100
                recall=100
                F1_score=100
            elif(TP==0 and (FP==0 or FN==0)):
                precision=0
                recall=0
                F1_score=0
            else:
                precision = mat[i,i]/np.sum(mat[:,i])
                recall = mat[i,i]/np.sum(mat[i,:])
                F1_score = 2*precision*recall/(precision+recall)*100
            f1Score.append([TP,FP,FN,precision,recall,F1_score])    
        F1Score = pd.DataFrame([],columns=["TP","FP","FN","precision","recall","F1_Score"])
        F1Score.loc["class_not_recom"] = f1Score[0]
        F1Score.loc["class_recommend"] = f1Score[1]
        F1Score.loc["class_very_recom"] = f1Score[2]
        F1Score.loc["class_priority"] = f1Score[3]
        F1Score.loc["class_spec_prior"] = f1Score[4]
        Weighted_F1_score = np.dot(cl,F1Score.iloc[:,5].values)/np.sum(cl)
        Macro_F1_score = np.sum(F1Score.iloc[:,5].values)/len(cl)
        print("micro_F1_score= ",Weighted_F1_score,Macro_F1_score,m,accuracy)
        if(f1<Weighted_F1_score and Weighted_F1_score==Weighted_F1_score):
            print("updated")
            f1= Weighted_F1_score
            lamIndex=n
            batchIndex = m
        a[n][m]=Weighted_F1_score
print(a)
X_train = features[:, :]
Y_train = classes[:, :]
X_test = features[:, :]
Y_test = classes[:, :]
print(X_train.shape,Y_train.shape)
# lamIndex, batchIndex =  np.where(a==np.max(a))[0][0], np.where(a==np.max(a))[1][0]
w_initial = np.zeros(28*5).reshape(28, 5)
costAr = []
k = batch[batchIndex]
l = X_train.shape[0]
costAr.append(0)
for j in range(1, 10000):
    for i in range(0,k):
        grad  = gradient(X_train[int((l/k)*i):int((l/k)*(i+1)),:] , Y_train[int((l/k)*i):int((l/k)*(i+1)),:], w_initial)
        w_initial = w_initial+0.01*(grad-alpha[lamIndex]*w_initial)
    c = cost(w_initial,X_train,Y_train)
    print((j,c), end="\r", flush=True)
    costAr.append(c)
    if(costAr[j-1]-costAr[j]<.00001 and j>1):
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