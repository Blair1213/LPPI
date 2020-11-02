# -*- coding: utf-8 -*-
# @Time    : 2020-10-10 18:11
# @Author  : xiaorui su
# @Email   :  suxiaorui19@mails.ucas.edu.cn
# @File    : classifers.py
# @Software : PyCharm

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve, auc

from sklearn.metrics import roc_auc_score

def run_regression(train_embeds, train_labels, test_embeds, test_labels):

    # nb
    print("nb")
    from sklearn.naive_bayes import GaussianNB
    nb = GaussianNB()
    nb.fit(train_embeds,train_labels)
    nb_results= nb.predict(test_embeds)
    nb_acc = accuracy_score(test_labels,nb_results)
    nb_auc = roc_auc_score(test_labels,nb_results)

    # rf
    print("rf")
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier()
    # from xgboost import XGBClassifier

    # model1 = XGBClassifier()
    rf.fit(np.array(train_embeds), np.array(train_labels))
    rf_results = rf.predict(np.array(test_embeds))
    rf_acc = accuracy_score(test_labels, rf_results)
    rf_auc = roc_auc_score(test_labels, rf_results)
    #y_score1 = model1.predict_proba(np.array(TestFeature))


    #svm
    '''
    from sklearn.svm import SVC
    print("svm")
    svm = SVC(kernel='linear', C=1.0, random_state=0, probability=True)
    svm.fit(np.array(train_embeds), np.array(train_labels).astype(float))
    svm_results = svm.predict(np.array(test_embeds, dtype='float64'))
    svm_acc = accuracy_score(test_labels, svm_results)
    svm_auc = roc_auc_score(test_labels, svm_results)
    #y_score1 = model1.predict_proba(np.array(TestFeature, dtype='float64'))'''

    acc = [nb_acc,rf_acc]
    auc = [nb_auc,rf_auc]

    print(acc)
    print(auc)

    return acc,auc;

def linkprediction(embeds_path,dataset):

    print("%%%%%% Split data %%%%%%")
    print(len(dataset))

    cross_parameter = 5
    acc_all = []
    f1_all = []
    precision_all = []
    sen_all = []
    mcc_all = []
    auc_all = []
    embeds = np.load(embeds_path)

    for k in range(0,cross_parameter):

        #classifers

        train_path = "./splitdata/" + dataset + "/train_" + str(k) + ".npy"
        trainl_path = "./splitdata/" + dataset + "/label/train_" + str(k) + ".npy"
        test_path = "./splitdata/" + dataset + "/test_" + str(k) + ".npy"
        testl_path = "./splitdata/" + dataset + "/label/test_" + str(k) + ".npy"
        train_ids = np.load(train_path)
        train_labels = np.load(trainl_path)
        test_ids = np.load(test_path)
        test_labels = np.load(testl_path)
        train_embeds = [np.concatenate((np.array(embeds[i[0]]), np.array(embeds[i[1]])), axis=0) for i in train_ids]
        test_embeds = [np.concatenate((np.array(embeds[i[0]]), np.array(embeds[i[1]])), axis=0) for i in test_ids]
        acc,auc = run_regression(train_embeds,train_labels,test_embeds,test_labels)
        acc_all.append(acc)
        auc_all.append(auc)

        print(acc_all)
        print(auc_all)


    return acc_all,auc_all;


embeds_path = "embed_results/ppi_deepwalk.npy"
linkprediction(embeds_path,"ppi")

embeds_path = "embed_results/gppi_deepwalk.npy"
linkprediction(embeds_path,"gppi")

