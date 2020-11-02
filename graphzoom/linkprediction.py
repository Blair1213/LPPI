# -*- coding: utf-8 -*-
# @Time    : 2020-09-28 11:23
# @Author  : xiaorui su
# @Email   :  suxiaorui19@mails.ucas.edu.cn
# @File    : linkprediction.py
# @Software : PyCharm


from __future__ import print_function
import json
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve, auc

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import math
import pandas as pd
from networkx.readwrite import json_graph


def run_regression(train_embeds, train_labels, test_embeds, test_labels):
    '''
    log         = LogisticRegression(solver='lbfgs', multi_class='auto')
    log.fit(train_embeds, train_labels)
    '''

    np.random.seed(1)
    from sklearn.linear_model import SGDClassifier
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import f1_score
    from sklearn.multioutput import MultiOutputClassifier
    #dummy = MultiOutputClassifier(DummyClassifier())
    #dummy.fit(train_embeds, train_labels)
    #log = MultiOutputClassifier(SGDClassifier(loss="log"), n_jobs=10)
    #log.fit(train_embeds, train_labels)
    log = LogisticRegression(solver='lbfgs', multi_class='auto')
    log.fit(train_embeds,train_labels)

    '''
    for i in range(test_labels.shape[1]):
        #print("F1 score", f1_score(test_labels[:, i], log.predict(test_embeds)[:, i], average="micro"))
        f1s.append(f1_score(test_labels[:, i], log.predict(test_embeds)[:, i], average="micro"))
        #print("Acc", accuracy_score(test_labels[:, i], log.predict(test_embeds)[:, i]))
        acc.append(accuracy_score(test_labels[:, i], log.predict(test_embeds)[:, i]))

    print("F1 score",sum(f1s)/len(f1s))
    print("ACC",sum(acc)/len(acc))

    for i in range(test_labels.shape[1]):
        #print("Random baseline F1 score",
        #      f1_score(test_labels[:, i], dummy.predict(test_embeds)[:, i], average="micro"))
        f1rs.append(f1_score(test_labels[:, i], dummy.predict(test_embeds)[:, i], average="micro"))
        #print("Random baseline Acc", accuracy_score(test_labels[:, i], dummy.predict(test_embeds)[:, i]))
        random_acc.append(accuracy_score(test_labels[:, i], dummy.predict(test_embeds)[:, i]))'''

    #print("Random baseline F1 score", sum(f1rs)/len(f1rs))
    #print("Random baseline ACC", sum(random_acc)/len(random_acc))
    y_predict = log.predict(test_embeds)

    return accuracy_score(test_labels, y_predict),precision_score(test_labels, y_predict),\
           recall_score(test_labels, y_predict),matthews_corrcoef(test_labels, y_predict),\
        roc_auc_score(test_labels, y_predict);

def linkprediction(edges,embeds,dataset):

    print("%%%%%% Split data %%%%%%")
    print(len(edges))

    cross_parameter = 5
    acc_all = []
    f1_all = []
    precision_all = []
    sen_all = []
    mcc_all = []
    auc_all = []

    for k in range(0,cross_parameter):

        '''
        test_ids = [i for i in range(0,len(edges)) if i % cross_parameter == k ]
        train_ids = [i for i in range(0,len(edges)) if i not in test_ids]
        train_embeds = [np.concatenate((np.array(embeds[edges[i][0]]), np.array(embeds[edges[i][1]])), axis=0) for i in train_ids]
        test_embeds = [np.concatenate((np.array(embeds[edges[i][0]]), np.array(embeds[edges[i][1]])), axis=0) for i in test_ids]


        train_embeds = np.array(train_embeds)
        test_embeds = np.array(test_embeds)
        train_labels = np.ones(train_embeds.shape[0])
        test_labels = np.ones(test_embeds.shape[0])

        #generate negative samples
        all_train_ids = np.array([[edges[i][0],edges[i][1]] for i in train_ids])
        negative_ids = []
        negative_embeds = []
        edges_pd = pd.DataFrame(edges)
        for i in all_train_ids[:,0]:
            for j in all_train_ids[:,1]:
                #print(edges_pd[edges_pd[0].isin([i])][1])
                if j not in edges_pd[edges_pd[0].isin([i])][1]:
                    negative_ids.append([i,j])
                    negative_embeds.append([np.concatenate((np.array(embeds[i]), np.array(embeds[j])), axis=0)])
                    if len(negative_ids) == len(train_ids):
                        break
            if len(negative_ids) == len(train_ids):
                break

        negative_embeds = np.array(negative_embeds).reshape(len(negative_embeds),256)
        negative_labels = np.zeros(negative_embeds.shape[0])
        train_embeds = np.array(np.concatenate((train_embeds,negative_embeds),axis=0))
        train_labels = np.array(np.concatenate((train_labels,negative_labels),axis=0))
        train_ids = [[edges[i][0],edges[i][1]] for i in train_ids]
        train_ids = np.array(np.concatenate((np.array(train_ids),np.array(negative_ids)),axis=0))
        test_ids = [[edges[i][0], edges[i][1]] for i in test_ids]

        print("train")
        print(train_labels.shape)
        print("test")
        print(test_labels.shape)

        #save files
        train_path = "./splitdata/"+ dataset +"/train_" + str(k)
        trainl_path = "./splitdata/"+ dataset +"/label/train_" + str(k)
        test_path = "./splitdata/"+ dataset +"/test_" + str(k)
        testl_path = "./splitdata/"+ dataset +"/label/test_" + str(k)
        np.save(train_path, train_ids)
        np.save(trainl_path,train_labels)
        np.save(test_path,test_ids)
        np.save(testl_path,test_labels)'''
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
        acc,pre,sen,mcc,auc = run_regression(train_embeds,train_labels,test_embeds,test_labels)
        acc_all.append(acc)
        precision_all.append(pre)
        sen_all.append(sen)
        mcc_all.append(mcc)
        auc_all.append(auc)

        print(acc_all)
        print(precision_all)
        print(sen_all)
        print(mcc_all)
        print(auc_all)


    return acc_all,precision_all,sen_all,mcc_all,auc_all;

def linkprediction_ogb(split_edge,embeds):

    train_edge, valid_edge, test_edge = split_edge["train"], split_edge["valid"], split_edge["test"]

    train_embeds = [np.concatenate((np.array(embeds[i[0]]), np.array(embeds[i[1]])), axis=0) for i in train_edge['edge']]
    train_labels = np.ones(len(train_embeds))

    test_embeds = [np.concatenate((np.array(embeds[i[0]]), np.array(embeds[i[1]])), axis=0) for i in train_edge['edge']]

    for i in train_edge['edge']:
        train_embeds.append([np.concatenate((np.array(embeds[i[0]]), np.array(embeds[i[1]])), axis=0)])



        train_embeds = [np.concatenate((np.array(embeds[i[0]]), np.array(embeds[i[1]])), axis=0) for i in train_ids]
        test_embeds = [np.concatenate((np.array(embeds[i[0]]), np.array(embeds[i[1]])), axis=0) for i in test_ids]
        acc,pre,sen,mcc,auc = run_regression(train_embeds,train_labels,test_embeds,test_labels)
        acc_all.append(acc)
        precision_all.append(pre)
        sen_all.append(sen)
        mcc_all.append(mcc)
        auc_all.append(auc)

        print(acc_all)
        print(precision_all)
        print(sen_all)
        print(mcc_all)
        print(auc_all)


    return acc_all,precision_all,sen_all,mcc_all,auc_all;


