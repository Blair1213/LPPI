# -*- coding: utf-8 -*-
# @Time    : 2020-10-04 19:02
# @Author  : xiaorui su
# @Email   :  suxiaorui19@mails.ucas.edu.cn
# @File    : addsamples.py
# @Software : PyCharm

import numpy as np
dataset = "ppi"
'''
train_path = "./splitdata/" + dataset + "/train_" + str(k) + ".npy"
trainl_path = "./splitdata/" + dataset + "/label/train_" + str(k) + ".npy"
test_path = "./splitdata/" + dataset + "/test_" + str(k) + ".npy"
testl_path = "./splitdata/" + dataset + "/label/test_" + str(k) + ".npy"
train_ids = np.load(train_path)
train_labels = np.load(trainl_path)
test_ids = np.load(test_path)
test_labels = np.load(testl_path)'''

for k in range(0,5):

    test_path = "./splitdata/" + dataset + "/test_" + str(k) + ".npy"
    testl_path = "./splitdata/" + dataset + "/label/test_" + str(k) + ".npy"
    test_ids = np.load(test_path)
    print(len(test_ids))
    test_labels = np.load(testl_path)
    print(len(test_ids))
    negative_numbers = test_ids
    for i in range(0,5):
        if i is not k :
            train_path = "./splitdata/" + dataset + "/train_" + str(i) + ".npy"
            train_ids = np.load(train_path)
            new_samples = train_ids[int(len(train_ids)/2):int(len(train_ids)/2)+int(len(negative_numbers)/4) , :]
            print(len(new_samples))
            negative_labels = np.zeros(len(new_samples))
            test_ids = np.concatenate((test_ids,new_samples),axis=0)
            test_labels = np.array(np.concatenate((test_labels, negative_labels), axis=0))
    print(k)
    print(len(test_ids))
    print(len(test_labels))
    np.save(test_path,test_ids)
    np.save(testl_path,test_labels)




