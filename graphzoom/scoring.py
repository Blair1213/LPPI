from __future__ import print_function
import json
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
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
    dummy = MultiOutputClassifier(DummyClassifier())
    dummy.fit(train_embeds, train_labels)
    log = MultiOutputClassifier(SGDClassifier(loss="log"), n_jobs=10)
    log.fit(train_embeds, train_labels)

    f1 = 0
    f1s = []
    f1rs = []
    acc = []
    random_acc = []
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
        random_acc.append(accuracy_score(test_labels[:, i], dummy.predict(test_embeds)[:, i]))

    print("Random baseline F1 score", sum(f1rs)/len(f1rs))
    print("Random baseline ACC", sum(random_acc)/len(random_acc))

def lr(dataset_dir, data_dir, dataset):
    print("%%%%%% Starting Evaluation %%%%%%")
    print("Loading data...")
    G      = json_graph.node_link_graph(json.load(open(dataset_dir + "/{}-G.json".format(dataset))))
    labels = json.load(open(dataset_dir + "/{}-class_map.json".format(dataset)))
    #labels = {int(i): l for i, l in labels.items()}
    #print(labels)

    train_ids    = [n for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']]
    test_ids     = [n for n in G.nodes() if G.node[n]['test']]
    test_ids     = test_ids[:1000]
    train_labels = np.array([labels[str(i)] for i in train_ids])
    print(train_labels)
    test_labels  = np.array([labels[str(i)] for i in test_ids])

    embeds       = np.load(data_dir)
    train_embeds = embeds[[id for id in train_ids]]
    test_embeds  = embeds[[id for id in test_ids]]
    print("Running regression..")
    run_regression(train_embeds, train_labels, test_embeds, test_labels)
