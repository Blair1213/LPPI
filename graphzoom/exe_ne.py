# -*- coding: utf-8 -*-
# @Time    : 2020-10-08 19:30
# @Author  : xiaorui su
# @Email   :  suxiaorui19@mails.ucas.edu.cn
# @File    : exe_ne.py
# @Software : PyCharm

import os

#graph embedding

def experiment_4(dataset,level=1,coarse="simple",search_ratio=2,num_neighs=12,lda= 0.1):
    #dataset = "ppi"
    embedding_method = "deepwalk"
    embed_path = "embed_results/" + str(dataset) + "_" + str(embedding_method) + str("alone") + ".npy"

    query = "python ne_alone.py --dataset " + dataset + " --search_ratio " + str(search_ratio) \
            + " --level " + str(level) + " --num_neighs " + str(num_neighs) + " --lda " + str(lda) + " --coarse " + coarse \
            + " --embed_method " + embedding_method;
    os.system(query)


    return ;


def experiment_4_1(dataset,level=1,coarse="simple",search_ratio=2,num_neighs=12,lda= 0.1):
    #dataset = "ppi"
    embedding_method = "node2vec"
    embed_path = "embed_results/" + str(dataset) + "_" + str(embedding_method) + "_" + str("alone") + ".npy"

    query = "python ne_alone.py --dataset " + dataset + " --search_ratio " + str(search_ratio) \
            + " --level " + str(level) + " --num_neighs " + str(num_neighs) + " --lda " + str(lda) + " --coarse " + coarse \
            + " --embed_method " + embedding_method;
    os.system(query)



    return ;


#experiment_4("gppi")
experiment_4_1("ppi")