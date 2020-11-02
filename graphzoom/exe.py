# -*- coding: utf-8 -*-
# @Time    : 2020-10-04 15:35
# @Author  : xiaorui su
# @Email   :  suxiaorui19@mails.ucas.edu.cn
# @File    : exe.py
# @Software : PyCharm

import os

'''
parser = ArgumentParser(description="GraphZoom")
    parser.add_argument("-d", "--dataset", type=str, default="cora", \
            help="input dataset")
    parser.add_argument("-o", "--coarse", type=str, default="simple", \
            help="choose either simple_coarse or lamg_coarse, [simple, lamg]")
    parser.add_argument("-c", "--mcr_dir", type=str, default="/opt/matlab/R2018A/", \
            help="directory of matlab compiler runtime (only required by lamg_coarsen)")
    parser.add_argument("-s", "--search_ratio", type=int, default=12, \
            help="control the search space in graph fusion process (only required by lamg_coarsen)")
    parser.add_argument("-r", "--reduce_ratio", type=int, default=2, \
            help="control graph coarsening levels (only required by lamg_coarsen)")
    parser.add_argument("-v", "--level", type=int, default=1, \
            help="number of coarsening levels (only required by simple_coarsen)")
    parser.add_argument("-n", "--num_neighs", type=int, default=2, \
            help="control k-nearest neighbors in graph fusion process")
    parser.add_argument("-l", "--lda", type=float, default=0.1, \
            help="control self loop in adjacency matrix")
    parser.add_argument("-e", "--embed_path", type=str, default="embed_results/embeddings.npy", \
            help="path of embedding result")
    parser.add_argument("-m", "--embed_method", type=str, default="deepwalk", \
            help="[deepwalk, node2vec, graphsage]")
    parser.add_argument("-f", "--fusion", default=True, action="store_false", \
            help="whether use graph fusion")
    parser.add_argument("-p", "--power", default=False, action="store_true", \
            help="Strong power of graph filter, set True to enhance filter power")
    parser.add_argument("-g", "--sage_model", type=str, default="mean", \
            help="aggregation function in graphsage")
    parser.add_argument("-w", "--sage_weighted", default=True, action="store_false", \
            help="whether consider weighted reduced graph")'''



dataset = "ppi"
embedding_method = "deepwalk"

#Performance on three large-scale data sets

def experiment_1(dataset,coarse="simple",search_ratio=2,level=1,num_neighs=12,lda= 0.1):
    #dataset = "ppi"
    embedding_method = "deepwalk"

    query = "python graphzoom.py --dataset " + dataset + " --search_ratio " + str(search_ratio)\
            + " --level " + str(level) + " --num_neighs " + str(num_neighs) + " --lda " + str(lda) + " --coarse " + coarse\
            +" --embed_method " + embedding_method
    os.system(query)
    return ;
'''
experiment_1("ppi")
experiment_1("gppi")'''

def experiment_2(dataset,coarse="simple",search_ratio=2,level=1,num_neighs=12,lda= 0.1):
    #dataset = "ppi"
    embedding_method = "node2vec"
    embed_path = "embed_results/" + str(dataset) + "_" + str(embedding_method) + ".npy"

    query = "python graphzoom.py --dataset " + dataset + " --search_ratio " + str(search_ratio)\
            + " --level " + str(level) + " --num_neighs " + str(num_neighs) + " --lda " + str(lda) + " --coarse " + coarse\
            +" --embed_method " + embedding_method + " --embed_path " + embed_path
    os.system(query)
    return ;

#parameter

def experiment_3(dataset,level,coarse="simple",search_ratio=2,num_neighs=12,lda= 0.1):
    #dataset = "ppi"
    embedding_method = "deepwalk"
    embed_path = "embed_results/" + str(dataset) + "_" + str(embedding_method) + ".npy"

    for i in level:
        print(i)
        query = "python graphzoom.py --dataset " + dataset + " --search_ratio " + str(search_ratio) \
                + " --level " + str(i) + " --num_neighs " + str(num_neighs) + " --lda " + str(
            lda) + " --coarse " + coarse \
                + " --embed_method " + embedding_method;
        os.system(query)


    return ;


def experiment_3_1(dataset,lda,coarse="simple",search_ratio=2,level=1,num_neighs=12):
    #dataset = "ppi"
    embedding_method = "deepwalk"
    embed_path = "embed_results/" + str(dataset) + "_" + str(embedding_method) + ".npy"

    for i in lda:
        query = "python graphzoom.py --dataset " + dataset + " --search_ratio " + str(search_ratio) \
                + " --level " + str(level) + " --num_neighs " + str(num_neighs) + " --lda " + str(
            i) + " --coarse " + coarse \
                + " --embed_method " + embedding_method;
        os.system(query)
        print("level" + str(i))

    return ;


'''

experiment_2("gppi")
experiment_2("ppi")'''

'''
experiment_1("gppi")
experiment_1("ppi")'''


level = [4,5]
#experiment_3("gppi",level)
#experiment_3("ppi", level)

lda = [1]
#experiment_3_1("gppi",lda)
experiment_3_1("ppi",lda)