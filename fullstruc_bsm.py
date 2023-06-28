#binary stress model algorithm utilised for display of structural layout of temporal network generated from team datasets
import networkx as nx
import matplotlib.pyplot as plt
import warnings
from sys import argv
import pandas as pd
import math
from scipy.sparse.linalg import cg
import time

#https://github.com/tomzhch/IES-Backbone/blob/master/bsm.py

import matplotlib
from teneto import TemporalNetwork, networkmeasures
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
from matplotlib.pyplot import figure, text
from temporal_graph import *

def expand_df(df):
    new_df=pd.DataFrame()
    for index, rows in df.iterrows():
        string=rows.To
        if len(string)>1 and string!="all":
            string_lst=[*string]
            for ele in string_lst:
                if ele.isalpha():
                    new_df=pd.concat([new_df,df.iloc[[index]].assign(To=ele)], ignore_index=True)
        else:
            new_df=pd.concat([new_df,df.iloc[[index]]], ignore_index=True)
        
    return new_df

def expand_df2(df):
    nodelist.remove('all')
    new_df=pd.DataFrame()
    for index, rows in df.iterrows():
        string=rows.To
        if len(string)>1 and string!="all":
            string_lst=[*string]
            for ele in string_lst:
                if ele.isalpha():
                    new_df=pd.concat([new_df,df.iloc[[index]].assign(To=ele)], ignore_index=True)
        elif string=="all":
            string_lst=nodelist
            for ele in string_lst:
                if ele!=rows.From:
                    new_df=pd.concat([new_df,df.iloc[[index]].assign(To=ele)], ignore_index=True)
        else:
            new_df=pd.concat([new_df,df.iloc[[index]]], ignore_index=True)
        
    return new_df

def timed (s):

    try:
        a = str(s).strip().split('.')
        b = int(a[0]) * 60.0*60.0 + int(a[1])*60.0 + int(a[2])
    except:
        return -1

    return b

def format_metric(df):
    new_df=pd.DataFrame()
    from_lst=[]
    to_lst=[]
    time_discrete=[]
    start=0
    for i in range(len(df['From'])):
        from_lst.append(df['From'][i])
        to_lst.append(df['To'][i])
        if i==0:
            time_discrete.append(start)
        else:
            if df['In'][i]==df['In'][i-1]:
                time_discrete.append(start)
            else:
                start+=1
                time_discrete.append(start)
    new_df['From']=from_lst
    new_df['To']=to_lst
    new_df['Time']=time_discrete
    return new_df

def get_between_dict2(df):
    et = df['event'][len(df['event'])-1]
    G = temporal_graph(et)
    G.add_vertices(df['From'].unique())
    #G.add_temporal_edges([(('A','C'),(1,1)),(('C','B'),(2,2)),(('A','D'),(3,3)),(('C','D'),(3,3)),(('C','A'),(4,4)),(('B','D'),(4,4))])
    #G.add_temporal_edges([(('A','C'),(1,1)),(('A','D'),(2,2)),(('B','D'),(3,3)),(('C','B'),(3,3))])
    #G.add_temporal_edges([(('A','C'),(1,1)),(('A','D'),(2,2)),(('B','D'),(2,3)),(('C','D'),(3,3))])
    #G.add_temporal_edges([(('A','C'),(1,1)),(('C','D'),(3,3))])
    #G.add_temporal_edges([(('A','B'),(1,3)),(('A','C'),(1,3)),(('A','D'),(1,3)),(('B','A'),(1,1)),(('B','C'),(1,1)),(('B','D'),(1,1)),(('C','A'),(1,1)),(('C','B'),(1,1)),(('C','D'),(1,1)),(('D','A'),(1,1)),(('D','B'),(1,1)),(('D','C'),(1,1))])
    complete_lst=[]
    for i in range(len(df['From'])):
        complete_lst.append(((df['From'][i],df['To'][i]),(df['event'][i],df['event'][i])))
    G.add_temporal_edges(complete_lst)
       
    #temp_closeness = compute_temporal_closeness(G,0,et)
    bet_dict= compute_temporal_betweenness(G,0,et)
    return bet_dict



def get_closeness_dict2(df):
    et = df['event'][len(df['event'])-1]
    G = temporal_graph(et)
    G.add_vertices(df['From'].unique())
    #G.add_temporal_edges([(('A','C'),(1,1)),(('C','B'),(2,2)),(('A','D'),(3,3)),(('C','D'),(3,3)),(('C','A'),(4,4)),(('B','D'),(4,4))])
    #G.add_temporal_edges([(('A','C'),(1,1)),(('A','D'),(2,2)),(('B','D'),(3,3)),(('C','B'),(3,3))])
    #G.add_temporal_edges([(('A','C'),(1,1)),(('A','D'),(2,2)),(('B','D'),(2,3)),(('C','D'),(3,3))])
    #G.add_temporal_edges([(('A','C'),(1,1)),(('C','D'),(3,3))])
    #G.add_temporal_edges([(('A','B'),(1,3)),(('A','C'),(1,3)),(('A','D'),(1,3)),(('B','A'),(1,1)),(('B','C'),(1,1)),(('B','D'),(1,1)),(('C','A'),(1,1)),(('C','B'),(1,1)),(('C','D'),(1,1)),(('D','A'),(1,1)),(('D','B'),(1,1)),(('D','C'),(1,1))])
    complete_lst=[]
    for i in range(len(df['From'])):
        complete_lst.append(((df['From'][i],df['To'][i]),(df['event'][i],df['event'][i])))
    G.add_temporal_edges(complete_lst)
       
    temp_closeness = compute_temporal_closeness(G,0,et)
    #bet_dict= compute_temporal_betweenness(G,0,et)
    return temp_closeness

def EuclideanDistances(A, B):
    BT = B.transpose()
    vecProd = A * BT
    SqA = A.getA()**2
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
    SqB = B.getA()**2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2*vecProd
    SqED[SqED < 0] = 0
    ED = (SqED.getA())**0.5
    return np.matrix(ED)


def normalize(x):
    max = np.max(x)
    min = np.min(x)
    a = 2*((x-min)/(max-min))-1
    return a

def solved(graph, matrixM, matrixL, pos, c):
    M = matrixM
    L = matrixL
    n = graph.number_of_nodes()
    B = np.zeros((n, 2))
    
    t = time.time()
    ED = EuclideanDistances(np.matrix(pos), np.matrix(pos))  # 欧氏距离
    ED = ED + np.eye(n)
    chaX = np.tile(pos[:, 0], [n, 1]) - \
        np.tile(np.matrix(pos[:, 0]).transpose(), [1, n])  # 坐标差
    Btemp = chaX/ED
    B[:, 0] = np.nansum(Btemp, axis=0)
    chaY = np.tile(pos[:, 1], [n, 1]) - \
        np.tile(np.matrix(pos[:, 1]).transpose(), [1, n])
    Btemp = chaY/ED
    B[:, 1] = np.nansum(Btemp, axis=0)
    # print(time.time() - t)

    t = time.time()
    mAndL = M + c*n*L
    pos[:, 0], _ = cg(mAndL, B[:, 0], pos[:, 0])
    pos[:, 1], _ = cg(mAndL, B[:, 1], pos[:, 0])

    pos[:, 0] = normalize(pos[:, 0])
    pos[:, 1] = normalize(pos[:, 1])
    # print(time.time() - t)
    return pos

def matrixInit(graph):
    # n = np.max(np.array(graph.nodes()))+1
    n = graph.number_of_nodes()
    node_dict = {}
    i = 0
    for v in graph.nodes():
        node_dict.update({v: i})
        i = i + 1

    matrixM = -1 * np.ones((n, n), dtype=float) + n * np.eye(n)
    # matrixM = n * np.eye(n)
    matrixL = np.zeros((n, n), dtype=float)
    for edge in graph.edges:
        source = node_dict[edge[0]]
        target = node_dict[edge[1]]
        matrixL[source][target] += -1
        matrixL[target][target] += 1
        matrixL[target][source] += -1
        matrixL[source][source] += 1
    # print(matrixL)
    # print(matrixM)
    pos = np.random.uniform(-1, 1, size=(n, 2))
    return matrixM, matrixL, pos, node_dict

def solve_bsm(graph, iter, initPos='none', firstIterRate=0.95):
    matrixM, matrixL, pos, node_dict = matrixInit(graph)
    if initPos != 'none':
        for v in graph.nodes():
            i = node_dict[v]
            pos[i][0] = initPos[v][0]
            pos[i][1] = initPos[v][1]

    iter1 = math.ceil(firstIterRate*iter)
    for i in range(0, iter1):

        pos = solved(graph, matrixM, matrixL, pos, 1000)
    for i in range(0, iter - iter1):

        pos = solved(graph, matrixM, matrixL, pos, 1)
    position = {}
    for v in graph.nodes():
        i = node_dict[v]
        x = pos[i][0]
        y = pos[i][1]
        position.update({v: np.array([x, y])})
    return position
            
if __name__ == '__main__':

    import networkx as nx
    from netgraph import Graph
    global edge_filterdf, node_index, nodelist

    if len(argv) != 2:
        print('usage: python3 reordernodes_structural5-1-1.py [file name]')
        exit()

    df = pd.read_csv(argv[1],delimiter = ';')
    df=df.dropna() #drop rows with NaN values
    df=df.reset_index(drop=True)
    
    df=expand_df(df)
    fromlist = df['From'].tolist()

    tolist = df['To'].tolist()
    nodelist = list(set(tolist).union(set(fromlist)))
    df=expand_df2(df)
    node2index = {a:i for i,a in enumerate(nodelist)} #create a dictionary where characters are given associated numerical values
    print(node2index) #in order of outputted activity levels
    dura_time=[]
    for i in range(len(df['Duration'])):
        dura_time.append(timed(df['Duration'][i]))
    
    df['timed_duration']=dura_time
    
    event=1
    event_lst=[]
    for i in range(len(df['From'])):
        df.at[i, 'In'] = timed(df['In'][i]) #for teneto measures computation
        df.at[i, 'From'] = node2index[df['From'][i]] #change to indexes for output array
        df.at[i, 'To'] = node2index[df['To'][i]]
        event_lst.append(event)
        if i!=(len(df['From'])-1):
            if df['In'][i]<timed(df['In'][i+1]):
                event+=1
                
    df['event']=event_lst
    
    netin = {'i': df['From'], 'j': df['To'], 't': df['In']}
    tnet_df=pd.DataFrame(data=netin)
    tnet= TemporalNetwork(from_df=tnet_df) #get temporal network measures computations
    print("Temporal Degree Centrality") #The sum of all connections each node has through time (either per timepoint or over the entire temporal sequence).
    print(networkmeasures.temporal_degree_centrality(tnet))
    tnet_dict=dict(enumerate(networkmeasures.temporal_degree_centrality(tnet))) #indexes to represent nodes as keys and temporal deg val as val
    print(tnet_dict)
    #alrdy converted to indexes
    weighted_agg=df.groupby(["From", "To"], as_index=False)["timed_duration"].sum() #sum total duration of interactions in seconds
    
    G = nx.MultiDiGraph()
    nodelist2=[i for i in range(len(nodelist))]
    G.add_nodes_from(nodelist2) #or can get values from node2index
    for i in range(len(weighted_agg['From'])):
        G.add_edge(weighted_agg['From'][i], weighted_agg['To'][i], weight=weighted_agg['timed_duration'][i]) #weight of timed duration for widths
    tnet_d2=tnet_dict.copy()
    for key,val in tnet_dict.items():
        tnet_dict[key]+=50
    
    edge_wts=dict([((u,v,),d['weight'])
             for u,v,d in G.edges(data=True)])
    print("Edge weights as values and edges as keys (dict)")
    print(edge_wts)
    
    fig, ax = plt.subplots()
    
    try:
        
        print("Temporal Betweenness centrality dict:")
        betw_dict=get_between_dict2(df) #vals for node colour gradient
        print(betw_dict)
        print("Temporal Closeness centrality dict:")
        close_dict=get_closeness_dict2(df) #proportionate to node mass
        print(close_dict)
        n_color= np.asarray([val for key, val in close_dict.items()])
    
        pos = solve_bsm(G, 100)
    
        res=nx.draw_networkx(G,pos=pos,edgelist=list(edge_wts.keys()),nodelist= list(close_dict.keys()), node_color= n_color, cmap='viridis',node_size= [tnet_dict[v] for v in list(close_dict.keys())], width=np.array([(val/sum(list(edge_wts.values())))*30 for key, val in edge_wts.items()]),with_labels= False,connectionstyle='arc3, rad = 0.1',vmin=min(n_color),vmax=max(n_color)) #create multiple edges 
    
        
        for node, (x, y) in pos.items():
            val=(tnet_dict[node]/sum(list(tnet_dict.values())))*40+12
            text(x, y+0.1, list(node2index.keys())[list(node2index.values()).index(node)], fontsize=val*2,ha='center', va='center')
            
        
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin = min(n_color), vmax=max(n_color)))
        cbar=fig.colorbar(sm)
        cbar.ax.tick_params(labelsize=15)
        
        
        cbar.set_label('temporal closeness centrality',fontsize=20)
    
    except:
        n_color= np.asarray([val for key, val in tnet_d2.items()])
    
        pos = solve_bsm(G, 100)
    
        res=nx.draw_networkx(G,pos=pos,edgelist=list(edge_wts.keys()),nodelist= list(tnet_dict.keys()), node_color= n_color, cmap='viridis',node_size= [v for k,v in tnet_dict.items()], width=np.array([(val/sum(list(edge_wts.values())))*30 for key, val in edge_wts.items()]),with_labels= False,connectionstyle='arc3, rad = 0.1', vmin=min(n_color),vmax=max(n_color)) #create multiple edges 

        
        for node, (x, y) in pos.items():
            val=(tnet_dict[node]/sum(list(tnet_dict.values())))*40+12
            text(x, y+0.1, list(node2index.keys())[list(node2index.values()).index(node)], fontsize=val*2,ha='center', va='center')
            
        
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin = min(n_color), vmax=max(n_color)))
        cbar=fig.colorbar(sm)
        cbar.ax.tick_params(labelsize=15)
        
        cbar.set_label('temporal degree centrality',fontsize=20)

    plt.axis("off")

    fig.set_size_inches(13, 10)
    plt.savefig(argv[1][:-3] + '_structurallo5_1.pdf', format = 'pdf', bbox_inches = 'tight')
    
    plt.show()
    


#weighted_agg to get edge widths combined interval, tnet for computation of degree centrality values for node size
