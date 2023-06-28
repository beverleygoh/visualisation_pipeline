import networkx as nx
import matplotlib.pyplot as plt
import warnings
from sys import argv
import pandas as pd
import math
from scipy.sparse.linalg import cg
import time
from scipy.stats import binom
from scipy.optimize import fsolve
#https://github.com/tomzhch/IES-Backbone/blob/master/bsm.py

import matplotlib
from teneto import TemporalNetwork, networkmeasures
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
from matplotlib.pyplot import figure, text
from temporal_graph import *


def expand_df2(df):
    nodelist.remove('All')
    new_df=pd.DataFrame()
    for index, rows in df.iterrows():
        string=rows.Receiver_Job
        snd= rows.Job
        if string=="All" and snd=="All":
            string_lst=nodelist
            for i in string_lst:
                for j in string_lst:
                    if j!=i:
                        new_df=pd.concat([new_df,df.iloc[[index]].assign(Receiver_Job=j, Job=i)], ignore_index=True)
        elif string=="All" and snd!="All":
            string_lst=nodelist
            for ele in string_lst:
                if ele!=rows.Job:
                    new_df=pd.concat([new_df,df.iloc[[index]].assign(Receiver_Job=ele)], ignore_index=True)
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

def get_between_dict2(df):
    et = df['event'][len(df['event'])-1]
    G = temporal_graph(et)
    G.add_vertices(df['Job'].unique())
    #G.add_temporal_edges([(('A','C'),(1,1)),(('C','B'),(2,2)),(('A','D'),(3,3)),(('C','D'),(3,3)),(('C','A'),(4,4)),(('B','D'),(4,4))])
    #G.add_temporal_edges([(('A','C'),(1,1)),(('A','D'),(2,2)),(('B','D'),(3,3)),(('C','B'),(3,3))])
    #G.add_temporal_edges([(('A','C'),(1,1)),(('A','D'),(2,2)),(('B','D'),(2,3)),(('C','D'),(3,3))])
    #G.add_temporal_edges([(('A','C'),(1,1)),(('C','D'),(3,3))])
    #G.add_temporal_edges([(('A','B'),(1,3)),(('A','C'),(1,3)),(('A','D'),(1,3)),(('B','A'),(1,1)),(('B','C'),(1,1)),(('B','D'),(1,1)),(('C','A'),(1,1)),(('C','B'),(1,1)),(('C','D'),(1,1)),(('D','A'),(1,1)),(('D','B'),(1,1)),(('D','C'),(1,1))])
    complete_lst=[]
    for i in range(len(df['Job'])):
        complete_lst.append(((df['Job'][i],df['Receiver_Job'][i]),(df['event'][i],df['event'][i])))
    G.add_temporal_edges(complete_lst)
       
    #temp_closeness = compute_temporal_closeness(G,0,et)
    bet_dict= compute_temporal_betweenness(G,0,et)
    return bet_dict


def get_closeness_dict2(df):
    et = df['event'][len(df['event'])-1]
    G = temporal_graph(et)
    G.add_vertices(df['Job'].unique())
    #G.add_temporal_edges([(('A','C'),(1,1)),(('C','B'),(2,2)),(('A','D'),(3,3)),(('C','D'),(3,3)),(('C','A'),(4,4)),(('B','D'),(4,4))])
    #G.add_temporal_edges([(('A','C'),(1,1)),(('A','D'),(2,2)),(('B','D'),(3,3)),(('C','B'),(3,3))])
    #G.add_temporal_edges([(('A','C'),(1,1)),(('A','D'),(2,2)),(('B','D'),(2,3)),(('C','D'),(3,3))])
    #G.add_temporal_edges([(('A','C'),(1,1)),(('C','D'),(3,3))])
    #G.add_temporal_edges([(('A','B'),(1,3)),(('A','C'),(1,3)),(('A','D'),(1,3)),(('B','A'),(1,1)),(('B','C'),(1,1)),(('B','D'),(1,1)),(('C','A'),(1,1)),(('C','B'),(1,1)),(('C','D'),(1,1)),(('D','A'),(1,1)),(('D','B'),(1,1)),(('D','C'),(1,1))])
    complete_lst=[]
    for i in range(len(df['Job'])):
        complete_lst.append(((df['Job'][i],df['Receiver_Job'][i]),(df['event'][i],df['event'][i])))
    G.add_temporal_edges(complete_lst)
       
    temp_closeness = compute_temporal_closeness(G,0,et)
    #bet_dict= compute_temporal_betweenness(G,0,et)
    return temp_closeness

def f(x):
    tau=edge_filterdf['Count'].sum()
    #node_index=list(edge_filterdf['From'].unique())
    n= len(node_index)
    F = np.empty((n))
    for i in range(n):
        indx=list(np.where(edge_filterdf['Job']==node_index[i])[0]) #indexes where i equals certain value
        fn=0
        for ele in indx:
            fn+=(edge_filterdf['Count'][ele]-tau*x[node_index[i]]*x[edge_filterdf['Receiver_Job'][ele]])/(1-x[node_index[i]]*x[edge_filterdf['Receiver_Job'][ele]])
        F[i]=fn 
    return F    
    
def significance_test(res,alpha):
    #2d array of reject null hypo or not
    arr=np.zeros(shape=(len(node_index),len(node_index)))
    tau=edge_filterdf['Count'].sum()
    for i in range(len(node_index)):
        for j in range(len(node_index)):
            if i!=j:
                try:
                    edgedf_pos=list(np.where((edge_filterdf["Job"]==node_index[i])&(edge_filterdf["Receiver_Job"]==node_index[j]))[0])[0]
                    print(edgedf_pos)
                    print(edge_filterdf['Count'][edgedf_pos])

                    prob= binom.cdf(edge_filterdf['Count'][int(edgedf_pos)], int(tau), res[node_index[i]]*res[node_index[j]]) #edge_filterdf['Count'][edgedf_pos] empirical val
                    if (1-prob)<alpha:
                        arr[node_index[i]][node_index[j]]=1.0
                    else:
                        arr[node_index[i]][node_index[j]]=0.0
                except IndexError:
                    continue
    return arr

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
        # print(i)
        pos = solved(graph, matrixM, matrixL, pos, 1000)
    for i in range(0, iter - iter1):
    # for i in range(0, 1):
            # print(i)
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
        print('usage: python3 reordernodes_structural5.py [file name]')
        exit()

    df = pd.read_csv(argv[1],delimiter = ';')
    df = df[['Job', 'Receiver_Job', 'Time']].copy()
    
    df=df.dropna() #drop rows with NaN values
    
    df = df[df.Job != 'Su']
    df=df.loc[df['Receiver_Job'].isin(list(df['Job'].unique()))]
    df=df.reset_index(drop=True)

    nodelist=list(df['Job'].unique()) #accounts for 'ALL' and individual cases
    dura_time=[]
    for i in range(len(df['Job'])):
        if i!=len(df['Job'])-1:
            dura_time.append(timed(df['Time'][i+1])-timed(df['Time'][i]))
        else:
            dura_time.append(0)
    df['timed_duration']=dura_time
    
    df=expand_df2(df)
    
    print(nodelist)

    node2index = {a:i for i,a in enumerate(nodelist)} #create a dictionary where characters are given associated numerical values

    # dura_time=[]
    # for i in range(len(df['Job'])):
    #     if i!=len(df['Job'])-1:
    #         dura_time.append(timed(df['Time'][i+1])-timed(df['Time'][i]))
    #     else:
    #         dura_time.append(0)
    # df['timed_duration']=dura_time
    
    edge_filterdf=df.groupby(["Job", "Receiver_Job"]).size().reset_index(name="Count") # get filtered edges 
    for i in range(len(edge_filterdf['Job'])):
        edge_filterdf.at[i, 'Job'] = node2index[edge_filterdf['Job'][i]] #change to indexes for output array
        edge_filterdf.at[i, 'Receiver_Job'] = node2index[edge_filterdf['Receiver_Job'][i]]
        
    node_index= list(edge_filterdf['Receiver_Job'].unique()) #indexes to represent the different nodes
    val = 1/len(node_index)#guessed activity levels, assumed equal
    print("Equal act values (start val):"+ str(val))
    Guess=np.zeros(shape=(len(node_index),0))
    for i in range(len(node_index)):
        Guess=np.append(Guess,val)
    res = fsolve(f,Guess)
    print("Significance matrix")
    print(res)
    arr=significance_test(res,alpha=0.01)
    print(arr)
    
    event=1
    event_lst=[]
    for i in range(len(df['Job'])):
        df.at[i, 'Time'] = timed(df['Time'][i]) #for teneto measures computation
        #df.at[i+1, 'In'] = timed(df['In'][i+1])
        df.at[i, 'Job'] = node2index[df['Job'][i]] #change to indexes for output array
        df.at[i, 'Receiver_Job'] = node2index[df['Receiver_Job'][i]]
        event_lst.append(event)
        if i!=(len(df['Job'])-1):
            if df['Time'][i]<timed(df['Time'][i+1]):
                event+=1
                
    df['event']=event_lst
    
    netin = {'i': df['Job'], 'j': df['Receiver_Job'], 't': df['Time']}
    tnet_df=pd.DataFrame(data=netin)
    tnet= TemporalNetwork(from_df=tnet_df) #get temporal network measures computations
    print("Temporal Degree Centrality") #The sum of all connections each node has through time (either per timepoint or over the entire temporal sequence).
    print(networkmeasures.temporal_degree_centrality(tnet))
    tnet_dict=dict(enumerate(networkmeasures.temporal_degree_centrality(tnet))) #indexes to represent nodes as keys and temporal deg val as val
    print(tnet_dict)
    #alrdy converted to indexes
    weighted_agg=df.groupby(["Job", "Receiver_Job"], as_index=False)["timed_duration"].sum() #sum total duration of interactions in seconds
    
    G = nx.MultiDiGraph()
    nodelist2=[i for i in range(len(nodelist))]
    G.add_nodes_from(nodelist2) #or can get values from node2index
    for i in range(len(weighted_agg['Job'])):
        G.add_edge(weighted_agg['Job'][i], weighted_agg['Receiver_Job'][i], weight=weighted_agg['timed_duration'][i]) #weight of timed duration for widths
    tnet_d2=tnet_dict.copy()
    for key,val in tnet_dict.items():
        tnet_dict[key]+=50 #set min as default node size
    
    edge_wts=dict([((u,v,),d['weight'])
             for u,v,d in G.edges(data=True)])
    print("Edge weights as values and edges as keys (dict)")
    print(edge_wts)
    
    mod_edge={}
    for k,v in (edge_wts.items()):
        if arr[k[0]][k[1]]==1 and v!=0: #filtered out unimportant irrelevant edges
            mod_edge[k]=v
    
    print("Modified edge wts")
    print(mod_edge)
    
    fig, ax = plt.subplots()
    
    try:
        
        print("Temporal Closeness centrality dict:")
        close_dict=get_closeness_dict2(df) #proportionate to node mass
        print(close_dict)
        n_color= np.asarray([val for key, val in close_dict.items()])
    
        pos = solve_bsm(G, 100)
    
        res=nx.draw_networkx(G,pos=pos,edgelist=list(mod_edge.keys()),nodelist= list(close_dict.keys()), node_color= n_color, cmap='viridis',node_size= [tnet_dict[v] for v in list(close_dict.keys())], width=np.array([(val/sum(list(mod_edge.values())))*30 for key, val in mod_edge.items()]),with_labels= False,connectionstyle='arc3, rad = 0.1', vmin=min(n_color), vmax=max(n_color)) #create multiple edges 
        
        
        
        for node, (x, y) in pos.items():
            val=(tnet_dict[node]/sum(list(tnet_dict.values())))*40+12 #set the maximum val as the standard default fontsize
            text(x, y+0.15, node, fontsize=val*2,ha='center', va='center')
            
        
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin = min(n_color), vmax=max(n_color)))
        cbar=fig.colorbar(sm)
        cbar.ax.tick_params(labelsize=15)
        
        cbar.set_label('temporal closeness centrality',fontsize=20)
            
    except:
        n_color= np.asarray([val for key, val in tnet_d2.items()])
    
        pos = solve_bsm(G, 100)
    
        res=nx.draw_networkx(G,pos=pos,edgelist=list(mod_edge.keys()),nodelist= list(tnet_dict.keys()), node_color= n_color, cmap='viridis',node_size= [v for k,v in tnet_dict.items()], width=np.array([(val/sum(list(mod_edge.values())))*30 for key, val in mod_edge.items()]),with_labels= False,connectionstyle='arc3, rad = 0.1', vmin=min(n_color), vmax=max(n_color))
    
        
        for node, (x, y) in pos.items():
            val=(tnet_dict[node]/sum(list(tnet_dict.values())))*40+12 #set the maximum val as the standard default fontsize
            text(x, y+0.15, node, fontsize=val*2,ha='center', va='center')
            
        
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin = min(n_color), vmax=max(n_color)))
        cbar=fig.colorbar(sm)
        cbar.ax.tick_params(labelsize=15)
        
        cbar.set_label('temporal degree centrality',fontsize=20)
    
    #cbar.yaxis.set_label_coords(-.1, .1)
    plt.axis("off")
    fig.set_size_inches(13, 10)
    plt.savefig(argv[1][:-3] + '_structuralfilt5_1.pdf', format = 'pdf', bbox_inches = 'tight')
    
    plt.show()
    

#circular layout; edge widths according to interval of comm, colour based on closeness centrality, node size temporal degree


#swap around the spatial ordering based on the different measures

#weighted_agg to get edge widths combined interval, tnet for computation of degree centrality values for node size
