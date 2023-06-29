#post-processing edge selection implementation on structural layout of hospital dataset using modified circular layout
from sys import argv
import matplotlib
#matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D 
from itertools import permutations, combinations
from copy import copy
from math import isnan
import numpy as np
import networkx as nx 
import seaborn as sns
from scipy.spatial import distance
from statistics import mean
import math
from teneto import TemporalNetwork, networkmeasures
from matplotlib.pyplot import figure, text
from temporal_graph import *
from scipy.stats import binom
from scipy.optimize import fsolve



#post-processing to remove unessential edges

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
        elif string!=snd:
            new_df=pd.concat([new_df,df.iloc[[index]]], ignore_index=True)
        
    return new_df

def time (s):

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


def area (ii,one2one):

    a = 0.0
    for i in range(one2one.shape[0]): #number of rows
        a += abs((ii[int(one2one[i,2])] - ii[int(one2one[i,3])]) * (one2one[i,1] - one2one[i,0]))

    return a #iteration through entire area

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #

def get_inx (one2one):
    global tend, n
    o2o = np.array(one2one)

    n = int(max(o2o[:,2:].flatten())) + 1

    ix = np.arange(n,dtype=int)
    amin = 1e100
    iter=1
    for jj in permutations(ix): #finding all possible permutations positions 0 to final node index
        a = area(jj,o2o) #reducing the rect area

        if a < amin:
            iter+=1
            amin = a
            print('new smallest', amin)
            ii = copy(jj)
        if iter>10:
            break
    return ii

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #

def draw (ax,t0,t1, me, you, fcol): #for non onetoall case

    height = me - you
    width = t1 - t0

    ax.add_artist(Rectangle([t0, you], width, height, facecolor = fcol, alpha = 0.3))
    line = Line2D([t0,t0+width], [me,me],color='black')
    ax.add_line(line)
    #ax.plot(t0+width/2, you+height, marker=matplotlib.markers.CARETDOWNBASE, linestyle='-', color='r', markersize=1, alpha=0.5) #marker for sender
    #ax.plot(t0+width/2, you, marker='o', linestyle='-', color='k', markersize=1, alpha=0.5) #marker for receiver

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #

def draw_onetoall (ax,t0,t1, me, you, snd, fcol):

    height = me - you
    width = t1 - t0

    ax.add_artist(Rectangle([t0, you], width, height, facecolor = fcol, alpha = 0.3))
    line = Line2D([t0,t0+width], [snd,snd],color='black')
    ax.add_line(line)
    #ax.plot(t0+width/2, snd, marker=matplotlib.markers.CARETDOWNBASE, linestyle='-', color='r',markersize=1, alpha=0.5) #marker for sender
    # for i in range(n):
    #     if i!=snd:
    #         ax.plot(t0+width/2, i, marker='o', linestyle='-', color='k', markersize=1, alpha=0.5) #marker for receiver
#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #



# main routine including the drawing

if __name__ == "__main__":
    global tend, n, fromlist, node2index2, edge_filterdf, node_index, nodelist

    if len(argv) != 2:
        print('usage: python3 postproc_filt_modc_hosp.py [file name]')
        exit()

    df = pd.read_csv(argv[1],delimiter = ';')
    df = df[['Job', 'Receiver_Job', 'Time']].copy()
    
    df=df.dropna() #drop rows with NaN values
    
    df = df[df.Job != 'Su']
    df=df.loc[df['Receiver_Job'].isin(list(df['Job'].unique()))]
    df=df.reset_index(drop=True)
    print(df[df.Receiver_Job=='All'])
    print(df['Job'].unique())
    
    nodelist=list(df['Job'].unique()) #accounts for 'ALL' and individual cases
    print(nodelist)
    timelist = df['Time'].tolist()

    tend = max([time(a) for a in timelist]) #finding stop time
    nodelist.remove('All') 

    node2index = {a:i for i,a in enumerate(nodelist)} #create a dictionary where characters are given associated numerical values

    node2index['All'] = -1 #create new key for 'all' and assign value -1
    nodelist.append('All') #add back to nodelist 
    print(node2index)
    #have to account for onetoall, alltoall, onetoone cases
    
    one2one = []
    one2all = []
    all2all= []
    for i in range(len(df['Job'])-1):
        try:
            t0 = time(df['Time'][i]) #time stamps are readable for both start and end times
            t1 = time(df['Time'][i+1])
            me = df['Job'][i] #find sender and receiver
            you = df['Receiver_Job'][i]
            if me in nodelist and you in nodelist:
                me = node2index[me] #identify corresponding value of sender and receiver 
                you = node2index[you]
                if me != you:
                    if me > -1 and you > -1: #finding not all case
                        one2one.append([t0,t1,max(me,you),min(me,you)]) #fraction of start and end time corr to max time duration
                    elif me>-1 and you==-1:
                        one2all.append([t0,t1, me])
                    elif me==-1 and you==-1:
                        all2all.append([t0,t1])
        except:
            continue
                    
                

    ii = get_inx(one2one) #updated indexes
    print(ii)
    
    dura_time=[]
    for i in range(len(df['Job'])):
        if i!=(len(df['Job'])-1):
            dura_time.append(time(df['Time'][i+1])-time(df['Time'][i]))
        else:
            dura_time.append(0)
    df['timed_duration']=dura_time #get duration at every point before expanding case
    
    df2=expand_df2(df) #previously taken from visualisation, didnt fully expand all cases
    # dura_time=[]
    # for i in range(len(df2['Job'])):
    #     if i!=len(df2['Job'])-1:
    #         dura_time.append(time(df2['Time'][i+1])-time(df2['Time'][i]))
    #     else:
    #         dura_time.append(0)
    # df2['timed_duration']=dura_time
    
    edge_filterdf=df2.groupby(["Job", "Receiver_Job"]).size().reset_index(name="Count") # get filtered edges 
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
    for i in range(len(df2['Job'])):
        df2.at[i, 'Time'] = time(df2['Time'][i]) #for teneto measures computation
        df2.at[i, 'Job'] = node2index[df2['Job'][i]] #change to indexes for output array
        df2.at[i, 'Receiver_Job'] = node2index[df2['Receiver_Job'][i]]
        event_lst.append(event)
        if i!=(len(df2['Job'])-1):
            if (df2['Time'][i])<time(df2['Time'][i+1]):
                event+=1
                
    df2['event']=event_lst
    
    
    netin = {'i': df2['Job'], 'j': df2['Receiver_Job'], 't': df2['Time']}

    tnet_df=pd.DataFrame(data=netin)
    tnet= TemporalNetwork(from_df=tnet_df) #get temporal network measures computations
    print("Temporal Degree Centrality") #The sum of all connections each node has through time (either per timepoint or over the entire temporal sequence).
    print(networkmeasures.temporal_degree_centrality(tnet))
    tnet_dict=dict(enumerate(networkmeasures.temporal_degree_centrality(tnet))) #indexes to represent nodes as keys and temporal deg val as val
    print(tnet_dict)
    #alrdy converted to indexes
    weighted_agg=df2.groupby(["Job", "Receiver_Job"], as_index=False)["timed_duration"].sum() #sum total duration of interactions in seconds
    
    
    r = 100 #customised to be a conservative value
    numPoints = n #depending on number of interacting points
    points = []
    x=[]
    y=[]
    for index in range(numPoints):
        x.append(r*math.cos((index*2*math.pi)/numPoints))
        y.append(r*math.sin((index*2*math.pi)/numPoints))


    coords_lst=list(zip(x,y))
    node_position={}
    for idx in range(len(list(ii))):
        #node_position[ii[idx]]=coords_lst[idx]
        node_position[idx]=coords_lst[ii[idx]]
    
    print("node positions:")
    print(node_position)
    weighted_agg=df2.groupby(["Job", "Receiver_Job"], as_index=False)["timed_duration"].sum() #sum total duration of interactions in seconds
    
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
    ax.axis("equal")
    
    try:
        
        print("Temporal Closeness centrality dict:")
        close_dict=get_closeness_dict2(df2) #proportionate to node mass
        print(close_dict)
        n_color= np.asarray([val for key, val in close_dict.items()])
        #ax=ax for this one also since set axis to equal so use same ax
        res=nx.draw_networkx(G,ax=ax,pos=node_position,edgelist=list(mod_edge.keys()),nodelist= list(close_dict.keys()), node_color= n_color, cmap='viridis',node_size= [tnet_dict[v] for v in list(close_dict.keys())], width=np.array([(val/sum(list(mod_edge.values())))*30 for key, val in mod_edge.items()]),with_labels= False,connectionstyle='arc3, rad = 0.1', vmin=min(n_color), vmax=max(n_color)) #create multiple edges 
    
    
        
        for node, (x, y) in node_position.items():
            val=(tnet_dict[node]/sum(list(tnet_dict.values())))*40+12
            text(x, y+12, node, fontsize=val*2,ha='center', va='center')
            
        
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin = min(n_color), vmax=max(n_color)))
        cbar=fig.colorbar(sm)
        cbar.ax.tick_params(labelsize=15)
        
        
        cbar.set_label('temporal closeness centrality',fontsize=20)
    
            
    except:
        n_color= np.asarray([val for key, val in tnet_d2.items()])
        #ax=ax for this one also since set axis to equal so use same ax
        res=nx.draw_networkx(G,ax=ax,pos=node_position,edgelist=list(mod_edge.keys()),nodelist= list(tnet_dict.keys()), node_color= n_color, cmap='viridis',node_size= [v for k,v in tnet_dict.items()], width=np.array([(val/sum(list(mod_edge.values())))*30 for key, val in mod_edge.items()]),with_labels= False,connectionstyle='arc3, rad = 0.1',vmin=min(n_color),vmax=max(n_color)) #create multiple edges 
    
       
        
        for node, (x, y) in node_position.items():
            val=(tnet_dict[node]/sum(list(tnet_dict.values())))*40+12
            text(x, y+12, node, fontsize=val*2,ha='center', va='center')
            
        
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin = min(n_color), vmax=max(n_color)))
        cbar=fig.colorbar(sm)
        cbar.ax.tick_params(labelsize=15)
        
        
        cbar.set_label('temporal degree centrality',fontsize=20)
        
            
    plt.axis("off")
    fig.set_size_inches(13, 10)
    
    plt.savefig(argv[1][:-3] + '_structuralfilt4_1.pdf', format = 'pdf', bbox_inches = 'tight')
    
    plt.show()
    

#circular layout; edge widths according to interval of comm, colour based on closeness centrality, node size temporal degree
