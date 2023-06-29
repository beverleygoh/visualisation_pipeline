#both area minimisation and recurrent neighbours node reordering methods explored and outputted for full event layouts of hospital dataset

from sys import argv
import matplotlib
matplotlib.use('Agg')
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
from scipy.stats import binom
from scipy.optimize import fsolve
from matplotlib.pyplot import text
import random
from collections import Counter
from teneto import TemporalNetwork, networkmeasures

#no pre or post processing because does not take into account any temporal measures for display; displays temporal network as it is to reflect visualisation patterns

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




#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #

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

def draw_alltoall (ax,t0,t1, me, you, fcol):

    height = me - you
    width = t1 - t0

    ax.add_artist(Rectangle([t0, you], width, height, facecolor = fcol, alpha = 0.3))

def time (s):

    try:
        a = str(s).strip().split('.')
        b = int(a[0]) * 60.0*60.0 + int(a[1])*60.0 + int(a[2])
    except:
        return -1

    return b

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #

# main routine including the drawing
def gen_plot(G,pos,my_yticks,colour_lst,from_event, to_event, val_lst): 
    fig, ax= plt.subplots(figsize=(13,10))#res=nx.draw_networkx(G,ax=ax,pos=pos,nodelist= nodelist2, cmap='viridis', node_color=colour_lst, node_size= [val*50 for (node, val) in G.degree()],with_labels= False)
    #giant = max((G.subgraph(c) for c in nx.connected_components(G)), key=len)
    #print(giant)
    largest_cc = max(nx.weakly_connected_components(G), key=len)
    print(largest_cc)
    S = G.subgraph(largest_cc)
    edge_color_list = ["black"]*len(G.edges)
#replace the color in edge_color_list with red if the edge belongs to the shortest path:
    for i, edge in enumerate(G.edges()):
        if edge in list(nx.edges(S)) or (edge[1],edge[0]) in list(nx.edges(S)):
            edge_color_list[i] = 'red'
    node_colour_list=["grey"]*len(nodelist2)
    alpha_list=[0.1]*len(nodelist2)
    for i, node in enumerate(nodelist2):
        if node in from_event or node in to_event:
            if colour_lst[node]>2:
                node_colour_list[i]='red'
                alpha_list[i]=0.7
            else:
                node_colour_list[i]= 'pink'
                alpha_list[i]=0.7
    node_size=[300]*len(nodelist2)
    for i in range(len(alpha_list)):
        if alpha_list[i]!=0.7:
            node_size[i]=30
        if alpha_list[i]==0.7:
            node_size[i]=70
    res=nx.draw_networkx(G,ax=ax,pos=pos,nodelist= nodelist2, edge_color= edge_color_list, node_size=node_size,node_color=node_colour_list,with_labels= False)
    #res=nx.draw_networkx(G,ax=ax,pos=pos,nodelist=nodelist2, node_color=node_colour_list, node_size=node_size,edge_color= edge_color_list, with_labels= False)
    nx.draw_networkx_nodes(G, pos=pos, nodelist=nodelist2, node_size=node_size,node_color=node_colour_list,alpha=alpha_list)
    #title=title+' thresh= '+ str(thresh)
    #ax.set_title(title, fontsize=20)
    ax.set_yticks(np.arange(n,dtype=int), my_yticks, fontsize=20)
    ax.set_xticks(list(np.arange(0, max(val_lst),500*5)), list(np.arange(0, len(colour_lst)-1,500)), fontsize=10)
    
    ax.tick_params(left=True,bottom = True, labelleft=True, labelbottom=True, labelsize=15)
    ax.grid(axis='y', linestyle= '-')
    ax.set_xlabel("Event", loc='right', fontsize=18)
    ax.set_ylabel("Sender", loc='top', fontsize=18)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
#     textstr = '\n'.join((
#     r'Connected nodes (coloured)',
#     r'Red=All parties interacting, Pink=2 parties interacting',
#     r'Unconnected nodes (grey)'))
#     props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# # place a text box in upper left in axes coords
#     ax.text(0.7, 0.99, textstr, transform=ax.transAxes, fontsize=15,
#         verticalalignment='top', bbox=props)
    
    
    return 

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
                        arr[i][j]=1.0
                    else:
                        arr[i][j]=0.0
                except IndexError:
                    continue
    return arr


def recurr_neighbourpos(tnet_dict,df):
    new_ytickslist=list(np.zeros(n))
    for i in range(len(new_ytickslist)):
        new_ytickslist[i]=100
    #centre max degree centrality node
    val=max(list(tnet_dict.values()))
    max_node=list(tnet_dict.keys())[list(tnet_dict.values()).index(val)]
    print(max_node)
    #consider 0 degree centrality cases
    new_ytickslist[int(n/2)]=max_node
    d=df.groupby(df[['Job', 'Receiver_Job']].agg(frozenset, 1))['Time'].count().reset_index()
    d.columns=['From-To', 'Count']
    print(d)
    ard_max=list(np.zeros(2)) #2 item list after first max node condition
    count_max=list(np.zeros(2))
    d2=d.copy()
    for i in range(len(d['From-To'])):
        try:
            if max_node in d['From-To'][i]:
                pos= list(d['From-To'][i]).index(max_node) #pos in 2 value tuple
                if d['Count'][i]>=count_max[0]:
                    ard_max[1]=ard_max[0] #get val of node corr to original max 
                    ard_max[0]=list(d['From-To'][i])[pos==False] #get val of node corr to new max
                    count_max[1]=count_max[0]
                    count_max[0]=d['Count'][i]
                     #new max , original max moves to second place, new max takes over
                if d['Count'][i]<count_max[0] and d['Count'][i]>count_max[1]:
                    ard_max[1]=list(d['From-To'][i])[pos==False] 
                    count_max[1]=d['Count'][i]
        except:
            continue
    print(count_max)
    print(ard_max)
    d2=d2.drop([list(d2['Count']).index(count_max[0]),list(d2['Count']).index(count_max[1])])  
    d2=d2.reset_index(drop=True)  
    print(d2)          
    new_ytickslist[int(n/2)-1]=ard_max[0]#largest recurr neighbour of max node 
    pos_up=int(n/2)-1
    max_up=0
    idx_up=0
    new_ytickslist[int(n/2)+1]=ard_max[1]
    pos_down=int(n/2)+1
    max_down=0
    idx_down=0
    #remain_nodes=n-3
    flag=0
    iter1=0
    iter=0
    
    while pos_up>0:
        iter+=1
        #print(iter)
        max_up=0
        for i in range(len(d2['From-To'])):
            #max_up=0
            try:
                if ard_max[0] in d2['From-To'][i]:
                    #flag=1
                    pos= list(d2['From-To'][i]).index(ard_max[0]) #pos in 2 value tuple
                    if d2['Count'][i]>max_up and list(d2['From-To'][i])[pos==False] not in new_ytickslist:
                        flag=1
                        idx_up=list(d2['From-To'][i])[pos==False]
                        max_up=d2['Count'][i] #new max , original max moves to second place, new max takes over
                    elif iter-iter1>=5:
                        val=random.choice([x for x in range(n) if x not in new_ytickslist])
                        pos_up-=1
                        new_ytickslist[pos_up]=val
                        iter=0
                        iter1=0
            except:
                continue
        if flag==1:
            iter1+=1
            print(iter1)
            pos_up-=1
            new_ytickslist[pos_up]=idx_up
            print(new_ytickslist)
            ard_max[0]=idx_up
            try:
                d2=d2.drop(index=list(d2['Count']).index(max_up))
                d2=d2.reset_index(drop=True)
                flag=0
            except:
                flag=0
                continue
        
    
    flag=0
    iter1=0
    iter=0
    while pos_down<(len(new_ytickslist)-1):
        max_down=0
        iter+=1
        for i in range(len(d2['From-To'])):
            #max_down=0
            try:
                if ard_max[1] in d2['From-To'][i]:
                    #flag=1
                    pos2= list(d2['From-To'][i]).index(ard_max[1]) #pos in 2 value tuple
                    if d2['Count'][i]>=max_down and list(d2['From-To'][i])[pos2==False] not in new_ytickslist:
                        
                        idx_down=list(d2['From-To'][i])[pos2==False]
                        max_down=d2['Count'][i] #new max , original max moves to second place, new max takes over
                        flag=1
                    elif iter-iter1>=5:
                        val=random.choice([x for x in range(n) if x not in new_ytickslist])
                        pos_up-=1
                        new_ytickslist[pos_up]=val
                        iter=0
                        iter1=0
            except:
                continue
                
        if flag==1:                
            pos_down+=1
            iter1+=1
            new_ytickslist[pos_down]=idx_down
            ard_max[1]=idx_down
            try:
                d2=d2.drop(index=list(d2['Count']).index(max_down))
                d2=d2.reset_index(drop=True)
                flag=0
            except:
                flag=0
                break

    
    return new_ytickslist
            
def max_tempstep_filt(df,max_t):
    curr_t=0  
    from_event=[]
    to_event=[]
    temp_wt=[]
    for i in range(len(df['Job'])): #thresholding set at 5; moving window
        t=1
        t_check=0
        curr_t=time(df['Time'][i])
        while t+t_check<=max_t: #changed from t<=
            try:
                if curr_t<time(df['Time'][i+t+t_check]):
                    if df['Receiver_Job'][i]==df['Job'][i+t+t_check] and df['Job'][i]!=df['Receiver_Job'][i+t+t_check]: #acyclic
                        from_event.append(df['event'][i])
                        to_event.append(df['event'][i+t_check+t])
                        temp_wt.append(t)
                        curr_t=time(df['Time'][i+t+t_check])
                        t+=1
                    elif df['Receiver_Job'][i]==-1 and df['Job'][i]!=df['Receiver_Job'][i+t+t_check]:
                        from_event.append(df['event'][i])
                        to_event.append(df['event'][i+t_check+t])
                        temp_wt.append(t)
                        curr_t=time(df['Time'][i+t+t_check])
                        t+=1
                    else:
                        curr_t=time(df['Time'][i+t])
                        t+=1
                else:
                    t_check+=1
            except:
                break
    return from_event,to_event,temp_wt

# def max_tempstep_filt(df,max_t):
#     curr_t=0  
#     from_event=[]
#     to_event=[]
#     temp_wt=[]
#     for i in range(len(df['Job'])): #thresholding set at 5; moving window
#         t=1
#         #t_check=0
#         while t<=max_t: #changed from t<=
#             try:
#                 if df['Receiver_Job'][i]==df['Job'][i+t] and df['Job'][i]!=df['Receiver_Job'][i+t]: #acyclic
#                     from_event.append(df['event'][i])
#                     to_event.append(df['event'][i+t])
#                     temp_wt.append(t)
#                     t+=1
#                 elif df['Receiver_Job'][i]==-1 and df['Job'][i]!=df['Receiver_Job'][i+t]:
#                     from_event.append(df['event'][i])
#                     to_event.append(df['event'][i+t])
#                     temp_wt.append(t)
#                     t+=1
#                 else:
#                     t+=1

#             except:
#                 print("hello")
#                 print(df['event'][i])
#                 break
#     return from_event,to_event,temp_wt

if __name__ == "__main__":
    global tend, n, node_index, edge_filterdf

    #time axis t for xpos, ii[index val] for ypos
    if len(argv) != 2:
        print('usage: python3 event_both_hosp.py [file name]')
        exit()

    df = pd.read_csv(argv[1],delimiter = ';')
    df = df[['Job', 'Receiver_Job', 'Time']].copy()
    
    df=df.dropna() #drop rows with NaN values
    
    df = df[df.Job != 'Su']
    df=df.loc[df['Receiver_Job'].isin(list(df['Job'].unique()))]
    df=df.reset_index(drop=True)
    print(df[df.Receiver_Job=='All'])
    print(df['Job'].unique())
    #df=expand_df(df)
    
    #tolist = df['Receiver_Job'].tolist()
    #fromlist = df['Job'].tolist()
    #nodelist = list(set(tolist).union(set(fromlist)))
    nodelist=list(df['Job'].unique()) #accounts for 'ALL' and individual cases
    print(nodelist)
    timelist = df['Time'].tolist()

    tend = max([time(a) for a in timelist]) #finding stop time
    nodelist.remove('All') 

    node2index = {a:i for i,a in enumerate(nodelist)} #create a dictionary where characters are given associated numerical values

    node2index['All'] = -1 #create new key for 'all' and assign value -1
    nodelist.append('All') #add back to nodelist 
    
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
                        one2one.append([t0,t1,me,you]) #fraction of start and end time corr to max time duration
                    elif me>-1 and you==-1:
                        one2all.append([t0,t1, me])
                    elif me==-1 and you==-1:
                        all2all.append([t0,t1])
        except:
            continue
                    

    ii_1 = get_inx(one2one) #for node reordering area minimisation case

    df2=expand_df2(df)
    for i in range(len(df2['Job'])):
        df2.at[i, 'Time'] = time(df2['Time'][i]) #for teneto measures computation
        df2.at[i, 'Job'] = node2index[df2['Job'][i]]
        df2.at[i, 'Receiver_Job'] = node2index[df2['Receiver_Job'][i]] #change to indexes
    netin = {'i': df2['Job'], 'j': df2['Receiver_Job'], 't': df2['Time']}
    tnet_df=pd.DataFrame(data=netin)
    tnet= TemporalNetwork(from_df=tnet_df) #get temporal network measures computations
    print("Temporal Degree Centrality") #The sum of all connections each node has through time (either per timepoint or over the entire temporal sequence).
    print(networkmeasures.temporal_degree_centrality(tnet))
    tnet_dict=dict(enumerate(networkmeasures.temporal_degree_centrality(tnet)))
    print(tnet_dict)
    
    ii=recurr_neighbourpos(tnet_dict, df2)
    print("number of unique time")
    print(len(list(df['Time'].unique())))
    event=0
    event_lst=[]
    for i in range(len(df['Job'])):
        df.at[i, 'Job'] = node2index[df['Job'][i]] #change to indexes for output array
        df.at[i, 'Receiver_Job'] = node2index[df['Receiver_Job'][i]]
        event_lst.append(event)
        if i!=(len(df['Job'])-1):
            if time(df['Time'][i])<time(df['Time'][i+1]):
                event+=1
                
    df['event']=event_lst
     
    from_event=[]
    to_event=[]
    temp_wt=[] 
    thresh=1
    from_event, to_event, temp_wt=max_tempstep_filt(df,thresh) # up to a tolerable value of 5 for temporal distance
    
    #create pos dict #position nodes according temporally in order of increasing time, based on location of sender node, doesnt scale well
    pos={}
    pos_1={}
    df3=df.copy()
    df3 = df3.drop_duplicates(subset=["event"], keep='first') #reassign to df2
    df3 = df3.reset_index(drop=True)
    
    val_lst=[]
    ii.reverse()
    for i in range(len(df3['event'])):
        val=5*i
        #print(i+1)
        val_lst.append(val)
        if df3['Job'][i]!=-1:
            pos[i]=(val,ii.index(df3['Job'][i])) #in increasing order of time
            pos_1[i]=(val,ii_1[df3['Job'][i]]) #in increasing order of time
        else:
            pos[i]=(val,int(len(ii)/2)) #in increasing order of time
            pos_1[i]=(val,int(len(ii_1)/2))
    
    G = nx.MultiDiGraph()
    nodelist2=[i for i in range(len(df3['event']))]
    G.add_nodes_from(nodelist2) #or can get values from node2index
    for i in range(len(from_event)):
        G.add_edge(from_event[i], to_event[i], weight=temp_wt[i])
    
    
    
    colour_lst=np.zeros(max(df['event'])+1) #according to number of interacting nodes
    #print(max(df['event']))
    print("length colour")
    print(len(colour_lst))
    for i in range(len(df['event'])):
        if df['Receiver_Job'][i]!=-1:
            colour_lst[df['event'][i]]+=1
        else:
            colour_lst[df['event'][i]]+=(n-1)

    for i in range(len(colour_lst)):
        colour_lst[i]+=1 #include sender node to get total num of interacting nodes, according to event number

    print(len(nodelist2))
    print(min(to_event))
    print(max(to_event))
    
    my_yticks = list(ii)
    #my_yticks.reverse()
    #title_rn='Event plot '+str(argv[1][:-4])+ ' (RN layout)'
    gen_plot(G,pos,my_yticks,colour_lst,from_event, to_event, val_lst)
    plt.savefig(argv[1][:-3] + '_eventlayoutrn1.pdf', format = 'pdf', bbox_inches = 'tight')
    
        
    my_yticks_chr=[]
    for i in range(len(my_yticks)):
        my_yticks_chr.append(list(node2index.keys())[list(node2index.values()).index(my_yticks[i])])
    gen_plot(G,pos,my_yticks_chr,colour_lst,from_event, to_event,val_lst)
    plt.savefig(argv[1][:-3] + '_eventlayoutrn1chr.pdf', format = 'pdf', bbox_inches = 'tight')
    
    my_yticks_1 = list(np.arange(n,dtype=int))
    for i in range(len(my_yticks_1)):
        my_yticks_1[i]=ii_1.index(i)
    
    gen_plot(G,pos_1,my_yticks_1,colour_lst,from_event, to_event, val_lst)
    plt.savefig(argv[1][:-3] + '_eventlayout1.pdf', format = 'pdf', bbox_inches = 'tight')
    
        
    my_yticks_chr1=[]
    for i in range(len(my_yticks_1)):
        my_yticks_chr1.append(list(node2index.keys())[list(node2index.values()).index(my_yticks_1[i])])
    gen_plot(G,pos_1,my_yticks_chr1,colour_lst,from_event, to_event, val_lst)
    plt.savefig(argv[1][:-3] + '_eventlayout1chr.pdf', format = 'pdf', bbox_inches = 'tight')    
    
#check structural layouts pls reorder temporal layout
#write definition for event layouts
