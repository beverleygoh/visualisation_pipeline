#both area minimisation and recurrent neighbours node reordering methods trialled and outputted for full temporal layouts of hospital dataset

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
from teneto import TemporalNetwork, networkmeasures
import random
import time

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
    # line = Line2D([t0,t0+width], [me,me],color='black')
    # ax.add_line(line)
    ax.plot([t0,t0+width], [me,me], color='black')

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
def draw_onetoall (ax,t0,t1, me, you, snd, fcol):

    height = me - you
    width = t1 - t0

    ax.add_artist(Rectangle([t0, you], width, height, facecolor = fcol, alpha = 0.3))
    # line = Line2D([t0,t0+width], [snd,snd],color='black')
    # ax.add_line(line)
    ax.plot([t0,t0+width], [snd,snd], color='black')
    
def draw_alltoall (ax,t0,t1, me, you, fcol):

    height = me - you
    width = t1 - t0

    ax.add_artist(Rectangle([t0, you], width, height, facecolor = fcol, alpha = 0.3))
    

def timed (s):

    try:
        a = str(s).strip().split('.')
        b = int(a[0]) * 60.0 + int(a[1]) + int(a[2])/60.0
    except:
        return -1

    return b

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #

# main routine including the drawing
def gen_plot(node_yticks,label_yticks,one2one, one2all, all2all): 
    plt.figure()
    fig, ax= plt.subplots()
    plt.yticks(np.arange(n,dtype=int), label_yticks, fontsize=15)
    plt.xticks(np.arange(0, tend, 10), fontsize=10)
    #ax.set_xticklabels([])
    plt.xlabel('Time (min)', fontsize=18)
    plt.ylabel('Interacting parties', fontsize=18)
    plt.tick_params(bottom = False)
    plt.grid(axis='y', linestyle= '-')
    ax.set_ylim(0, n)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    
    for x0,x1,me,you in one2one:
        #a = ii[y0] / (n-1)
        a = node_yticks.index(me)
        b = node_yticks.index(you)
        # if a>b:
        #     draw(ax,x0,x1,a,b,'black')
        # else:
        #     draw(ax,x0,x1,b,a,'black')
        draw(ax,x0,x1,a,b,'black')
        if a==n-1:
            ax.plot([x0,x1], [a,a], color='black')
    for x0,x1,sender in one2all:
        c = node_yticks.index(sender)
        draw_onetoall(ax,x0,x1,n-1,0,c,'red')
        if c==n-1:
            ax.plot([x0,x1], [c,c], color='black')
    for x0,x1,sender in all2all:
        draw_alltoall(ax,x0,x1,n-1,0,'red')
    return 



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
        print(iter)
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

    print(new_ytickslist)
    return new_ytickslist

if __name__ == "__main__":
    global tend, n, nodelist

    if len(argv) != 2:
        print('usage: python3 temp_both_full_hosp.py [file name]')
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

    tend = max([timed(a) for a in timelist]) #finding stop time
    nodelist.remove('All') 

    node2index = {a:i for i,a in enumerate(nodelist)} #create a dictionary where characters are given associated numerical values

    node2index['All'] = -1 #create new key for 'all' and assign value -1
    nodelist.append('All') #add back to nodelist 

    #have to account for onetoall, alltoall, onetoone cases


    fig = plt.figure()
    ax = fig.add_subplot(111)
    #plt.axis('off') #removal of axes for clarity
    
    one2one = []
    one2all = []
    all2all= []
    for i in range(len(df['Job'])-1):
        try:
            t0 = timed(df['Time'][i]) #time stamps are readable for both start and end times
            t1 = timed(df['Time'][i+1])
            me = df['Job'][i] #find sender and receiver
            you = df['Receiver_Job'][i]
            if me in nodelist and you in nodelist:
                me = node2index[me] #identify corresponding value of sender and receiver 
                you = node2index[you]
                if me != you:
                    if me > -1 and you > -1: #finding not all case
                        #one2one.append([t0,t1,max(me,you),min(me,you)]) #fraction of start and end time corr to max time duration
                        one2one.append([t0,t1,me,you])
                    elif me>-1 and you==-1:
                        one2all.append([t0,t1, me])
                    elif me==-1 and you==-1:
                        all2all.append([t0,t1])
        except:
            continue
                    
                

    ii = get_inx(one2one) #updated indexes
    print(ii)

    #title="Temporal Layout area minimisation layout "+argv[1][:-4]
    my_yticks = list(np.arange(n,dtype=int))
    for i in range(len(my_yticks)):
        my_yticks[i]=ii.index(i)
    
    gen_plot(my_yticks,my_yticks,one2one,one2all,all2all)
    plt.savefig(argv[1][:-3] + '_temporallayout.pdf', format = 'pdf', bbox_inches = 'tight')

    node_yticks=[]
    for ele in my_yticks:
        node_yticks.append(list(node2index.keys())[list(node2index.values()).index(ele)])
    gen_plot(my_yticks,node_yticks,one2one,one2all, all2all)

    plt.savefig(argv[1][:-3] + '_temporallayout1.pdf', format = 'pdf', bbox_inches = 'tight')
    
    df2=expand_df2(df)
    
    for i in range(len(df2['Job'])):
        df2.at[i, 'Time'] = timed(df2['Time'][i]) #for teneto measures computation
        df2.at[i, 'Job'] = node2index[df2['Job'][i]]
        df2.at[i, 'Receiver_Job'] = node2index[df2['Receiver_Job'][i]] #change to indexes
    netin = {'i': df2['Job'], 'j': df2['Receiver_Job'], 't': df2['Time']}
    tnet_df=pd.DataFrame(data=netin)
    tnet= TemporalNetwork(from_df=tnet_df) #get temporal network measures computations
    print("Temporal Degree Centrality") #The sum of all connections each node has through time (either per timepoint or over the entire temporal sequence).
    print(networkmeasures.temporal_degree_centrality(tnet))
    tnet_dict=dict(enumerate(networkmeasures.temporal_degree_centrality(tnet)))
    print(tnet_dict)
    
    #title_rn="Temporal Layout with RN layout "+argv[1][:-4]
    my_yticks_rn = recurr_neighbourpos(tnet_dict, df2)
    my_yticks_rn.reverse()
    gen_plot(my_yticks_rn,my_yticks_rn,one2one,one2all, all2all)

    plt.savefig(argv[1][:-3] + '_temporallayout_rn.pdf', format = 'pdf', bbox_inches = 'tight')

        
    node_yticks_rn=[]
    for ele in my_yticks_rn:
        node_yticks_rn.append(list(node2index.keys())[list(node2index.values()).index(ele)])
    gen_plot(my_yticks_rn,node_yticks_rn,one2one,one2all,all2all)

    plt.savefig(argv[1][:-3] + '_temporallayout_rn1.pdf', format = 'pdf', bbox_inches = 'tight')

