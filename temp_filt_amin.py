#edge selection implementation on temporal layouts of team datasets using area minimisation node reordering 

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


#no pre or post processing because does not take into account any temporal measures for display; displays temporal network as it is to reflect visualisation patterns

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
    for jj in permutations(ix): #finding all possible permutations positions 0 to final node index
        a = area(jj,o2o) #reducing the rect area

        if a < amin:
            amin = a
            print('new smallest', amin)
            ii = copy(jj)
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
#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #


def time (s): #min

    try:
        a = str(s).strip().split('.')
        b = int(a[0]) * 60.0 + int(a[1]) + int(a[2])/60.0
    except:
        return -1

    return b

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #

# main routine including the drawing
def gen_plot(node_yticks,one2one,one2all,ii): 
    plt.figure()
    fig, ax= plt.subplots()
    # node_yticks=[]
    # for ele in list(ii):
    #     node_yticks.append(list(node2index.keys())[list(node2index.values()).index(ele)])
    #plt.title(title, fontsize=20)
    plt.yticks(np.arange(n,dtype=int), node_yticks, fontsize=15)
    plt.xticks(np.arange(0, tend, 10), fontsize=10)
    #ax.set_xticklabels([])
    plt.tick_params(bottom = False)
    plt.xlabel('Time (min)', fontsize=18)
    plt.ylabel('Interacting parties', fontsize=18)
    plt.grid(axis='y', linestyle= '-')
    ax.set_ylim(0, n)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    
    for x0,x1,me,you in one2one:
        #a = ii[y0] / (n-1)
        a = ii[me]
        b = ii[you]
        draw(ax,x0,x1,a,b,'black')
        if a==n-1 or a==0:
            ax.plot([x0,x1], [a,a], color='black')
        # if a > b:
        #     draw(ax,x0,x1,a,b,'black')
        # else:
        #     draw(ax,x0,x1,b,a,'black')
    for x0,x1,sender in one2all:
        c=ii[sender]
        draw_onetoall(ax,x0,x1,n-1,0,c,'red')
        if c==n-1:
            ax.plot([x0,x1], [c,c], color='black')
    return 

def f(x):
    tau=edge_filterdf['Count'].sum()
    #node_index=list(edge_filterdf['From'].unique())
    n= len(node_index)
    F = np.empty((n))
    for i in range(n):
        indx=list(np.where(edge_filterdf['From']==node_index[i])[0]) #indexes where i equals certain value
        fn=0
        for ele in indx:
            fn+=(edge_filterdf['Count'][ele]-tau*x[node_index[i]]*x[edge_filterdf['To'][ele]])/(1-x[node_index[i]]*x[edge_filterdf['To'][ele]])
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
                    edgedf_pos=list(np.where((edge_filterdf["From"]==node_index[i])&(edge_filterdf["To"]==node_index[j]))[0])[0]
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

if __name__ == "__main__":
    global tend, n, node_index, edge_filterdf, nodelist

    if len(argv) != 2:
        print('usage: python3 temporal_filt4-1.py [file name]')
        exit()

    df = pd.read_csv(argv[1],delimiter = ';')
    df=df.dropna() #drop rows with NaN values
    df=df.reset_index(drop=True)
    df=expand_df(df)
    tolist = df['To'].tolist()
    fromlist = df['From'].tolist()
    nodelist = list(set(tolist).union(set(fromlist)))
    timelist = df['Out'].tolist()

    tend = max([time(a) for a in timelist]) #finding stop time

    nodelist.remove('all') 

    node2index = {a:i for i,a in enumerate(nodelist)} #create a dictionary where characters are given associated numerical values

    node2index['all'] = -1 #create new key for 'all' and assign value -1
    nodelist.append('all') #add back to nodelist 

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #plt.axis('off') #removal of axes for clarity
    
    one2one = []
    one2all = []
    for i in df.index:
        t0 = time(df['In'][i]) #time stamps are readable for both start and end times
        t1 = time(df['Out'][i])
        if t0 > 0 and t1 > 0:
            me = df['From'][i] #find sender and receiver
            you = df['To'][i]

            if me in nodelist and you in nodelist:
                me = node2index[me] #identify corresponding value of sender and receiver 
                you = node2index[you]
                if me != you:
                    if me > -1 and you > -1: #finding not all case
                        one2one.append([t0,t1,me,you]) #fraction of start and end time corr to max time duration
                    else:
                        one2all.append([t0,t1, me])

    ii = get_inx(one2one) #updated indexes
    print(ii)

    df2=expand_df2(df) #previously taken from visualisation, didnt fully expand all cases
    
    edge_filterdf=df2.groupby(["From", "To"]).size().reset_index(name="Count") # get filtered edges 
    for i in range(len(edge_filterdf['From'])):
        edge_filterdf.at[i, 'From'] = node2index[edge_filterdf['From'][i]] #change to indexes for output array
        edge_filterdf.at[i, 'To'] = node2index[edge_filterdf['To'][i]]
        
    node_index= list(edge_filterdf['To'].unique()) #indexes to represent the different nodes
    val = 1/len(node_index)#guessed activity levels, assumed equal
    print("Equal act values (start val):"+ str(val))
    Guess=np.zeros(shape=(len(node_index),0))
    for i in range(len(node_index)):
        Guess=np.append(Guess,val)
    res = fsolve(f,Guess)
    print("Significance matrix")
    print(res)
    #print(res[1])
    arr=significance_test(res,alpha=0.01)
    print(arr)
    
    new_one2one=[]
    for i in df.index: #want to consider those one to all cases still
        t0 = time(df['In'][i]) #time stamps are readable for both start and end times
        t1 = time(df['Out'][i])
        if t0 > 0 and t1 > 0:
            me = df['From'][i] #find sender and receiver
            you = df['To'][i]

            if me in nodelist and you in nodelist:
                me = node2index[me] #identify corresponding value of sender and receiver 
                you = node2index[you]
                if me != you:
                    if me > -1 and you > -1 and arr[me][you]==1: #finding not all case
                        new_one2one.append([t0,t1,me,you]) #fraction of start and end time corr to max time duration
                
    title="Temporal Layout backbone with area minimisation layout "+argv[1][:-4]
    my_yticks = list(np.arange(n,dtype=int))
    for i in range(len(my_yticks)):
        my_yticks[i]=ii.index(i)
    gen_plot(my_yticks,new_one2one,one2all,ii)

    plt.savefig(argv[1][:-3] + '_temporalfilt4_1.pdf', format = 'pdf', bbox_inches = 'tight')
    
        
    node_yticks=[]
    for ele in my_yticks:
        node_yticks.append(list(node2index.keys())[list(node2index.values()).index(ele)])
    print(node_yticks)
    gen_plot(node_yticks,new_one2one,one2all,ii)

    plt.savefig(argv[1][:-3] + '_temporalfilt4_1_2.pdf', format = 'pdf', bbox_inches = 'tight')
