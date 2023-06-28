import matplotlib
matplotlib.use('Agg')
import csv
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import numpy as np
from sys import argv
from copy import deepcopy, copy
from alive_progress import alive_bar
from scipy.spatial import ConvexHull
from scipy.ndimage import uniform_filter1d
import networkx as nx
import math
import pandas as pd
import graphviz
from itertools import permutations
from statistics import mean
from matplotlib.pyplot import figure, text
layout_step = 0.2 # for the layout algorithm
layout_k = 5.0 # preferred edge length
layout_c = 0.3 # Relative strength of repulsive forces
layout_p = 2.0 # Repulsive force exponent
size_fac = 0.02 # rescaling node sizes

expansion_fac = 1.5 # how much larger blobs should be after the smoothing
link_weight_low = 0.5 # lower link weight cut off
node_size_low = 0.9 # lower node size limit
weight2sizefac = 0.2 # for converting node weight to size
weight2widthfac = 0.15 # for converting link weight to width

nwindow = 600 # for smoothing of bounding box in the time dimension
xymargin = 0.2 # fraction of margins to the (non-strict) bounding box

niter = 2 # how long to iterate the spring algorithm

clock_x = 1.0 # setting dimensions of the clock
clock_y = 1.0
clock_radius = 0.05

nframes = 10 # number of frames per round 4 sec = 4 * 25 fps = 100 

dt = 20 # time (in seconds) between frames

fpm = 60 // dt # frames per minute
fph = fpm * 60 * nframes # frames per hour
fpd = 12 * fph # frames per 12h

frame0 = int(fph * 8.5) # 8:30 AM = time of initial frame

decay_fac = 0.9 # for edge weights
frame_decay_fac = decay_fac ** (1.0 / nframes)
weight_threshold = 0.1
fig, ax = plt.subplots()
#ax.axis("equal")
fig.set_size_inches(12, 10)
#node_ids = {} # translating between the file and internal ID numbers. These should rather be replaced byt graph meta data
# teacher = {} # indicator function of the proterty of being a teacher
# teacher_class_count = {} # counting the times a teached is in connection with a student (to assign a class to them)
# node_class = {} # deictionary giving the class of a node
# class_nodes = [[] for i in range(10)]

link_weight = 1.0 # all links have this weight in the layout algorithm

colors = [[32/255,142/255,183/255],[10/255,79/255,78/255],[68/255,124/255,254/255],[115/255,42/255,102/255],[202/255,80/255,211/255],[99/255,20/255,175/255],[136/255,125/255,175/255],[63/255,67/255,109/255],[226/255,50/255,9/255],[110/255,57/255,13/255]]

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

def time (s): #min

    try:
        a = str(s).strip().split('.')
        b = int(a[0]) * 60.0 + int(a[1]) + int(a[2])/60.0
    except:
        return -1

    return b

def draw_clock (time_val, ax):
    global aspect
    
    ax.add_artist(matplotlib.patches.Ellipse((clock_x, clock_y), 2.0 * clock_radius , 2.0 * clock_radius, linewidth = 1.5, edgecolor = 'black', facecolor = 'none', zorder = 1))
    x = float(time_val/tend)
    hand_dx = clock_radius * np.sin(2.0 * np.pi * x)
    hand_dy = clock_radius * np.cos(2.0 * np.pi * x)
	
    xx = [clock_x, clock_x + hand_dx]
    yy = [clock_y, clock_y + hand_dy]
    ax.add_artist(matplotlib.lines.Line2D(xx, yy, linewidth = 1.5, color = 'black', solid_capstyle = 'round', zorder = 1))
	

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
# calculating the area of a blob (for expanding it)

def polygon_area (xs, ys):

	return 0.5 * (np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
# calculating the coordinates of a centroid

def polygon_centroid (yx):

	xs = yx[:,0]
	ys = yx[:,1]
	xy = np.array([xs, ys])

	return np.dot(xy + np.roll(xy, 1, axis = 1), xs * np.roll(ys, 1) - np.roll(xs, 1) * ys) / (6 * polygon_area(xs, ys))


def chaikins_corner_cutting (a, refinements = 1):

	for i in range(refinements):
		nn = a.shape[0]
		qr = np.zeros((2 * nn - 1,2))

		for j in range(1,nn):
			qr[2*j-2,:] = 0.75 * a[j-1,:] + 0.25 * a[j,:]
			qr[2*j-1,:] = 0.25 * a[j-1,:] + 0.75 * a[j,:]
		qr[-2,:] = 0.75 * a[-1,:] + 0.25 * a[0,:]
		qr[-1,:] = 0.25 * a[-1,:] + 0.75 * a[0,:]

		a = qr

	return qr

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
# expanding the size of a blob around its centroid

def expand_path (xy):

	c = polygon_centroid(xy)

	a = np.empty_like(xy)

	for i in range(xy.shape[0]):
		for j in range(2):
			a[i,j] = (xy[i,j] - c[j]) * expansion_fac + c[j]

	return a

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
# creating a blob roughly (but not certainly) enclosing the students and teacher of a class

def smooth_patch (points):

	points = np.array(points)

	# get the convex hull of the nodes of a class
	hull = ConvexHull(points)

	a = np.array([points[hull.vertices,0],points[hull.vertices,1]]).T
	b = chaikins_corner_cutting(a, 5) # smoothing

	b = expand_path(b)
	b = np.append(b,b[0:1,:], axis = 0)

	codes = [mpath.Path.LINETO] * b.shape[0] # all elements are LINETO except the first (Bezier curves seems a bit unstable)
	codes[0] = mpath.Path.MOVETO

	return mpath.Path([(b[i,0],b[i,1]) for i in range(b.shape[0])], codes)

# mixing the colors for cross-class links

def get_color (c1, c2):

	return [0.5*(c1[0]+c2[0]),0.5*(c1[1]+c2[1]),0.5*(c1[2]+c2[2])]

def expand_df(df): #does not expand one to all cases
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

def time (s):

    try:
        a = str(s).strip().split('.')
        b = int(a[0]) * 60.0*60.0 + int(a[1])*60.0 + int(a[2])
    except:
        return -1

    return b


def update(num):
    global node_position, df, nsize
    ax.clear()
    i = int(num / 10) #will iterate through same row information 10 times
    r= num%10 #gets remainder; if 0, need recall prev node position
    cpoints=[]
    
    G = nx.DiGraph()
    
    nodelist2=[i for i in range(n)]
    G.add_nodes_from(nodelist2)
    if df['To'][i]!=-1:
        G.add_edge(df['From'][i],df['To'][i],weight=((10-r)/10))
        if r==0:
            nsize[df['From'][i]]+=30

        
    elif df['To'][i]==-1:
        if r==0:
            nsize[df['From'][i]]+=30*(n-1)
        for ele in nodelist2:
            if ele!=df['From'][i]:
                G.add_edge(df['From'][i],ele,weight=((10-r)/10))
                
                    
    pos=nx.spring_layout(G, pos=node_position, weight='weight') 
    #node_position=pos.copy()
    
    x_lst=[]
    y_lst=[]
    
    for j in range(n):
        x_lst.append(pos[j][0])
        y_lst.append(pos[j][1])
    
    min_x=min(x_lst)
    max_x=max(x_lst)
    lx=max_x-min_x
    min_y=min(y_lst)
    max_y=max(y_lst)
    ly=max_y-min_y
    xlen,ylen = matplotlib.pyplot.rcParams.get('figure.figsize')
    ax.set_xlim(-xymargin, 1.0 + xymargin)
    ax.set_ylim(-xymargin, 1.0 + xymargin)
    matplotlib.pyplot.axis('off')
    #print('hi')

    
    for j in range(n):
        pos[j][0]=float((pos[j][0]-min_x)/lx)
        
        pos[j][1]=float((pos[j][1]-min_y)/ly)
        
    
    node_position=pos.copy()
    
    if df['To'][i]==-1:
        #print(i)
        #print('bye')
        cpoints.append(pos[df['From'][i]])
        for ele in nodelist2:
            if ele!=df['From'][i]:
                cpoints.append(pos[ele])
        
        ax.add_patch(mpatches.PathPatch(smooth_patch(cpoints), facecolor = colors[df['From'][i]], linestyle = '', alpha = 0.4, zorder = 3))       
    
    nx.draw_networkx(G,ax=ax,pos=node_position,nodelist=nodelist2,node_color= colors[:n], node_size=nsize, with_labels=False)
    for node, (x, y) in pos.items():
            text(x, y, list(node2index.keys())[list(node2index.values()).index(node)], fontsize=12*(nsize[node]/300),color='white',ha='center', va='center') 

    draw_clock(df['In'][i],ax)
    ax.set_xticks([])
    ax.set_yticks([])
    
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

if __name__ == "__main__":
    global n, aspect, node2index, tend, nsize, node_position, df#nsize is for increasing size of nodes according to number of interactions over time

    if len(argv) != 2:
        print('isage: python3 anime.py [file name]')
        exit()
        
    df = pd.read_csv(argv[1],delimiter = ';')
    df=df.dropna() #drop rows with NaN values
    df=df.reset_index(drop=True)
    fromlist = df['From'].tolist()
    #print(list(df['From'].unique()))
    df2=df.copy()
    df=expand_df(df)
    tolist = df['To'].tolist()
    nodelist = list(set(tolist).union(set(fromlist)))

    timelist = df['Out'].tolist()

    tend = max([time(a) for a in timelist]) #finding stop time

    nodelist.remove('all') 
    n= len(nodelist)
    node2index = {a:i for i,a in enumerate(nodelist)} #create a dictionary where characters are given associated numerical values

    node2index['all'] = -1 #create new key for 'all' and assign value -1
    nodelist.append('all') #add back to nodelist
    one2one_all=[]
    one2one=[] #for ii derivation
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
                    one2one_all.append([t0,t1,me,you]) #t0, t1 for clock display, me, you for node interactions
                    if me > -1 and you > -1: #finding not all case
                        one2one.append([t0,t1,max(me,you),min(me,you)])
                    
	# read and construct network
    ii = get_inx(one2one) #updated indexes
    print(ii)
    
    print(df['From'].unique())
    print(df['To'].unique())
    for i in range(len(df['From'])):
        df.at[i, 'In'] = time(df['In'][i]) #for teneto measures computation
        df.at[i, 'From'] = node2index[df['From'][i]] #change to indexes for output array
        df.at[i, 'To'] = node2index[df['To'][i]]
    
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
        
    print(len(df['From']))
    nsize=[300]*n
    ani = animation.FuncAnimation(fig, update, frames=len(df['From'])*10)
    FFwriter = animation.FFMpegWriter()
    ani.save("dynamic.mp4", writer=FFwriter)
    plt.show()

    
    
