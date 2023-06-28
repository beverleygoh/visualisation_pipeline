import pandas as pd
import teneto
from teneto import TemporalNetwork
import matplotlib.pyplot as plt
from teneto import networkmeasures
import networkx as nx
from matplotlib import pylab
from pylab import *
import numpy as np

#https://github.com/juancamilog/temporal_centrality/blob/master/temporal_graph.py

class temporal_graph:
    # construct a temporal network that has snapshots from time 0 until time t_end
    def __init__(self, t_end):
        self.t_end = t_end
        # this list will contain the graphs snapshots
        self.snapshots =[]
        for t in range(0,t_end+1):
            self.snapshots.append(nx.Graph())

        # we will use this list of vertices to for all of the snapshots
        self.vertices = []
        # this list will contain all the temporal edges
        self.edges = []
        # this structure will keep our time ordered graph
        self.time_ordered_graph = nx.DiGraph()

    # add new vertices to all the snapshots
    def add_vertices(self,verts):
        self.vertices.extend(verts)
        n = len(verts)
        t=0
        for G_t in self.snapshots:
            # add nodes to the snapshot at  time t
            G_t.add_nodes_from(verts)
            # create time stamped copies of the vertices
            time_stamped_verts = zip(verts,[t]*n)
            # add them to the time ordered graph 
            self.time_ordered_graph.add_nodes_from(time_stamped_verts)
            # add edges from vertex (v,t-1) to (v,t)
            if t > 0:
               new_edges = [((v,t-1),(v,t)) for v in verts]
               self.time_ordered_graph.add_edges_from(new_edges)
            t = t + 1
    # add an edge to the subset of snapshots G_{start_time} to G_{end_time}
    # the edges should come in a tuple ((v_1,v_2), (start_time, end_time))
    def add_temporal_edges(self,edges):
        # only consider the start and end times that fall within 0 and self.t_end
        for e in edges:
            start_time = e[1][0]
            end_time = e[1][1]
            # check if the time frame is valid
            if end_time < start_time or start_time > self.t_end or end_time < 0:
                #ignore this edge if it has an invalid time interval
                continue

            # clip the time interval to the bounds given by self.t_start and self.t_end
            if start_time < 0:
                start_time = self.t_start
            if end_time > self.t_end:
                end_time = self.t_end

            # add edges to the snapshots
            for t in range(start_time, end_time+1):

                self.snapshots[t].add_edge(e[0][0],e[0][1])
                if t > 0:
                    # add edges to the time ordered graph ( from t-1 to t)
                    new_edges = [((e[0][0], t-1), (e[0][1],t)), ((e[0][1],t-1),(e[0][0],t))]
                    self.time_ordered_graph.add_edges_from(new_edges)
           
            # add edges to our global edge list
            self.edges.append((e[0],(start_time,end_time)))

    # add a graph snapshot to the end of the current snapshot list.
    def append_snapshot(self,G_t):
        self.t_end = self.t_end+1
        t= self.t_end
        # TODO check that we are passing a valid snapshot (same vertex set)
        self.snapshots.append(nx.Graph())
        self.snapshots[t].add_nodes_from(self.vertices)
        edgeset = G_t.edges()

        # get timestamped edges from current snapshot
        new_edges = zip(edgeset,[(self.t_end,self.t_end)]*len(edgeset))
        # add list of new edges to the temporal graph
        self.add_temporal_edges(new_edges)

        # append self edges from previous timestep to the current timestep
        new_edges = [((v,t-1),(v,t)) for v in G_t.nodes_iter()]
        self.time_ordered_graph.add_edges_from(new_edges)

        #if self.t_end>1:
        #self.draw_snapshot(self.t_end)
        #    self.draw_time_ordered_graph()
        #    plt.show()

    # draw the snapshot of the network at time t
    def draw_snapshot (self,t,labels =None):
        plt.figure()
        if labels is not None:
            npos = nx.graphviz_layout(self.snapshots[t], prog="fdp")
            nx.draw(self.snapshots[t], pos = npos, node_color='w', node_size=500, with_labels=False)
            nx.draw_networkx_labels(self.snapshots[t], npos, labels)
        else:
            nx.draw(self.snapshots[t], node_color='w', node_size=10, with_labels=True)

        plt.draw()
    
    # draw the time ordered graph derived from G_{i,j}
    def draw_time_ordered_graph(self,v_scale=1.0,h_scale=1.0):
        plt.figure()
        # compute the layout, the horizontal axis represents time and the vertical axeis corresponds to node label
        v_pos = {}
        
        #first, for each vertex we select a value for the vertical coordinate
        y = len(self.vertices)*v_scale
        for v in self.vertices:
            v_pos[v] = y
            y = y - v_scale

        npos = {}
        labels = {}
        # now, for every time step we select a horizontal coordinate and populate the layout dictionary
        
        print(self.time_ordered_graph.nodes())
        for v in self.time_ordered_graph.nodes():
            npos[v] = np.array((h_scale*v[1],v_pos[v[0]]))
            labels[v] = "$%s_{%d}$"%(v[0],v[1])
        nx.draw(self.time_ordered_graph, pos = npos, with_labels=False, node_color='w', node_size=1000)
        nx.draw_networkx_labels(self.time_ordered_graph, npos, labels)
        plt.draw()
        

def compute_temporal_betweenness(G,start_time,end_time):
    import sys
    sys.setswitchinterval(2000)
    # we start at the end time t = G.t_end, and go backwards in time
    verts = G.vertices
    n = len(verts)
    m = end_time - start_time
    labels = {}
    # sets for computing the correct normalization values
    V_s = [None]*n
    V_d = [None]*n
    # compute integer labels
    idx = 0
    for v in verts:
        labels[v] = idx
        V_s[idx] = set()
        V_s[idx].add(idx)
        V_d[idx] = set()
        V_d[idx].add(idx)
        idx = idx + 1
    
    # this dictionary stores the distance matrices D_t
    D = {}
    # this dictionary stores the  D_t
    S = {}
    # this stores the cumulative closeness score of G
    betweenness = np.zeros(n)

    for t in range(end_time,start_time-1,-1): # m time steps
        # this matrix stores the distances at time t
        D[t] = np.ones((n,n))*np.inf
        # this stores the number of shortest paths between two nodes
        S[t] = np.eye(n)

        for v in verts: # n vertices
            vi = labels[v]
            if t < end_time:
                for k_t in G.time_ordered_graph.successors((v,t)): # at most n
                    ki = labels[k_t[0]]
                    D[t][vi,ki] = 1
                    S[t][vi,ki] = 1

                for u in verts: # n_vertices
                    ui = labels[u]
                    # if u is not reachable in one time step
                    if D[t][vi,ui] > 1:
                       for k_t in G.time_ordered_graph.successors((v,t)): # at most n
                             ki = labels[k_t[0]]
                             d = D[t+1][ki,ui] + 1
                             if d < D[t][vi,ui]:
                                 # there is a shortest path through k!
                                 D[t][vi,ui] = d
                                 S[t][vi,ui] = S[t+1][ki,ui]
                             elif d == D[t][vi,ui]:
                                 # we accumulate the number of shortest paths
                                 S[t][vi,ui] += S[t+1][ki,ui]
        #print "shortest temporal path computation done for time %d"%(t)
        vert_indices = labels.values()
        total_its = 0
        for si in vert_indices: # n vertices
            for di in vert_indices: # n vertices
                # s and d should be different
                if si == di:
                    continue
                # there should be a shortest path between s and d  
                if S[t][si,di] <= 0:
                    continue
                norm_const = 1.0/S[t][si,di]
                for vi in vert_indices: # n vertices
                    # v should be different from s and d
                    if vi == si or vi == di: 
                        continue
                    # there should exist a path between s and v
                    if S[t][si,vi] <= 0:
                        continue
                    k_0 = int(D[t][si,vi]) + t
                    total_its +=1
                    V_s[vi].add(si)
                    V_d[vi].add(di)
                    for k in range(k_0,end_time):
                        if S[k][vi,di] > 0:
                            d_tk = D[t][si,vi]
                            if d_tk < k-t:
                                d_tk = k-t
                            d_kj = D[k][vi,di]
                            if D[t][si,di] == d_tk + d_kj:
                                if np.isnan(S[t][si,vi]*S[k][vi,di]/S[t][si,di]):
                                    print("Whoops! %d %d %d"%(si,vi,di))
                                    print (S[t][si,vi],S[k][vi,di],S[t][si,di])
                                    input()
                                betweenness[vi] += S[t][si,vi]*S[k][vi,di]/S[t][si,di]
                                if np.isnan(betweenness[vi]):
                                    print("Whoops! bt is nan, %d %d %d"%(si,vi,di))
                                    print (S[t][si,vi],S[k][vi,di],S[t][si,di])
                                    input()
                        else:
                            # no path will exist for t>k
                            break
                # end for vi
            # end for di
        # end for si
        #print "betweenness computation done for time %d"%(t)
        #print "number of vertex triplets visited for betweennness %d"%(total_its)
    norm_ct = np.zeros(n)
    for vi in range(n):
        if len(V_s[vi]) > 1:
            V_s[vi].remove(vi)
        if len(V_d[vi]) > 1:
            V_d[vi].remove(vi)
        norm_ct[vi] = m*(len(V_s[vi])*len(V_d[vi]))
    #print V_s
    #print V_d
    #print norm_ct
    #betweenness = betweenness/((n-1)*(n-2)*m)
    betweenness = np.divide(betweenness,norm_ct)

    return dict(zip(verts,betweenness))

def compute_temporal_closeness(G,start_time,end_time):
    # we start at the end time t = G.t_end, and go backwards in time
    verts = G.vertices
    n = len(verts)
    m = end_time - start_time
    labels = {}
    # compute integer labels
    idx = 0
    for v in verts:
        labels[v] = idx
        idx = idx + 1
    
    # this matrix stores the distances at time t+1
    D_tplus1 = np.eye(n)

    # this stores the cumulative closeness score of G
    closeness = np.zeros(n)
    for t in range(end_time,start_time-1,-1): # m time steps
        # this matrix stores the distances at time t
        D_t = np.ones((n,n))*np.inf

        for v in verts: # n vertices
            vi = labels[v]
            # k is reachable in one step from v
            for k_t in G.time_ordered_graph.successors((v,t)): # at most n
                ki = labels[k_t[0]]
                D_t[vi,ki] = 1

            if t < end_time:
                for u in verts: # n_vertices
                    ui = labels[u]
                    # if u is not reachable in one time step
                    if D_t[vi,ui] > 1:
                       for k_t in G.time_ordered_graph.successors((v,t)): # at most n
                             ki = labels[k_t[0]]
                             d = D_tplus1[ki,ui] + 1
                             if d < D_t[vi,ui]:
                                 D_t[vi,ui] = d
        D_tplus1 = D_t
        if t < end_time:
            # closeness is the sum of inverse shortest path distances for all v and u (with v!= u)
            closeness += (1/D_t - np.eye(n)).sum(1)
        #print "closeness computation done for time %d"%(t)

    # at the end, we normalize closeness by (|V| - 1)*m
    closeness = closeness/((n-1)*m)
    return dict(zip(verts, closeness))

