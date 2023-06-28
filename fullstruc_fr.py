#fruchterman-reingold model algorithm utilised for display of structural layout of temporal network generated from team datasets

import networkx as nx
import matplotlib.pyplot as plt
import warnings
from sys import argv
import pandas as pd
from math import sqrt
from temporal_graph import *

import matplotlib
from teneto import TemporalNetwork, networkmeasures
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
from matplotlib.pyplot import figure, text
# pip install netgraph
from netgraph._main import BASE_SCALE
from netgraph._utils import (
    _get_unique_nodes,
    _edge_list_to_adjacency_matrix,
)
from netgraph._node_layout import (
    _is_within_bbox,
    _get_temperature_decay,
    _get_fr_repulsion,
    _get_fr_attraction,
    _rescale_to_frame,
    _handle_multiple_components,
    _reduce_node_overlap,
)

DEBUG = False

@_handle_multiple_components
def get_fruchterman_reingold_newton_layout(edges,
                                           edge_weights         = None,
                                           k                    = None,
                                           g                    = 1.,
                                           scale                = None,
                                           origin               = None,
                                           gravitational_center = None,
                                           initial_temperature  = 1.,
                                           total_iterations     = 50,
                                           node_size            = 0,
                                           node_mass            = 1,
                                           node_positions       = None,
                                           fixed_nodes          = None,
                                           *args, **kwargs):
    """Modified Fruchterman-Reingold node layout.

    Uses a modified Fruchterman-Reingold algorithm [Fruchterman1991]_ to compute node positions.
    This algorithm simulates the graph as a physical system, in which nodes repell each other.
    For connected nodes, this repulsion is counteracted by an attractive force exerted by the edges, which are simulated as springs.
    Unlike the original algorithm, there is an additional attractive force pulling nodes towards a gravitational center, in proportion to their masses.

    Parameters
    ----------
    edges : list
        The edges of the graph, with each edge being represented by a (source node ID, target node ID) tuple.
    edge_weights : dict
        Mapping of edges to edge weights.
    k : float or None, default None
        Expected mean edge length. If None, initialized to the sqrt(area / total nodes).
    g : float or None, default 1.
        Gravitational constant that sets the magnitude of the gravitational pull towards the center.
    origin : tuple or None, default None
        The (float x, float y) coordinates corresponding to the lower left hand corner of the bounding box specifying the extent of the canvas.
        If None is given, the origin is placed at (0, 0).
    scale : tuple or None, default None
        The (float x, float y) dimensions representing the width and height of the bounding box specifying the extent of the canvas.
        If None is given, the scale is set to (1, 1).
    gravitational_center : tuple or None, default None
        The (float x, float y) coordinates towards which nodes experience a gravitational pull.
        If None, the gravitational center is placed at the center of the canvas defined by origin and scale.
    total_iterations : int, default 50
        Number of iterations.
    initial_temperature: float, default 1.
        Temperature controls the maximum node displacement on each iteration.
        Temperature is decreased on each iteration to eventually force the algorithm into a particular solution.
        The size of the initial temperature determines how quickly that happens.
        Values should be much smaller than the values of `scale`.
    node_size : scalar or dict, default 0.
        Size (radius) of nodes.
        Providing the correct node size minimises the overlap of nodes in the graph,
        which can otherwise occur if there are many nodes, or if the nodes differ considerably in size.
    node_mass : scalar or dict, default 1.
        Mass of nodes.
        Nodes with higher mass experience a larger gravitational pull towards the center.
    node_positions : dict or None, default None
        Mapping of nodes to their (initial) x,y positions. If None are given,
        nodes are initially placed randomly within the bounding box defined by `origin` and `scale`.
        If the graph has multiple components, explicit initial positions may result in a ValueError,
        if the initial positions fall outside of the area allocated to that specific component.
    fixed_nodes : list or None, default None
        Nodes to keep fixed at their initial positions.

    Returns
    -------
    node_positions : dict
        Dictionary mapping each node ID to (float x, float y) tuple, the node position.

    References
    ----------
    .. [Fruchterman1991] Fruchterman, TMJ and Reingold, EM (1991) ‘Graph drawing by force‐directed placement’,
       Software: Practice and Experience

    """

    # This is just a wrapper around `_fruchterman_reingold`, which implements (the loop body of) the algorithm proper.
    # This wrapper handles the initialization of variables to their defaults (if not explicitely provided),
    # and checks inputs for self-consistency.

    assert len(edges) > 0, "The list of edges has to be non-empty."

    if origin is None:
        if node_positions:
            minima = np.min(list(node_positions.values()), axis=0)
            origin = np.min(np.stack([minima, np.zeros_like(minima)], axis=0), axis=0)
        else:
            origin = np.zeros((2))
    else:
        # ensure that it is an array
        origin = np.array(origin)

    if scale is None:
        if node_positions:
            delta = np.array(list(node_positions.values())) - origin[np.newaxis, :]
            maxima = np.max(delta, axis=0)
            scale = np.max(np.stack([maxima, np.ones_like(maxima)], axis=0), axis=0)
        else:
            scale = np.ones((2))
    else:
        # ensure that it is an array
        scale = np.array(scale)

    assert len(origin) == len(scale), \
        "Arguments `origin` (d={}) and `scale` (d={}) need to have the same number of dimensions!".format(len(origin), len(scale))
    dimensionality = len(origin)

    if gravitational_center is None:
        gravitational_center = origin + 0.5 * scale
    else:
        # ensure that it is an array
        gravitational_center = np.array(gravitational_center)

    if fixed_nodes is None:
        fixed_nodes = []

    connected_nodes = _get_unique_nodes(edges)

    if node_positions is None: # assign random starting positions to all nodes
        node_positions_as_array = np.random.rand(len(connected_nodes), dimensionality) * scale + origin
        unique_nodes = connected_nodes

    else:
        # 1) check input dimensionality
        dimensionality_node_positions = np.array(list(node_positions.values())).shape[1]
        assert dimensionality_node_positions == dimensionality, \
            "The dimensionality of values of `node_positions` (d={}) must match the dimensionality of `origin`/ `scale` (d={})!".format(dimensionality_node_positions, dimensionality)

        is_valid = _is_within_bbox(list(node_positions.values()), origin=origin, scale=scale)
        if not np.all(is_valid):
            error_message = "Some given node positions are not within the data range specified by `origin` and `scale`!"
            error_message += "\n\tOrigin : {}, {}".format(*origin)
            error_message += "\n\tScale  : {}, {}".format(*scale)
            error_message += "\nThe following nodes do not fall within this range:"
            for ii, (node, position) in enumerate(node_positions.items()):
                if not is_valid[ii]:
                    error_message += "\n\t{} : {}".format(node, position)
            error_message += "\nThis error can occur if the graph contains multiple components but some or all node positions are initialised explicitly (i.e. node_positions != None)."
            raise ValueError(error_message)

        # 2) handle discrepancies in nodes listed in node_positions and nodes extracted from edges
        if set(node_positions.keys()) == set(connected_nodes):
            # all starting positions are given;
            # no superfluous nodes in node_positions;
            # nothing left to do
            unique_nodes = connected_nodes
        else:
            # some node positions are provided, but not all
            for node in connected_nodes:
                if not (node in node_positions):
                    warnings.warn("Position of node {} not provided. Initializing to random position within frame.".format(node))
                    node_positions[node] = np.random.rand(2) * scale + origin

            unconnected_nodes = []
            for node in node_positions:
                if not (node in connected_nodes):
                    unconnected_nodes.append(node)
                    fixed_nodes.append(node)
                    # warnings.warn("Node {} appears to be unconnected. The current node position will be kept.".format(node))

            unique_nodes = connected_nodes + unconnected_nodes

        node_positions_as_array = np.array([node_positions[node] for node in unique_nodes])

    total_nodes = len(unique_nodes)

    if isinstance(node_size, (int, float)):
        node_size = node_size * np.ones((total_nodes))
    elif isinstance(node_size, dict):
        node_size = np.array([node_size[node] if node in node_size else 0. for node in unique_nodes])

    if isinstance(node_mass, (int, float)):
        node_mass = node_mass * np.ones((total_nodes))
    elif isinstance(node_mass, dict):
        node_mass = np.array([node_mass[node] if node in node_mass else 0. for node in unique_nodes])

    adjacency = _edge_list_to_adjacency_matrix(
        edges, edge_weights=edge_weights, unique_nodes=unique_nodes)

    # Forces in FR are symmetric.
    # Hence we need to ensure that the adjacency matrix is also symmetric.
    #adjacency = adjacency + adjacency.transpose()

    if fixed_nodes:
        is_mobile = np.array([False if node in fixed_nodes else True for node in unique_nodes], dtype=bool)

        mobile_positions = node_positions_as_array[is_mobile]
        fixed_positions = node_positions_as_array[~is_mobile]

        mobile_node_sizes = node_size[is_mobile]
        fixed_node_sizes = node_size[~is_mobile]

        mobile_node_masses = node_mass[is_mobile]
        fixed_node_masses = node_mass[~is_mobile]

        # reorder adjacency
        total_mobile = np.sum(is_mobile)
        reordered = np.zeros((adjacency.shape[0], total_mobile))
        reordered[:total_mobile, :total_mobile] = adjacency[is_mobile][:, is_mobile]
        reordered[total_mobile:, :total_mobile] = adjacency[~is_mobile][:, is_mobile]
        adjacency = reordered
    else:
        is_mobile = np.ones((total_nodes), dtype=bool)

        mobile_positions = node_positions_as_array
        fixed_positions = np.zeros((0, 2))

        mobile_node_sizes = node_size
        fixed_node_sizes = np.array([])

        mobile_node_masses = node_mass
        fixed_node_masses = np.array([])

    if k is None:
        area = np.product(scale)
        k = np.sqrt(area / float(total_nodes)) #default area is 1

    temperatures = _get_temperature_decay(initial_temperature, total_iterations)

    # --------------------------------------------------------------------------------
    # main loop

    for ii, temperature in enumerate(temperatures):
        candidate_positions = _fruchterman_reingold_newton(mobile_positions, fixed_positions,
                                                           mobile_node_sizes, fixed_node_sizes,
                                                           adjacency, temperature, k,
                                                           mobile_node_masses, fixed_node_masses,
                                                           gravitational_center, g)
        is_valid = _is_within_bbox(candidate_positions, origin=origin, scale=scale)
        mobile_positions[is_valid] = candidate_positions[is_valid]

    # --------------------------------------------------------------------------------
    # format output

    node_positions_as_array[is_mobile] = mobile_positions

    if np.all(is_mobile):
        node_positions_as_array = _rescale_to_frame(node_positions_as_array, origin, scale)

    node_positions = dict(zip(unique_nodes, node_positions_as_array))

    return node_positions


def _fruchterman_reingold_newton(mobile_positions, fixed_positions,
                                 mobile_node_radii, fixed_node_radii,
                                 adjacency, temperature, k,
                                 mobile_node_masses, fixed_node_masses,
                                 gravitational_center, g):
    """Inner loop of modified Fruchterman-Reingold layout algorithm."""

    combined_positions = np.concatenate([mobile_positions, fixed_positions], axis=0)
    combined_node_radii = np.concatenate([mobile_node_radii, fixed_node_radii])

    delta = mobile_positions[np.newaxis, :, :] - combined_positions[:, np.newaxis, :]
    distance = np.linalg.norm(delta, axis=-1)

    # alternatively: (hack adapted from igraph)
    if np.sum(distance==0) - np.trace(distance==0) > 0: # i.e. if off-diagonal entries in distance are zero
        warnings.warn("Some nodes have the same position; repulsion between the nodes is undefined.")
        rand_delta = np.random.rand(*delta.shape) * 1e-9
        is_zero = distance <= 0
        delta[is_zero] = rand_delta[is_zero]
        distance = np.linalg.norm(delta, axis=-1)

    # subtract node radii from distances to prevent nodes from overlapping
    distance -= (mobile_node_radii[np.newaxis, :] + combined_node_radii[:, np.newaxis]*3) #to further extend distance of separation

    # prevent distances from becoming less than zero due to overlap of nodes
    distance[distance <= 0.] = 1e-6 # 1e-13 is numerical accuracy, and we will be taking the square shortly

    with np.errstate(divide='ignore', invalid='ignore'):
        direction = delta / distance[..., None] # i.e. the unit vector

    # calculate forces
    repulsion    = _get_fr_repulsion(distance, direction, k)
    attraction   = _get_fr_attraction(distance, direction, adjacency, k)
    gravity      = _get_gravitational_pull(mobile_positions, mobile_node_masses, gravitational_center, g)

    if DEBUG:
        r = np.median(np.linalg.norm(repulsion, axis=-1))
        a = np.median(np.linalg.norm(attraction, axis=-1))
        g = np.median(np.linalg.norm(gravity, axis=-1))
        print(r, a, g)

    displacement = attraction + repulsion + gravity

    # limit maximum displacement using temperature
    displacement_length = np.linalg.norm(displacement, axis=-1)
    displacement = displacement / displacement_length[:, None] * np.clip(displacement_length, None, temperature)[:, None]

    mobile_positions = mobile_positions + displacement

    return mobile_positions


def _get_gravitational_pull(mobile_positions, mobile_node_masses, gravitational_center, g):
    delta = gravitational_center[np.newaxis, :] - mobile_positions
    direction = delta / np.linalg.norm(delta, axis=-1)[:, np.newaxis]
    magnitude = mobile_node_masses - np.mean(mobile_node_masses)
    return g * magnitude[:, np.newaxis] * direction

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
    new_df=pd.DataFrame()
    nodelist.remove('all')
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
            
if __name__ == '__main__':

    import networkx as nx
    from netgraph import Graph
    global edge_filterdf, node_index, nodelist

    if len(argv) != 2:
        print('usage: python3 reordernodes_structural3-1-1.py [file name]')
        exit()

    df = pd.read_csv(argv[1],delimiter = ';')
    df=df.dropna() #drop rows with NaN values
    df=df.reset_index(drop=True)
    
    df=expand_df(df)
    #random.seed(37)
    fromlist = df['From'].tolist()
    
    tolist = df['To'].tolist()
    nodelist = list(set(tolist).union(set(fromlist)))
    df=expand_df2(df)
    node2index = {a:i for i,a in enumerate(nodelist)} #create a dictionary where characters are given associated numerical values
    print(node2index) #in order of outputted activity levels
    dura_time=[]
    for i in range(len(df['Duration'])):
        dura_time.append(time(df['Duration'][i]))
    
    df['timed_duration']=dura_time
    event=1
    event_lst=[]
    for i in range(len(df['From'])):
        df.at[i, 'In'] = time(df['In'][i]) #for teneto measures computation
        #df.at[i+1, 'In'] = timed(df['In'][i+1])
        df.at[i, 'From'] = node2index[df['From'][i]] #change to indexes for output array
        df.at[i, 'To'] = node2index[df['To'][i]]
        event_lst.append(event)
        if i!=(len(df['From'])-1):
            if df['In'][i]<time(df['In'][i+1]):
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
    
    
    # node_positions = get_fruchterman_reingold_newton_layout( #added k param to increase edge length
    #     list(G.edges()), k=10/sqrt(len(nodelist)),edge_weights=edge_wts,
    #     node_size={key : val  for key, val in tnet_dict.items()}, #according to temporal measures
    #     node_mass=mod_close_dict, g=2 #closeness centrality affecting node mass for stronger gravitational pull
    # )
    
    fig, ax = plt.subplots()
    
    try:
        #metric_df=format_metric(df)
        print("Temporal Betweenness centrality dict:")
        betw_dict=get_between_dict2(df) #vals for node colour gradient
        print(betw_dict)
        print("Modified Temporal Closeness centrality dict:")
        mod_close_dict=get_closeness_dict2(df) #proportionate to node mass
        print(mod_close_dict)
    
        n_color= np.asarray([val for key, val in betw_dict.items()])

        node_positions = get_fruchterman_reingold_newton_layout( #added k param to increase edge length
        list(G.edges()), k=10/sqrt(len(nodelist)),edge_weights=edge_wts,
        node_size={key : val  for key, val in tnet_dict.items()}, #according to temporal measures
        node_mass=mod_close_dict, g=2 #closeness centrality affecting node mass for stronger gravitational pull
        )
        res=nx.draw_networkx(G,pos=node_positions,edgelist=list(edge_wts.keys()),nodelist= list(betw_dict.keys()), alpha=0.7,node_color= n_color, cmap='viridis',node_size= [tnet_dict[v] for v in list(betw_dict.keys())], width=np.array([(val/sum(list(edge_wts.values())))*30 for key, val in edge_wts.items()]),with_labels= False,connectionstyle='arc3, rad = 0.1', vmin=min(n_color), vmax=max(n_color)) #create multiple edges 
        
        
        for node, (x, y) in node_positions.items():
            val=(tnet_dict[node]/sum(list(tnet_dict.values())))*40+12
            text(x, y+0.05, list(node2index.keys())[list(node2index.values()).index(node)], fontsize=val*2,ha='center', va='center')
            
        
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin = min(n_color), vmax=max(n_color)))
        cbar=fig.colorbar(sm)
        cbar.ax.tick_params(labelsize=15)
        
        
        cbar.set_label('temporal betweenness centrality',fontsize=20)
    
    except:
        n_color= np.asarray([val for key, val in tnet_d2.items()]) #colour according to degree centrality

        node_positions = get_fruchterman_reingold_newton_layout( #added k param to increase edge length; default is 1/sqrt...
        list(G.edges()), k=10/sqrt(len(nodelist)),edge_weights=edge_wts,
        node_size={key : val  for key, val in tnet_dict.items()}, #according to temporal measures
        node_mass=tnet_dict, g=2 
        ) #node mass affected by temporal degree centrality
        res=nx.draw_networkx(G,pos=node_positions,edgelist=list(edge_wts.keys()),nodelist= list(tnet_dict.keys()), alpha=0.7,node_color= n_color, cmap='viridis',node_size= [v for k,v in tnet_dict.items()], width=np.array([(val/sum(list(edge_wts.values())))*30 for key, val in edge_wts.items()]),with_labels= False,connectionstyle='arc3, rad = 0.1', vmin=min(n_color), vmax=max(n_color)) #create multiple edges 
        
    
        
        for node, (x, y) in node_positions.items():
            val=(tnet_dict[node]/sum(list(tnet_dict.values())))*40+12
            text(x, y+0.1, list(node2index.keys())[list(node2index.values()).index(node)], fontsize=val*2,ha='center', va='center')
            
        
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin = min(n_color), vmax=max(n_color)))
        cbar=fig.colorbar(sm)
        cbar.ax.tick_params(labelsize=15)
        
        
        cbar.set_label('temporal degree centrality',fontsize=20)
        
    plt.axis("off")
    fig.set_size_inches(13, 10)
    plt.savefig(argv[1][:-3] + '_structurallo3_1.pdf', format = 'pdf', bbox_inches = 'tight')
    plt.show()
    
#currently, layout uses the fruchterman algorithm, with node masses proportionate to temporal closeness centrality values; edge widths to be proportionate to total duration of interactions between respective nodes and colour map to display colour gradient according to betweenness centrality 

#1. spatial arrangement (layout of choice-> node reordering to reduce overlap)
#2. improve aesthetics (redundant coding)
#3. edge selection
