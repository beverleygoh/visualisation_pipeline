# Visual network exploration of intra-team dynamics

 The code compiled in the different files create temporal network visualisations of intra-team dynamics in different kinds of interaction contexts. The team discussion datasets are obtained from a business idea generation process as part of an entrepreneurship oprogram hosted by Waseda University. The hospital dataset involves data collected across the span of a surgical procedure.
 
 The different methods of node reordering and layout display explored in the project are given here. The dynamic plot refers to a possible direction of future work that could incorporate dynamic features to represent the time-varying nature of the temporal network, with further incorporation of possible user interface functionalities. 
 
 Temporal centrality metrics were computed based on results proposed by proposed by Hyounshick Kim and Ross Anderson in their paper on "Temporal Node Centrality in Complex Networks" available at http://journals.aps.org/pre/abstract/10.1103/PhysRevE.85.026107, with implementation modified from https://github.com/juancamilog/temporal_centrality/blob/master/temporal_graph.py. The literature accounts for temporal steps taken at regular intervals across the interaction time frame. However, given the continuos nature of the discussion, I have chosen to consider the discretised interaction instances as corresponding to the discretised time step. The assumption of these interaction instances being regularly spaced was made.

The binary stress model was obtained from https://github.com/tomzhch/IES-Backbone/blob/master/bsm.py and incorporated into the structural layout implementation of the team and hospital datasets.

 
