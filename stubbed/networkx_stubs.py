import networkx

from stubbed.containers.std import TrackedDict


def stub():
    networkx.Graph.node_dict_factory = TrackedDict
    networkx.Graph.node_attr_dict_factory = TrackedDict
    networkx.Graph.adjlist_inner_dict_factory = TrackedDict
    networkx.Graph.adjlist_outer_dict_factory = TrackedDict
    networkx.Graph.edge_attr_dict_factory = TrackedDict
