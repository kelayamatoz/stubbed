import networkx
import collections
import itertools
import typing
import atexit

from stubbed.core import FunctionWrapper


class TraceElement(typing.NamedTuple):
    memory_space: int
    offset: int
    size: int
    type: str


class GraphWrapper(FunctionWrapper):
    graph_element_sizes = collections.defaultdict(int)
    graph_node_id_size = 32  # Bytes
    graph_node_hashtable_size = 128
    graph_edge_hashtable_size = 128
    trace = []


    loc_counter = itertools.count(step=4)
    graph_id_to_loc_map = {}
    NODE_OFFSET = 0
    EDGE_OFFSET = 1
    NODE_HASHSET = 2
    EDGE_HASHSET = 3

    @staticmethod
    def get_location(graph):
        if id(graph) not in GraphWrapper.graph_id_to_loc_map:
            GraphWrapper.graph_id_to_loc_map[id(graph)] = next(GraphWrapper.loc_counter)
        return GraphWrapper.graph_id_to_loc_map[id(graph)]


    @staticmethod
    def get_graph_node_attr_size(graph: networkx.Graph):
        attr_dicts = (j for i, j in graph.nodes.data(True))
        unique_attr_names = set(itertools.chain.from_iterable(attr_dicts))
        return len(unique_attr_names) * 8

    @staticmethod
    def get_graph_edge_attr_size(graph: networkx.Graph):
        print(graph.edges.data(True))
        attr_dicts = (d for i, j, d in graph.edges.data(True))
        unique_attr_names = set(itertools.chain.from_iterable(attr_dicts))
        return len(unique_attr_names) * 8

    @staticmethod
    def dump():
        # freeze trace so that further actions don't affect the trace
        traces = GraphWrapper.trace[:]
        materialized = [trace() for trace in traces]
        GraphWrapper.trace = traces
        print([i for i in materialized if i.size])


class GraphAddNode(GraphWrapper):
    def before(self, graph: networkx.Graph, node, **attrs):
        index = graph.number_of_nodes()
        def materialize():
            return TraceElement(GraphWrapper.get_location(graph) + GraphWrapper.NODE_OFFSET, index, GraphWrapper.get_graph_node_attr_size(graph), "WRITE")
        self.trace.append(materialize)


GraphAddNode.method_hook(networkx.Graph, "add_node")


class GraphAddNodesFrom(GraphWrapper):
    def before(self, graph: networkx.Graph, nodes, **attrs):
        index = graph.number_of_nodes()
        # push current count.
        self.current_count = index

    def after(self, val, graph, *args, **kwargs):
        current_count = self.current_count
        new_count = graph.number_of_nodes()
        def materialize():
            return TraceElement(GraphWrapper.get_location(graph) + GraphWrapper.NODE_OFFSET, self.current_count,
                                GraphWrapper.get_graph_node_attr_size(graph) * (new_count - current_count), "WRITE")

        self.trace.append(materialize)


GraphAddNodesFrom.method_hook(networkx.Graph, "add_nodes_from")


class GraphAddEdge(GraphWrapper):
    def before(self, graph: networkx.Graph, edge, **attrs):
        index = graph.number_of_nodes()
        def materialize():
            return TraceElement(GraphWrapper.get_location(graph) + GraphWrapper.EDGE_OFFSET, index, GraphWrapper.get_graph_edge_attr_size(graph), "WRITE")

        self.trace.append(materialize)

        self.current_node_count = graph.number_of_nodes()

    def after(self, val, graph: networkx.Graph, edge, **attrs):
        current_count = self.current_node_count
        new_count = graph.number_of_nodes()
        if current_count == new_count:
            return
        
        def materialize():
            return TraceElement(GraphWrapper.get_location(graph) + GraphWrapper.NODE_OFFSET, current_count, GraphWrapper.get_graph_node_attr_size(graph), "WRITE")

        self.trace.append(materialize)


GraphAddEdge.method_hook(networkx.Graph, "add_edge")


class GraphAddEdgesFrom(GraphWrapper):
    def before(self, graph: networkx.Graph, nodes, **attrs):
        index = graph.number_of_edges()
        # push current count.
        self.current_count = index

        # since add_edges_from can also be adding nodes, we should also count those
        self.node_count = graph.number_of_nodes()

    def after(self, val, graph, *args, **kwargs):
        current_count = self.current_count
        new_count = graph.number_of_edges()

        current_node_count = self.node_count
        new_node_count = graph.number_of_nodes()

        def materialize():
            return TraceElement(GraphWrapper.get_location(graph) + GraphWrapper.EDGE_OFFSET, current_count,
                                GraphWrapper.get_graph_edge_attr_size(graph) * (new_count - current_count), "WRITE")
        self.trace.append(materialize)

        def materialize_nodes():
            return TraceElement(GraphWrapper.get_location(graph) + GraphWrapper.NODE_OFFSET, current_node_count,
                                GraphWrapper.get_graph_node_attr_size(graph) * (new_node_count - current_node_count), "WRITE")
        self.trace.append(materialize_nodes)


GraphAddEdgesFrom.method_hook(networkx.Graph, "add_edges_from")


class GraphContainsNode(GraphWrapper):
    def before(self, graph, n, *args, **kwargs):
        def materialize():
            return TraceElement(GraphWrapper.get_location(graph) + GraphWrapper.NODE_HASHSET, hash(n) % GraphWrapper.graph_node_hashtable_size,
                                GraphWrapper.graph_node_id_size, "READ")
        self.trace.append(materialize)


GraphContainsNode.method_hook(networkx.Graph, "__contains__")
GraphContainsNode.method_hook(networkx.Graph, "has_node")

atexit.register(GraphWrapper.dump)
