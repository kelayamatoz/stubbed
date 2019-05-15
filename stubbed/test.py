import networkx
from stubbed import networkx_stubs

g = networkx.Graph()
networkx.add_path(g, range(100))
g.add_node(101, label=1)

100 in g
