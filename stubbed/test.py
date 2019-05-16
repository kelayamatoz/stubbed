import networkx
from stubbed import networkx_stubs
networkx_stubs.stub()

g = networkx.Graph()
networkx.add_path(g, range(100))
g.add_node(101, label=1)

print(networkx_stubs.TraceRegistry)
