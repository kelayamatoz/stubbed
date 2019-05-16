import networkx
from stubbed import networkx_stubs
networkx_stubs.stub()

g = networkx.Graph()
networkx.add_path(g, range(3))

networkx_stubs.dump()
