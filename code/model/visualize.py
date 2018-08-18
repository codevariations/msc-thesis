from nltk.corpus import wordnet as wn
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
matplotlib("Agg")

def closure_graph(synset, fn):
    seen = set()
    graph = nx.DiGraph()

    def recurse(s):
        if not s in seen:
           seen.add(s)
           graph.add_node(s.name)
           for s1 in fn(s):
               graph.add_node(s1.name)
               graph.add_edge(s.name, s1.name)
               recurse(s1)

    recurse(synset)
    return graph

dog = wn.synset('dog.n.01')
graph = closure_graph(dog, lambda s: s.hypernyms())

nx.draw(graph)
plt.show()
