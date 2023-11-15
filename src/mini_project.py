import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import tensorflow as tf
import gpflow
# from graph_matern.kernels.graph_matern_kernel import GraphMaternKernel

# create a networkx graph
G = nx.Graph()

# add vertices to G (first two arguments of add_edge)
# and then add edges between those vertices
# and then add weights to those edges
G.add_edge("A", "B", weight=5)
G.add_edge("A", "C", weight=4)
G.add_edge("A", "D", weight=6)
G.add_edge("A", "G", weight=7)
G.add_edge("B", "C", weight=3)
G.add_edge("C", "D", weight=3)
G.add_edge("B", "E", weight=1)
G.add_edge("C", "E", weight=2)
G.add_edge("C", "D", weight=3)
G.add_edge("D", "G", weight=1)
G.add_edge("D", "F", weight=3)
G.add_edge("C", "F", weight=5)
G.add_edge("C", "F", weight=5)
G.add_edge("G", "F", weight=3)
G.add_edge("E", "F", weight=8)

# create the line graph of G
L_of_G = nx.line_graph(G)

# make an array of nodes of L(G)
L_of_G_node_array = np.asarray(L_of_G.nodes)

# print("G edges:", G.edges)
# print("L(G) nodes:", L_of_G.nodes)
# print("L(G) first node:", list(L_of_G.nodes)[0])

# print(list(G.edges))
# print(type(L_of_G.nodes))

# for edge in G.edges:
#     print(G.edges(edge))
#     print(G.edges[edge])

# print(list(G.edges))

# print(G.edges[('A', 'B')])

# set the vertex weights of L(G)
# to be equal to the corresponding edge weights of G
# and make a list of the weights
L_of_G_node_weight_list = []
for node in L_of_G.nodes:
    L_of_G.nodes[node]["weight"] = G.edges[node]["weight"]
    L_of_G_node_weight_list.append(L_of_G.nodes[node]["weight"])

# list the edges of G and their weights
for edge in sorted(list(G.edges)):
    print("G edge:", edge, " Weight:", G.edges[edge]["weight"])

# list the vertices of L(G) and their weights
# (should match above list)
for node in sorted(list(L_of_G.nodes)):
    print("L(G) node:", node, " Weight:", L_of_G.nodes[node]["weight"])

# set the edge weights of L(G) to 1
for edge in L_of_G.edges:
    L_of_G.edges[edge]["weight"] = 1
    # print(L_of_G.edges[edge])

# for edge in sorted(list(L_of_G.edges)):
    # print("L(G) edge:", edge, " Weight:", L_of_G.edges[edge]["weight"])

# print the array of nodes of L(G)
# and the array of vertex weights of L(G)
print("L(G) node array:", L_of_G_node_array)
L_of_G_node_weight_array = np.asarray(L_of_G_node_weight_list)
print("L(G) node weight array:", L_of_G_node_weight_array)

# shuffle the L(G) node array
# and the L(G) node weight array
# together
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

L_of_G_shuffled_nodes, L_of_G_shuffled_node_weights = unison_shuffled_copies(L_of_G_node_array, L_of_G_node_weight_array)

# print the shuffled arrays
print("L(G) shuffled nodes:", L_of_G_shuffled_nodes)
print("L(G) shuffled node weights:", L_of_G_shuffled_node_weights)

# divide the shuffled arrays into
# training and test sets
training_set_proportion = 0.25
training_set_size = round((training_set_proportion * L_of_G_node_array.shape[0]))
training_vertices = L_of_G_shuffled_nodes[:training_set_size]
training_weights = L_of_G_shuffled_node_weights[:training_set_size]
testing_vertices = L_of_G_shuffled_nodes[training_set_size:]
testing_weights = L_of_G_shuffled_node_weights[training_set_size:]

# print the training and test sets
print("training vertices:", training_vertices)
print("training weights:", training_weights)
print("testing vertices:", testing_vertices)
print("testing weights:", testing_weights)

training_data = (training_vertices, training_weights)
testing_data = (testing_vertices, testing_weights)

# more or less pasting in code
# from the Graph-Gaussian-Processes readme
# G_laplacian = nx.laplacian_matrix(G)
# print("G laplacian:", G_laplacian)
laplacian = nx.laplacian_matrix(L_of_G)
print("laplacian:", laplacian)
print(type(laplacian))
laplacian = laplacian.toarray()
print("laplacian:", laplacian)
print(type(laplacian))
laplacian = tf.convert_to_tensor(laplacian, dtype=float)
print("laplacian:", laplacian)
print(type(laplacian))
eigenvalues, eigenvectors = tf.linalg.eigh(laplacian)  # only should be done once-per-graph
# kernel = GraphMaternKernel((eigenvectors, eigenvalues))
# model = gpflow.models.GPR(data=training_data, kernel=kernel)
# need to figure out how GPR works, I guess

# next steps:
# i guess put together code to train and test
# the gaussian process
# i should probably look at
# https://github.com/spbu-math-cs/Graph-Gaussian-Processes/blob/main/examples/regression.ipynb
# and 
# https://gpflow.github.io/GPflow/develop/notebooks/getting_started/basic_usage.html
# for guidance