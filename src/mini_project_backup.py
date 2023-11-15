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

# print("G edges:", G.edges)
# print("L(G) nodes:", L_of_G.nodes)
# print("L(G) first node:", list(L_of_G.nodes)[0])

# print(list(G.edges))
# print(type(L_of_G.nodes))



# # experimenting with ways to
# # randomly partition the vertices of L(G)
# training_set_proportion = 0.25
# print("L(G) nodes:", list(L_of_G.nodes))
L_of_G_node_array = np.asarray(L_of_G.nodes)
# print("L(G) nodes as numpy array:", L_of_G_node_array)
# print("number of nodes of L(G):", L_of_G_node_array.shape[0])
# training_set_size = round((training_set_proportion * L_of_G_node_array.shape[0]))
# L_of_G_random_perm = np.random.permutation(L_of_G_node_array)
# print("random permutation of nodes of L(G):", L_of_G_random_perm)
# training_vertices = L_of_G_random_perm[:training_set_size]
# # print("training set of nodes of L(G):", training_vertices)
# testing_vertices = L_of_G_random_perm[training_set_size:]
# # print("test set of nodes of L(G):", testing_vertices)

L_of_G_node_weight_list = []

# set the vertex weights of L(G)
# to be equal to the corresponding edge weights of G
for node in L_of_G.nodes:
    L_of_G.nodes[node]["weight"] = G.edges[node]["weight"]
    L_of_G_node_weight_list.append(L_of_G.nodes[node]["weight"])

print("L(G) node array:", L_of_G_node_array)
L_of_G_node_weight_array = np.asarray(L_of_G_node_weight_list)
print("L(G) node weight array:", L_of_G_node_weight_array)

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

L_of_G_shuffled_nodes, L_of_G_shuffled_node_weights = unison_shuffled_copies(L_of_G_node_array, L_of_G_node_weight_array)

print("L(G) shuffled nodes:", L_of_G_shuffled_nodes)
print("L(G) shuffled node weights:", L_of_G_shuffled_node_weights)

training_set_proportion = 0.25
training_set_size = round((training_set_proportion * L_of_G_node_array.shape[0]))
training_vertices = L_of_G_shuffled_nodes[training_set_size:]
training_weights = L_of_G_shuffled_node_weights[training_set_size:]
testing_vertices = L_of_G_shuffled_nodes[:training_set_size]
testing_weights = L_of_G_shuffled_node_weights[:training_set_size]

# for edge in G.edges:
#     print(G.edges(edge))
#     print(G.edges[edge])

# print(list(G.edges))

# print(G.edges[('A', 'B')])

# list the edges of G and their weights
for edge in sorted(list(G.edges)):
    print("G edge:", edge, " Weight:", G.edges[edge]["weight"])

# list the vertices of L(G) and their weights
# (should match above list)
for node in sorted(list(L_of_G.nodes)):
    print("L(G) node:", node, " Weight:", L_of_G.nodes[node]["weight"])

# training_weights = []
# testing_weights = []
# for node in L_of_G_random_perm:
#     if node in training_vertices:
#         # print(L_of_G.nodes[node]["weight"])
#         # print(type(L_of_G.nodes[node]["weight"]))
#         training_weights.append(L_of_G.nodes[node]["weight"])
#     else:
#         testing_weights.append(L_of_G.nodes[node]["weight"])

# print("training set of nodes of L(G):", training_vertices)
# print("training set of weights of L(G):", training_weights)
# print("test set of nodes of L(G):", testing_vertices)
# print("test set of weights lf L(G):", testing_weights)

# set the edge weights of L(G) to 1
for edge in L_of_G.edges:
    L_of_G.edges[edge]["weight"] = 1
    # print(L_of_G.edges[edge])

# for edge in sorted(list(L_of_G.edges)):
    # print("L(G) edge:", edge, " Weight:", L_of_G.edges[edge]["weight"])



# more or less pasting in code
# from the Graph-Gaussian-Processes readme
# laplacian = nx.laplacian_matrix(L_of_G)
# eigenvalues, eigenvectors = tf.linalg.eigh(laplacian)  # only should be done once-per-graph
# kernel = GraphMaternKernel((eigenvectors, eigenvalues))
# model = gpflow.models.GPR(data=data, kernel=kernel)
# need to figure out how GPR works, I guess

# # create adjacency matrix
# adj_mat = np.zeros((7, 7))
# adj_mat[0, :] = [0, 1, 1, 1, 0, 0, 1]
# adj_mat[1, :] = [1, 0, 1, 0, 1, 0, 0]
# adj_mat[2, :] = [1, 1, 0, 1, 0, 1, 0]
# adj_mat[3, :] = [1, 0, 1, 0, 0, 0, 1]
# adj_mat[4, :] = [0, 1, 0, 0, 0, 1, 0]
# adj_mat[5, :] = [0, 0, 1, 0, 1, 0, 1]
# adj_mat[6, :] = [1, 0, 0, 1, 0, 1, 0]
# print("Adjacency matrix symmetric?", np.array_equal(adj_mat, np.transpose(adj_mat)))

# # create weight matrix
# weight_mat = np.zeros((7, 7))
# weight_mat[0, :] = [0, 5, 4, 6, 0, 0, 7]
# weight_mat[1, :] = [0, 0, 3, 0, 1, 0, 0]
# weight_mat[2, :] = [0, 0, 0, 3, 0, 5, 0]
# weight_mat[3, :] = [0, 0, 0, 0, 0, 0, 1]
# weight_mat[4, :] = [0, 0, 0, 0, 0, 8, 0]
# weight_mat[5, :] = [0, 0, 1, 0, 1, 0, 1]
# weight_mat[6, :] = [1, 0, 0, 1, 0, 1, 0]

# # create degree matrix
# deg_mat = np.diag((4, 3, 4, 3, 2, 3, 3))

# # create graph Laplacian
# graph_lap = deg_mat - weight_mat

