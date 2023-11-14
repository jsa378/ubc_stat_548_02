import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import tensorflow as tf
import gpflow
from graph_matern.kernels.graph_matern_kernel import GraphMaternKernel

G = nx.Graph()

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

L_of_G = nx.line_graph(G)

# print("G edges:", G.edges)
# print("L(G) nodes:", L_of_G.nodes)
# print("L(G) first node:", list(L_of_G.nodes)[0])

# print(list(G.edges))
# print(list(L_of_G.nodes))

for node in L_of_G.nodes:
    L_of_G.nodes[node]["weight"] = G.edges[node]["weight"]

# for edge in G.edges:
#     print(G.edges(edge))
#     print(G.edges[edge])

# print(list(G.edges))

# print(G.edges[('A', 'B')])

# for edge in sorted(list(G.edges)):
#     print("G edge:", edge, " Weight:", G.edges[edge]["weight"])

# for node in sorted(list(L_of_G.nodes)):
#     print("L(G) node:", node, " Weight:", L_of_G.nodes[node]["weight"])

for edge in L_of_G.edges:
    L_of_G.edges[edge]["weight"] = 1
    print(L_of_G.edges[edge])

for edge in sorted(list(L_of_G.edges)):
    print("L(G) edge:", edge, " Weight:", L_of_G.edges[edge]["weight"])

# more or less pasting in code
# from the Graph-Gaussian-Processes readme
laplacian = nx.laplacian_matrix(L_of_G)
eigenvalues, eigenvectors = tf.linalg.eigh(laplacian)  # only should be done once-per-graph
kernel = GraphMaternKernel((eigenvectors, eigenvalues))
model = gpflow.models.GPR(data=data, kernel=kernel)
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

