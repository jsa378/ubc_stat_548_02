import numpy as np

# create adjacency matrix
adj_mat = np.zeros((7, 7))
adj_mat[0, :] = [0, 1, 1, 1, 0, 0, 1]
adj_mat[1, :] = [1, 0, 1, 0, 1, 0, 0]
adj_mat[2, :] = [1, 1, 0, 1, 0, 1, 0]
adj_mat[3, :] = [1, 0, 1, 0, 0, 0, 1]
adj_mat[4, :] = [0, 1, 0, 0, 0, 1, 0]
adj_mat[5, :] = [0, 0, 1, 0, 1, 0, 1]
adj_mat[6, :] = [1, 0, 0, 1, 0, 1, 0]
print("Adjacency matrix symmetric?", np.array_equal(adj_mat, np.transpose(adj_mat)))

# create weight matrix
weight_mat = np.zeros((7, 7))
weight_mat[0, :] = [0, 5, 4, 6, 0, 0, 7]
weight_mat[1, :] = [0, 0, 3, 0, 1, 0, 0]
weight_mat[2, :] = [0, 0, 0, 3, 0, 5, 0]
weight_mat[3, :] = [0, 0, 0, 0, 0, 0, 1]
weight_mat[4, :] = [0, 0, 0, 0, 0, 8, 0]
weight_mat[5, :] = [0, 0, 1, 0, 1, 0, 1]
weight_mat[6, :] = [1, 0, 0, 1, 0, 1, 0]

# create degree matrix
deg_mat = np.diag((4, 3, 4, 3, 2, 3, 3))

