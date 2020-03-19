import numpy as np


def build_collage(mat, n_col, n_row):
    rows = []
    for i in range(n_row):
        row = []
        for j in range(n_col):
            row.append(mat[i * n_col + j])
        row = np.concatenate(row, axis=1)
        rows.append(row)
    collage = np.concatenate(rows, axis=0)
    return collage
