import numpy as np

# S - simple reduction, P - persistence-compatible reduction
TESTING_S = False
TESTING_P = False

def reduce_matrix_mod2(M):
    """
    Legacy matrix reduction algorithm, not used for persistence.
    """
    if type(M) == int:
        return 0, 0

    rows, cols = np.shape(M)
    column_index = dict.fromkeys(range(rows), None)
    new_M = M.copy()
    rank = 0
    
    for j in range(cols):
        while True:
            col = new_M[:, j]
            pivot = np.where(col == 1)[0]
            if pivot.size == 0:
                break
            pivot = pivot[0]

            if column_index[pivot] is not None:
                new_M[:, j] = (new_M[:, column_index[pivot]] + col) % 2
            else:
                rank += 1
                column_index[pivot] = j
                break
    return new_M, rank


def reduce_boundary_matrix(M):
    M = M.copy()
    n_cols = M.shape[1]

    def low(col):
        nz = np.where(col == 1)[0]
        return nz[-1] if len(nz) else None

    pivot_to_col = {}

    for j in range(n_cols):
        while True:
            lj = low(M[:, j])
            if lj is None:
                break

            if lj in pivot_to_col:
                i = pivot_to_col[lj]
                M[:, j] = (M[:, j] + M[:, i]) % 2
            else:
                pivot_to_col[lj] = j
                break
    
    return M


if __name__ == "__main__":
    if TESTING_S:
        M = np.array([[1,0], [1,0], [1,0], [1,0], [1,0]])
        print(reduce_matrix_mod2(M))

    if TESTING_P:
        bd_matrix = np.array([
            [0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0]
        ])
        print(reduce_boundary_matrix(bd_matrix))