import torch
import numpy as np
def linear_to_triu_idx(n, lin_idx):
    """Linear index to upper triangular matrix without diagonal. This is
    similar to
    https://stackoverflow.com/questions/242711/algorithm-for-index-numbers-of-triangular-matrix-coefficients/28116498#28116498
    with number nodes decremented and col index incremented by one.
    """
    nn = n * (n - 1)
    row_idx = n - 2 - torch.floor(
        torch.sqrt(-8 * lin_idx.double() + 4 * nn - 7) / 2.0 - 0.5).long()
    col_idx = 1 + lin_idx + row_idx - nn // 2 + torch.div(
        (n - row_idx) * (n - row_idx - 1), 2, rounding_mode='floor')
    row_idx[lin_idx == -1] = -1
    col_idx[lin_idx == -1] = -1
    return row_idx.to(lin_idx.device), col_idx.to(lin_idx.device)

def triu_to_linear_idx(n, row_idx, col_idx):
    """Upper triangular matrix without diagonal to linear index. This is
    similar to
    https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix/27086432#27086432
    with number nodes decremented and col index incremented by one.
    """
    return n * (n - 1) // 2 - (n - row_idx) * (n - row_idx - 1) // 2 + col_idx - row_idx - 1

def triangular_matrix_index(M, rows, n_samples=None):
    # Calculate the start index of the row in the vector
    
    indices_all = []
    for row in (rows):
        start_index = (M * (M - 1)) // 2 - ((M - row) * (M - row - 1)) // 2
        
        # Elements in that row will be at start_index to start_index + (M - row - 1)
        if row < M - 1:
            if n_samples is None:
                indices = list(range(start_index, start_index + (M - row - 1)))
            else:
                indices = np.unique(np.random.randint(start_index, start_index + (M - row - 1), n_samples))
        else:
            indices = []
            
        indices_all.extend(indices)

        
    return indices_all

# def triangular_matrix_index(M, rows, n_samples=None):
    # Calculate the start index of the row in the vector
    
    # rows= np.array(rows)
    # rows = rows[rows < M-1]
    # start_indecis = (M * (M - 1)) // 2 - ((M - rows) * (M - rows - 1)) // 2
    # end_indecis = start_indecis + (M - rows - 1)
    # indices = np.unique((np.random.randint(start_indecis, end_indecis, (n_samples, len(end_indecis)) ).reshape(-1)))
    
    # # Elements in that row will be at start_index to start_index + (M - row - 1)
    # if row < M - 1:
    #     if n_samples is None:
    #         indices = list(range(start_index, start_index + (M - row - 1)))
    #     else:
    #         indices = np.random.randint(start_index, start_index + (M - row - 1), n_samples)
    # else:
    #     indices = []
        
        
    # return indices
def triangular_column_indices(M, col, n_samples=None):
    indices = []
    
    # Calculate indices for column `col`
    for i in range(col):
        # Start index of row `i` in the vector
        start_index = (M * (M - 1)) // 2 - ((M - i) * (M - i - 1)) // 2
        # Column `col` is (col - i - 1)th element in row `i`
        index = start_index + (col - i - 1)
        indices.append(index)
    indices = np.array(indices)
    if n_samples is not None:
        indices = np.random.choice(indices, n_samples, replace=False)
        
    
    return indices