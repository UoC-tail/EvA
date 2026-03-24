import scipy.sparse as sp

from torch_sparse import SparseTensor
import torch
import numpy as np
import hashlib
import copy


def sparse_tensor(spmat: sp.spmatrix, grad: bool = False):
    """

    Convert a scipy.sparse matrix to a SparseTensor.
    Parameters
    ----------
    spmat: sp.spmatrix
        The input (sparse) matrix.
    grad: bool
        Whether the resulting tensor should have "requires_grad".
    Returns
    -------
    sparse_tensor: SparseTensor
        The output sparse tensor.
    """
    if str(spmat.dtype) == "float32":
        dtype = torch.float32
    elif str(spmat.dtype) == "float64":
        dtype = torch.float64
    elif str(spmat.dtype) == "int32":
        dtype = torch.int32
    elif str(spmat.dtype) == "int64":
        dtype = torch.int64
    elif str(spmat.dtype) == "bool":
        dtype = torch.uint8
    else:
        dtype = torch.float32

    result = torch.sparse_coo_tensor(
        indices=torch.tensor([spmat.row, spmat.col]), 
        values=torch.tensor(spmat.data).to(dtype), size=torch.Size(spmat.shape)).coalesce()
    
    return result

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.random.manual_seed(seed)


def hash_numpy_array(array):
    # Ensure the array is in a consistent state (e.g., no uninitialized memory)
    if isinstance(array, torch.Tensor):
        array_v = array.cpu().numpy()
    else:
        array_v = copy.deepcopy(array)
    array_v = np.ascontiguousarray(array_v)
    
    # Convert the array to a bytes string
    array_bytes = array.view(np.uint8).tobytes()
    
    # Create a hash object
    hash_object = hashlib.sha256()
    
    # Update the hash object with the array's bytes
    hash_object.update(array_bytes)
    
    # Return the hexadecimal digest
    return hash_object.hexdigest()