import torch

"""
Code adapted to tensors from:
https://github.com/scipy/scipy/blob/v1.2.0/scipy/spatial/kdtree.py#L936-L987
"""
    
def minkowski_distance_p(x, y, p=2):
    """
    Compute the p-th power of the L**p distance between two arrays.
    For efficiency, this function computes the L**p distance but does
    not extract the pth root. If `p` is 1 or infinity, this is equal to
    the actual L**p distance.
    Parameters
    ----------
    x : (M, K) tensor
        Input array.
    y : (N, K) tensor
        Input array.
    p : float, 1 <= p <= infinity
        Which Minkowski p-norm to use.
    Examples
    --------
    >>> x=torch.tensor([[0,0],[0,0]], dtype=torch.double)
    >>> y=torch.tensor([[1,1],[0,1]], dtype=torch.double)
    >>> print(minkowski_distance_p(x, y))
    tensor([ 2.,  1.], dtype=torch.float64)
    """
    if p == float("Inf"):
        return torch.amax(torch.abs(y-x), dim=-1)
    elif p == 1:
        return torch.sum(torch.abs(y-x), dim=-1)
    else:
        return torch.sum(torch.abs(y-x)**p, dim=-1)
    
# if __name__=='__main__':
#     x=torch.tensor([[0,0],[0,0]], dtype=torch.double)
#     y=torch.tensor([[1,1],[0,1]], dtype=torch.double)
#     print(minkowski_distance_p(x, y)) # tensor([ 2,  1])
#     print(minkowski_distance_p(x, y).dtype) #tensor.float64
    
#     x=torch.tensor([[0,0],[0,0]], dtype=torch.float)
#     y=torch.tensor([[1,1],[0,1]], dtype=torch.float)
#     print(minkowski_distance_p(x, y).dtype) # tensor.float32

def minkowski_distance(x, y, p=2, dtype=1):
    """
    Compute the L**p distance between two arrays.
    Parameters
    ----------
    x : (M, K) tensor
        Input array.
    y : (N, K) tensor
        Input array.
    p : float, 1 <= p <= infinity
        Which Minkowski p-norm to use.
        
    If args.dtype==0, the output is float.
    If args.dtype==0, the output is double.
    
    Examples
    --------
    >>> x=torch.tensor([[0,0],[0,0]], dtype=torch.double)
    >>> y=torch.tensor([[1,1],[0,1]], dtype=torch.double)
    >>> print(minkowski_distance(x, y))
    tensor([ 1.4142,  1.0000], dtype=torch.float64)
    """
    if p == float("Inf") or p == 1:
        if dtype==1: return (minkowski_distance_p(x, y, p)).double()
        else: return (minkowski_distance_p(x, y, p)).float()
    else:
        if dtype==1: return (minkowski_distance_p(x, y, p)**(1./p)).double()
        else: return (minkowski_distance_p(x, y, p)**(1./p)).float()

# if __name__=='__main__':
#     x=torch.tensor([[0,0],[0,0]], dtype=torch.double)
#     y=torch.tensor([[1,1],[0,1]], dtype=torch.double)
#     print(minkowski_distance(x, y, dtype=1))
#     print(minkowski_distance(x, y, dtype=0))

    
def distance_matrix(x, y, p=2, threshold=1000000, dtype=1):
    """
    Compute the distance matrix.
    Returns the matrix of all pair-wise distances.
    Parameters
    ----------
    x : (M, K) tensor
        Matrix of M vectors in K dimensions.
    y : (N, K) tensor
        Matrix of N vectors in K dimensions.
    p : float, 1 <= p <= infinity
        Which Minkowski p-norm to use.
    threshold : positive int
        If ``M * N * K`` > `threshold`, algorithm uses a Python loop instead
        of large temporary arrays.
    Returns
    -------
    result : (M, N) tensor
        Matrix containing the distance from every vector in `x` to every vector
        in `y`.
        If args.dtype==0, the output is float.
        If args.dtype==0, the output is double.
    Examples
    --------
    >>> x=torch.tensor([[0,0],[0,1]], dtype=torch.double)
    >>> y=torch.tensor([[1,0],[1,1]], dtype=torch.double)
    >>> print(distance_matrix(x, y))
    tensor([[ 1.0000,  1.4142],
            [ 1.4142,  1.0000]], dtype=torch.float64)
    """

    m, k = x.shape
    n, kk = y.shape

    if k != kk:
        raise ValueError("x contains %d-dimensional vectors but y contains %d-dimensional vectors" % (k, kk))

    if m*n*k <= threshold:
#         import numpy as np
#         return minkowski_distance(x[:,np.newaxis,:],y[np.newaxis,:,:],p)
        return minkowski_distance(x.unsqueeze(1),y.unsqueeze(0),p, dtype)
    else:
        if dtype==1: result = torch.empty((m,n),dtype=torch.double)  
        else: result = torch.empty((m,n),dtype=torch.float)  
        if m < n:
            for i in range(m):
                result[i,:] = minkowski_distance(x[i],y,p, dtype)
        else:
            for j in range(n):
                result[:,j] = minkowski_distance(x,y[j],p, dtype)
        return result
    
# if __name__=='__main__':
#     x=torch.tensor([[0,0],[0,0]], dtype=torch.double)
#     y=torch.tensor([[1,1],[0,1]], dtype=torch.double)
#     print(distance_matrix(x, y))
#     print(distance_matrix(x, y, dtype=0))
    
#     x=torch.tensor([[0,0],[0,1]], dtype=torch.double)
#     y=torch.tensor([[1,0],[1,1]], dtype=torch.double)
#     print(distance_matrix(x, y))
#     print(distance_matrix(x, y, dtype=0))
    
# #     import numpy as np
# #     print(x)
# #     print(x[:,np.newaxis,:])
# #     print(y)
# #     print(y[np.newaxis,:,:])
# #         return minkowski_distance(x.unsqueeze(1),y.unsqueeze(0),p)