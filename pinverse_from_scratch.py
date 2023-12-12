import numpy as np
import torch
#from numpy.linalg import norm
from random import normalvariate
from math import sqrt
 
def random_unit_vector(size):
    unnormalized = [normalvariate(0, 1) for _ in range(size)]
    norm = sqrt(sum(v * v for v in unnormalized))
    normalized = [v / norm for v in unnormalized]
    t = torch.tensor(normalized)
    return t
 
def power_iterate(X, iter_limit=100, epsilon=1e-10):    
    """ Recursively compute X^T X dot v to compute weights vector/eignevector """
 
    n, m = X.shape
    start_v = random_unit_vector(m) # start of random surf
    prev_eigenvector = None
    curr_eigenvector = start_v
    covariance_matrix = torch.matmul(X.T, X)
 
    ## power iterationn until converges
    it = 0        
    while True:
        it += 1
        prev_eigenvector = curr_eigenvector
        curr_eigenvector = torch.matmul(covariance_matrix, prev_eigenvector)
        curr_eigenvector = curr_eigenvector / torch.norm(curr_eigenvector)
 
        if torch.norm(curr_eigenvector - prev_eigenvector) < epsilon:
            print(it)
            return curr_eigenvector
# =============================================================================
#         if abs(torch.dot(curr_eigenvector, prev_eigenvector)) > 1 - epsilon:            
#             return curr_eigenvector
# =============================================================================
        if it == iter_limit:
            return curr_eigenvector

def svd(X, epsilon=1e-10):
    """after computed change of basis matrix from power iteration, compute distance"""
    n, m = X.shape
    change_of_basis = []
 
    for i in range(m):
        data_matrix = torch.clone(X)
 
        for sigma, u, v in change_of_basis[:i]:
            data_matrix -= sigma * torch.outer(u, v) 
 
        v = power_iterate(data_matrix, epsilon=epsilon) ## eigenvector 
        u_sigma = torch.matmul(X, v) ## 2nd step: XV = U Sigma 
        sigma = torch.norm(u_sigma)  
        u = u_sigma / sigma
 
        change_of_basis.append((sigma, u, v))
     
    sigmas = []
    us = []
    v_transposes = []
    for sv in change_of_basis:
        sigmas.append(sv[0].tolist())
        us.append(sv[1].tolist())
        v_transposes.append(sv[2].tolist())  
    sigmas = torch.tensor(sigmas)
    us = torch.tensor(us)
    v_transposes = torch.tensor(v_transposes)
    return sigmas, us.T, v_transposes

def moore_penrose_pseudo_inverse(U, singular_values, VT):
    tolerance = 1e-12  # Set a tolerance value for zero singular values
    singular_inv = []
    for s in singular_values:
        if s > tolerance:
            singular_inv.append(1/s)
        else:
            singular_inv.append(0)
    singular_inv = torch.diag(torch.tensor(singular_inv))
    return torch.matmul(np.matmul(VT.T, singular_inv), U.T)

A = torch.tensor([[1.5, 2],
              [3, 4],
              [5, 6]]).float()

singular_values, U, VT = svd(A)
# =============================================================================
# print("U:\n", U)
# print("Singular Values:", singular_values)
# print("V Transpose:\n", VT)
# =============================================================================

print('pinv' , moore_penrose_pseudo_inverse(U, singular_values, VT))
print('pinv built in', torch.pinverse(A))