import torch

class optimizer():
    def __init__(self, model, A, f, iterations, error):
        self.model = model
        self.A = A
        self.f = f
        self.x = torch.zeros(model.hidden_neurons, model.size_output)
        self.iterations = iterations
        self.error = error
    
    def jacobi(self):
        B = torch.diag(self.A)
        R = self.A - torch.diagflat(B)
        B_inv = torch.inverse(torch.diagflat(B))                                                                                                                                                                      
        for i in range(self.iterations):
            self.x = torch.matmul(B_inv, self.f - torch.matmul(R,self.x))
        return self.x
    
    def element_jacobi(self):
        A2 = torch.matmul(torch.transpose(self.A, 0, 1), self.A)
        f = torch.matmul(torch.transpose(self.A, 0, 1), self.f) 
        for it_count in range(self.iterations):
            x_new = torch.zeros_like(self.x)
            for i in range(A2.shape[0]):
                s1 = torch.matmul(A2[i, :i], self.x[:i])
                s2 = torch.matmul(A2[i, i + 1:], self.x[i + 1:])
                x_new[i] = (f[i] - s1 - s2) / A2[i, i]
            if torch.allclose(self.x, x_new, atol=self.error):
                break
            self.x = x_new
        return self.x

    def gaussSeidel(self): 
        A2 = torch.matmul(torch.transpose(self.A, 0, 1), self.A)
        f = torch.matmul(torch.transpose(self.A, 0, 1), self.f)
        D = torch.diag(A2)
        B = torch.tril(A2)
        R = torch.triu(A2) - torch.diagflat(D)
        B_inv = torch.inverse(B)  
        for i in range(self.iterations):
            self.x = torch.matmul(B_inv, f - torch.matmul(R, self.x))
        return self.x
    
    def element_gaussSeidel(self):
        A2 = torch.matmul(torch.transpose(self.A, 0, 1), self.A)
        f = torch.matmul(torch.transpose(self.A, 0, 1), self.f)  
        for it_count in range(self.iterations):
            x_new = torch.zeros_like(self.x)
            for i in range(A2.shape[0]):
                s1 = torch.matmul(A2[i, :i], x_new[:i])
                s2 = torch.matmul(A2[i, i + 1 :], self.x[i + 1 :])
                x_new[i] = (f[i] - s1 - s2) / A2[i, i]
            if torch.allclose(self.x, x_new, atol=self.error):
                break
            self.x = x_new
        return self.x
    
    def SOR(self):
        A2 = torch.matmul(torch.transpose(self.A, 0, 1), self.A)
        f = torch.matmul(torch.transpose(self.A, 0, 1), self.f) 
        omega = 0.5
        for it_count in range(self.iterations):
            for i in range(A2.shape[0]):
                sigma = 0
                for j in range(A2.shape[1]):
                    if j != i:
                        sigma += A2[i, j] * self.x[j]
                self.x[i] = (1 - omega) * self.x[i] + (omega / A2[i, i]) * (f[i] - sigma)
                if torch.allclose(self.x[i], self.x[i-1], atol=self.error):
                    break
            return self.x

    def pseudo_inv(self):
        B = torch.pinverse(self.A)
        self.x = torch.matmul(B, self.f)
        return self.x
    
    def pinv(self):
        import numpy as np
        from math import sqrt
        from random import normalvariate
        def random_unit_vector(size):
            unnormalized = [normalvariate(0, 1) for _ in range(size)]
            norm = sqrt(sum(v * v for v in unnormalized))
            normalized = [v / norm for v in unnormalized]
            t = torch.tensor(normalized)
            return t
         
        def power_iterate(X, iter_limit=500, epsilon=1e-12):        
            n, m = X.shape
            start_v = random_unit_vector(m) 
            prev_eigenvector = None
            curr_eigenvector = start_v
            covariance_matrix = torch.matmul(X.T, X)
            it = 0        
            while True:
                it += 1
                prev_eigenvector = curr_eigenvector
                curr_eigenvector = torch.matmul(covariance_matrix, prev_eigenvector)
                curr_eigenvector = curr_eigenvector / torch.norm(curr_eigenvector)
         
                if torch.allclose(curr_eigenvector, prev_eigenvector, atol=epsilon):
                    return curr_eigenvector
                if it == iter_limit:
                    return curr_eigenvector

        def svd(X, epsilon=1e-12):
            n, m = X.shape
            change_of_basis = []
         
            for i in range(m):
                data_matrix = torch.clone(X)
         
                for sigma, u, v in change_of_basis[:i]:
                    data_matrix -= sigma * torch.outer(u, v) 
         
                v = power_iterate(data_matrix, epsilon=epsilon)  
                u_sigma = torch.matmul(X, v)
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

        def moore_penrose_pseudo_inverse(singular_values, U, VT):
            tolerance = 1e-12 
            singular_inv = []
            for s in singular_values:
                if s > tolerance:
                    singular_inv.append(1/s)
                else:
                    singular_inv.append(0)
            singular_inv = torch.diag(torch.tensor(singular_inv))
            return torch.matmul(np.matmul(VT.T, singular_inv), U.T)

        singular_values, U, VT = svd(self.A)
        B = moore_penrose_pseudo_inverse(singular_values, U, VT)
        self.x = torch.matmul(B, self.f)
        return self.x