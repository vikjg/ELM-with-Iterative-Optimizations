# Linear solving methods .py file
# All methods approximate solutions, x, of the equation Ax = f
import torch

class optimizer():
    def __init__(self, model, A, f, iterations):
        self.model = model
        self.A = A
        self.f = f
        self.x = torch.zeros(model.hidden_neurons, model.size_output)
        self.iterations = iterations
    
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
            if torch.allclose(self.x, x_new, atol=1e-10, rtol=0.):
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
            if torch.allclose(self.x, x_new, rtol=1e-8):
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
                if torch.allclose(self.x[i], self.x[i-1], rtol=1e-8):
                    break
            return self.x

    def pseudo_inv(self):
        B = torch.pinverse(self.A)
        self.x = torch.matmul(B, self.f)
        return self.x
