import numpy as np
import cvxpy as cp
x = np.array([[-1,-1], [0,0],[1,0],[0,-1],[-1,1],[-2,1],[-1,2],[-2,0]]) # training samples
y = np.array([[-1],[-1],[-1],[-1],[1],[1],[1],[1]]) # training labels
m = len(y) # # of samples
d = x.shape[1] # dim of samples
a = cp.Variable(shape=(m,1),pos=True) # lagrange multiplier
C = 1 # trade-off parameter
G = np.matmul(y*x, (y*x).T) # Gram matrix
objective = cp.Maximize(cp.sum(a)-(1/2)*cp.quad_form(a, G))
constraints = [a <= C, cp.sum(cp.multiply(a,y)) == 0] # box constraint
prob = cp.Problem(objective, constraints)
result = prob.solve()
print(a.value)