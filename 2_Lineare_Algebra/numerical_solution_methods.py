# -*- coding: utf-8 -*-
"""
@author: Benny Botsch
"""


from numpy import array, matrix, zeros, diag, diagflat, dot, matmul
from numpy.linalg import norm, inv

def jacobi(A,b,x,e):
    """Solves the equation Ax=b via the Jacobi iterative method.
    Parameters:
    -----------
        A : array
            coefficient matrix
        b : array
            the right-hand side of the system of equations
        x : array
            initial vector
        e : int
            error condition

    Returns:
    --------
        x : int
            solution
    """                                                                                                                                                                    
    D = diag(A)
    R = A - diagflat(D)
    error = norm(dot(A,x)-b)                                                                                                                                                                      
    while error>e:
        x = dot(inv(diagflat(D)),(b - dot(R,x)))
        error = norm(dot(A,x)-b)
    return x

def gaussSeidel(A,b,x,e):
    """
    Solves the equation Ax=b via the Gauss-Seidel method.
    Parameters:
    -----------
        A : array
            coefficient matrix
        b : array
            the right-hand side of the system of equations
        x : array
            initial vector
        e : int
            error condition
    Returns:
    --------
            x : array
                solution
    """
    error = norm(dot(A,x)-b)
    n=b.shape[0]
    xnew = zeros(n)
    while error>e:
        x  = xnew.copy()
        for i in range(0,n):
            xnew[i] = (b[i] - 
                       sum([A[i, j]*x[j] for j in range(i+1, n)]) - 
                       sum([A[i, j]*xnew[j] for j in range(i)]))/A[i, i]
        error = norm(dot(A,x)-b)
    return x

if __name__ == "__main__":
    A = array([
        [4, 1, 0], 
        [1, 4, 1],
        [0, 1, 4]])
    b = array([[1],[1],[1]])
    x0 = array([[1],[2],[3]])
    e = 1e-8
    x = jacobi(A,b,x0,e)
    x = gaussSeidel(A,b,x0,e)
    print(x)