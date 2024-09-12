import numpy as np

def gaussian_weight(distance, bandwidth):
    # Gaussian weight function
    return np.exp(-0.5 *(distance/bandwidth)**2)

def gloess(x,y, bandwidth,n=2):
    m = n+1
    # Apply GLOESS smoothing
    y_smoothed = np.zeros_like(y)

    for i in range(len(x)):
        weights = gaussian_weight(x - x[i], bandwidth)
        
        A = np.zeros((m, m))
        
        # Populates the A matrix with Aij = w*x^(i+j) where i,j -> [0,m]
        for j in range(m):
            for k in range(m):
                A[j, k] = np.nansum(weights*x**(j+k))
        
        X = []
        
        # Xi = w*x^i*y where i-> [0,m]
        for j in range(m):
            X.append(np.nansum(weights*(x**j)* y))
                     
        X = np.array(X)
        
        coeffs = np.linalg.solve(A,X)
        
        # y = a + bx + cx^2 + ... + zx^n 
        for j, c in enumerate(coeffs):
            y_smoothed[i] += c*x[i]**j

    return y_smoothed