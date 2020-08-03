import numpy as np


def danilevksy_method(A):
    
    '''
    A      :target matrix NxN dim.
    
    Returns last iter A for Danilevksy method for characteristic poly.
    
    '''
    
    if A.shape[0] != A.shape[1]:
        print("Matrix must be NxN! Error!")
        raise SystemExit(1)
    
    for _ in range(1,A.shape[1]):
        S = np.identity(A.shape[1])
        S[_,:] = A[:,_-1]
        S = S.transpose()
        A = np.dot(np.dot(np.linalg.inv(S),A),S)
        
    return A


def iter_method(A, B, eps = 0.01):

    '''
    
    A       :AX=B;
    B       :AX=B;
    eps     :accuracy.

    Iterative method realization.
    Returns count of iters, vector X.
    
    '''
    
    aii = np.diag(A)
    beta = (B/aii).reshape(A.shape[1],1)
    sup = np.ones((A.shape[0],A.shape[1]))
    np.fill_diagonal(sup,0)
    alpha = -A/(aii.reshape(A.shape[0],1))*sup
    X = beta
    print(beta)
    print(alpha)
    i = 0
    
    while True:
        check = 0
        if i == 10000:
            print(X)
            raise SystemExit(1)
        i += 1
        X_old = X
        X = beta + (np.dot(alpha,X))
        for _ in range(0,X.shape[0]):
            if abs(X[_][0] - X_old[_][0]) > eps:
                check = abs(X[_][0] - X_old[_][0])
        if check < eps:
            return i, X


def qr_decomp(A):

    R = np.array(a, copy=True, dtype=float)
    m, n = R.shape
    
    Q = np.identity(m)

    for i in range(n):
        vec = np.zeros(m)
        vec[i:] = R[i:, i]
        support_vec = np.zeros(vec.shape[0])
        support_vec[i] = LA.norm(vec)
        u = (vec - support_vec)/LA.norm(vec - support_vec)
        H = np.identity(m) - 2 * u.reshape(-1,1) * u
        print(H)

        Q = Q.dot(H)
        R = H.dot(R)
    
    return Q, R
