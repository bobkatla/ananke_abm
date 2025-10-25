import numpy as np


def minutes_share(y_pred, P):
    N,L = y_pred.shape
    out = np.zeros(P, dtype=np.float64)
    for p in range(P):
        out[p] = np.mean((y_pred==p).sum(axis=1))/L
    return out

def tod_marginals(y_pred, P):
    N,L = y_pred.shape
    m = np.zeros((L,P), dtype=np.float64)
    for t in range(L):
        col = y_pred[:,t]
        for p in range(P):
            m[t,p] = np.mean(col==p)
    return m

def bigram_matrix(y_pred, P):
    N,L = y_pred.shape
    M = np.zeros((P,P), dtype=np.float64); Z=0.0
    for i in range(N):
        a = y_pred[i,:-1]; b = y_pred[i,1:]
        for u,v in zip(a,b):
            M[u,v]+=1; Z+=1
    if Z>0: M/=Z
    return M

def l1_distance(A,B): return float(np.abs(A-B).sum())
