#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Function: 
# Author: Yijia Li

# v.0.0.27
import numpy as np
from scipy import sparse as sp
from scipy import linalg as lg

def DN(w, method = 'ave'):  # 没有问题
    #w = w * w.shape[0]
    D = sp.dia_matrix(w.shape, dtype='d')
    if method == 'ave':
        D.setdiag(np.array(1/(np.absolute(w).sum(axis=1))[:,0])[:,0])
    elif method == 'gph':
        D.setdiag(np.array(1/(np.sqrt(np.absolute(w).sum(axis=1)))[:,0])[:,0])
    else:
        raise NameError("Method not supproted: either 'ave' or 'gph'")
    wn = D * w
    return wn

def TF(w): # 没问题
    #w = w * w.shape[0]
    w = DN(w)
    wt = np.sqrt(np.absolute(w).sum(axis=0))
    w = w/np.repeat(wt, repeats=w.shape[0], axis=0)
    w = w.dot(w.transpose())
    return w

def TF(w): # 没问题
    if sp.issparse(w):
        w = sp.csr_matrix(w)
        w = DN(w)
        wt = np.sqrt(np.absolute(w).sum(axis=0))
        w_iter = w.tocoo()
        for i,j,v in zip(w_iter.row, w_iter.col, w_iter.data):
            w[i,j] = v/wt[0,j]
        w = w.dot(w.transpose())
        #w.setdiag(0)
    else:
        w = DN(w)
        wt = np.sqrt(np.absolute(w).sum(axis=0))
        w = w/np.repeat(wt, repeats=w.shape[0], axis=0)
        w = w.dot(w.transpose())
        #np.fill_diagonal(w, 0)
    return w

def LocalAffinity(W_in):
    W = DN(W_in)
    W = (W+W.transpose())/2
    WW = sp.dia_matrix(W.shape, dtype='d')
    WW.setdiag(np.array(np.absolute(W).sum(axis=1))[:,0])
    W = W + (sp.eye(W.shape[0], dtype='d') + WW)
    W = TF(W)

def NetEnhance(W_in, order = 2, alpha = 0.9, k = 30): # all check
    k = min(k*2, W_in.shape[0]//10+1)
    W = DN(W_in)
    W = (W+W.transpose())/2
    DD = np.squeeze(np.asarray(np.absolute(W).sum(axis=0)))
    P = W
    PP = sp.dia_matrix(P.shape, dtype='d')
    PP.setdiag(np.array(np.absolute(W).sum(axis=1))[:,0])
    P = P + (sp.eye(P.shape[0], dtype='d') + PP)
    P = TF(P)
    D, V = lg.eig(P) #https://stackoverflow.com/questions/51247998/numpy-equivalents-of-eig-and-eigs-functions-of-matlab
    d = sp.dia_matrix(P.shape, dtype = 'd')
    d.setdiag(D.real)
    d = (1- alpha)*d /np.subtract(1, alpha*np.power(d, order).toarray())
    D = d.real
    W = V.dot(D.dot(V.transpose()))
    diag = W.diagonal().copy()
    np.fill_diagonal(W, 0)
    W = W/np.repeat((1- diag.transpose()),  W.shape[0], axis=1)
    D = sp.dia_matrix((len(DD),len(DD)), dtype='d')
    D.setdiag(DD)
    W = D.dot(W)
    W[W < 0] = 0
    W = (W+W.transpose())/2
    return W

def NetEnhance(W_in, order = 2, alpha = 0.9, k = 30): # all check
    k = min(k*2, W_in.shape[0]//10+1)
    W = DN(W_in)
    W = (W+W.transpose())/2
    # W is 
    DD = np.squeeze(np.asarray(np.absolute(W).sum(axis=0)))
    P = W
    PP = sp.dia_matrix(P.shape, dtype='d')
    PP.setdiag(np.array(np.absolute(W).sum(axis=1))[:,0])
    P = P + (sp.eye(P.shape[0], dtype='d') + PP)
    P = TF(P)
"""    if 1-np.count_nonzero(P)/np.product(P.shape) > 0.5: # sparse matrix
        P = sp.csc_matrix(P)
        D, V = sp.linalg.eigs(P, k=k)
    else:"""
    D, V = lg.eig(P) #https://stackoverflow.com/questions/51247998/numpy-equivalents-of-eig-and-eigs-functions-of-matlab
    d = sp.dia_matrix(P.shape, dtype = 'd')
    d.setdiag(D.real)
    d = (1- alpha)*d /np.subtract(1, alpha*np.power(d, order).toarray())
    D = d.real
    W = V.dot(D.dot(V.transpose()))
    diag = W.diagonal().copy()
    np.fill_diagonal(W, 0)
    W = W/np.repeat((1- diag.transpose()),  W.shape[0], axis=1)
    D = sp.dia_matrix((len(DD),len(DD)), dtype='d')
    D.setdiag(DD)
    W = D.dot(W)
    W[W < 0] = 0
    W = (W+W.transpose())/2
    return W