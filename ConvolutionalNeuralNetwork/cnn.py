import numpy as np
import matplotlib.pyplot as pyplot
from optimize import *

def zero_pad(X, pad):
    return np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="constant", constant_values=(0, 0))

def conv_single_step(a_slice, W, b):
    return np.sum(np.multiply(a_slice, W)) + float(b)

def conv_forward(A, W, b, hyper_parameters):
    (m, n_H_prev, n_W_prev, n_C_prev) = A.shape
    (f, f, n_C_prev, n_C) = W.shape

    stride = hyper_parameters["stride"]
    pad = hyper_parameters["pad"]
    
    n_H = int((n_H_prev+2*pad-f) / stride + 1)
    n_W = int((n_W_prev+2*pad-f) / stride + 1)
    
    Z = np.zeros((m, n_H, n_W, n_C))
    
    A_pad = zero_pad(A, pad)
    
    for i in range(m):              
        a_pad = A_pad[i]               
        for h in range(n_H):           
            vert_start = h*stride
            vert_end = vert_start+f
            
            for w in range(n_W):       
                horiz_start = w*stride
                horiz_end = horiz_start+f
                
                for c in range(n_C):   
                    a_slice = a_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    
                    weights = W[:, :, :, c]
                    biases = b[:, :, :, c]
                    Z[i, h, w, c] = conv_single_step(a_slice, weights, biases)
    
    assert(Z.shape == (m, n_H, n_W, n_C))
    cache = (A, W, b, hyper_parameters)
    
    return Z, cache

def pool_forward(A, hyper_parameters, mode = "max"):
    (m, n_H_prev, n_W_prev, n_C_prev) = A.shape

    f = hyper_parameters["f"]
    stride = hyper_parameters["stride"]

    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev

    A_next = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):                          
        for h in range(n_H):                     
            vert_start = h*stride
            vert_end = vert_start+f
            
            for w in range(n_W):                 
                horiz_start = h*stride
                horiz_end = horiz_start+f
                
                for c in range (n_C):            
                    
                    a_slice = A[i, vert_start:vert_end, horiz_start:horiz_end, c]
                    
                    if mode == "max":
                        A_next[i, h, w, c] = np.max(a_slice)
                    elif mode == "average":
                        A_next[i, h, w, c] = np.average(a_slice)

    cache = (A, hyper_parameters)
    assert(A_next.shape == (m, n_H, n_W, n_C))
    
    return A_next, cache   

def conv_backward(dZ, cache):
    (A_prev, W, b, hparameters) = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    
    (m, n_H, n_W, n_C) = dZ.shape
    
    dA_prev = np.zeros(A_prev.shape)                           
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)

    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)
    
    for i in range(m): 
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        
        for h in range(n_H):                   
            for w in range(n_W):               
                for c in range(n_C):           
                    vert_start = h*stride
                    vert_end = vert_start+f
                    horiz_start = w*stride
                    horiz_end = horiz_start+f

                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                    dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                    db[:,:,:,c] += dZ[i, h, w, c]
                    
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]

    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
    
    return dA_prev, dW, db

def create_mask_from_window(x):
    return (x == np.max(x))

def distribute_value(dz, shape):
    (n_H, n_W) = shape
    average = dz / (n_H*n_W)
    return np.ones(shape)*average

def pool_backward(dA, cache, mode = "max"):
    (A_prev, hparameters) = cache

    stride = hparameters["stride"]
    f = hparameters["f"]
    
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape
    
    dA_prev = zeros(A_prev.shape)
    
    for i in range(m):                       
        a_prev = A_prev[i]
        
        for h in range(n_H):                   
            for w in range(n_W):               
                for c in range(n_C):           
                    vert_start = h*stride
                    vert_end = vert_start+f
                    horiz_start = w*stride
                    horiz_end = horiz_start+f
                    
                    if mode == "max":
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        mask = create_mask_from_window(a_prev_slice)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += a_prev_slice*mask
                        
                    elif mode == "average":
                        da = dA[i, vert_start: vert_end, horiz_start: horiz_end, c]
                        shape = (f, f)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da, shape)   

    assert(dA_prev.shape == A_prev.shape)
    
    return dA_prev
    
