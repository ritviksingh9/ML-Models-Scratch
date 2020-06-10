import numpy as np

def initialize_adam(parameters) :    
    L = len(parameters) // 2 
    v = {}
    s = {}

    for l in range(L):
        v["dW" + str(l+1)] = np.zeros(parameters["W"+str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters["b"+str(l+1)].shape)
        s["dW" + str(l+1)] = np.zeros(parameters["W"+str(l+1)].shape)
        s["db" + str(l+1)] = np.zeros(parameters["b"+str(l+1)].shape)

    return v, s

def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01, beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    L = len(parameters) // 2                 
    v_corrected = {}                         
    s_corrected = {}                         

    for l in range(L):
        v["dW" + str(l+1)] = beta1*v["dW"+str(l+1)] + (1-beta1) * grads["dW"+str(l+1)]
        v["db" + str(l+1)] = beta1*v["db"+str(l+1)] + (1-beta1) * grads["db"+str(l+1)]

        v_corrected["dW" + str(l+1)] = v["dW"+str(l+1)] / (1-beta1 ** t)
        v_corrected["db" + str(l+1)] = v["db"+str(l+1)] / (1-beta1 ** t)

        s["dW" + str(l+1)] = beta2*s["dW"+str(l+1)] + (1-beta2) * ((grads["dW"+str(l+1)]) ** 2)
        s["db" + str(l+1)] = beta2*s["db"+str(l+1)] + (1-beta2) * ((grads["db"+str(l+1)]) ** 2)

        s_corrected["dW" + str(l+1)] = s["dW"+str(l+1)] / (1-beta2 ** 2)
        s_corrected["db" + str(l+1)] = s["db"+str(l+1)] / (1-beta2 ** 2)

        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v_corrected["dW"+str(l+1)] / (s_corrected["dW"+str(l+1)]+epsilon) 
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v_corrected["db"+str(l+1)] / (s_corrected["db"+str(l+1)]+epsilon)  

    return parameters, v, s