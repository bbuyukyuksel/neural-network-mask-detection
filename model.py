
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os

import Parameters
from initialize_paramters import initialize_parameters
from propagation_forward import L_model_forward
from cost import compute_cost
from propagation_backward import L_model_backward, update_parameters


# GRADED FUNCTION: model
def model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False, save_parametrs=False, validation=None):#lr was 0.009

    np.random.seed(1)
    costs = []                         # keep track of cost
    test_accuracy = []
    train_accuracy = []

    # Parameters initialization.
    parameters = None
    iteration = None
    
    if os.path.exists("parameters/parameters.h5"):
        iteration, parameters = Parameters.load_last()
    else:
        if save_parametrs:
            Parameters.save(mode='w', iteration=None, parameters=None)
    if parameters == None:
        iteration = -1
        parameters = initialize_parameters(layers_dims)
    
    
    # Loop (gradient descent)
    for i in range(iteration+1, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)
        
        # Compute cost.
        cost = compute_cost(AL, Y)
    
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)
 
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # Print the cost every 100 training example
        msg_save = ''
        msg_acc = ''

        if print_cost and i % 100 == 0:
            
            if save_parametrs:
                Parameters.save(mode='a', iteration=i, parameters=parameters)
                msg_save = 'parameters saved'
            
            if validation:    
                _pred, _train_accuracy = predict(X, Y, parameters)
                _pred, _test_accuracy = predict(*validation, parameters)
                
                train_accuracy.append(_train_accuracy)
                test_accuracy.append(_test_accuracy)
                
                msg_acc = "Train Acc: {}, Test Acc:{}".format(_train_accuracy, _test_accuracy)

            print ("Cost after iteration %i: %f | %s | %s" %(i, cost, msg_acc, msg_save))

        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    plt.plot(train_accuracy, 'b')
    plt.plot(test_accuracy, 'r')
    plt.ylabel('Accuracy')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.legend(['train', 'test'])
    plt.show()
    
    return parameters

def predict(X, y, parameters):
    """    
    Arguments:
    X -- data set
    y -- labels
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    #print(f"predictions: {p}")
    #print(f"true labels: {y}")
    accuracy = round(np.sum((p == y)/m), 4)
    
    return p, accuracy
