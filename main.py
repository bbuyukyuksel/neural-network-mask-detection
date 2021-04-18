import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os

from collections import namedtuple

import utils
import model
import Parameters


np.random.seed(1)

if __name__ == '__main__':
    # Train
    # Predict
    Make = "Train"

    #Prepare Dataset
    if(not os.path.exists("./tdataset/prepared")):
        utils.prepare_dataset('tdataset', l_train=60, l_dev=20, l_test=20)

    train_x_orig, train_y = utils.load("tdataset/prepared/train")
    test_x_orig , test_y  = utils.load("tdataset/prepared/dev")
    
    classes = [
        "without_mask",
        "with_mask",
    ]

    # Example of a picture
    num_images = 10
    for index in range(0, num_images):
        plt.subplot(2, (num_images//2), index + 1)
        plt.imshow(train_x_orig[index, :], interpolation='nearest')
        plt.title(f'{classes[train_y[0, index]]}')
        plt.axis('off')
    plt.show()


    # Shapes
    m_train = train_x_orig.shape[0] # Train Dataset Size
    num_px = train_x_orig.shape[1] # Feaure Size
    m_test = test_x_orig.shape[0] # Test Dataset Size

    print ("Number of training examples: " + str(m_train))
    print ("Number of testing examples: " + str(m_test))
    print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print ("train_x_orig shape: " + str(train_x_orig.shape))
    print ("train_y shape: " + str(train_y.shape))
    print ("test_x_orig shape: " + str(test_x_orig.shape))
    print ("test_y shape: " + str(test_y.shape))

    # Reshape the training and test examples 
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.

    print ("train_x's shape: " + str(train_x.shape))
    print ("test_x's shape: " + str(test_x.shape))

    # Model
    if Make.lower() == "train":
        layers_dims = [12288, 20, 7, 1] #  3-layer model
        parameters = model.model(train_x, train_y, layers_dims, learning_rate=0.01, num_iterations = 40000, print_cost = True, save_parametrs=True, validation=(test_x, test_y))

    elif Make.lower() == "predict":
        _, parameters = Parameters.load_last()
        pred_train = model.predict(train_x, train_y, parameters)
        pred_test = model.predict(test_x, test_y, parameters)

        print("Pred Train Acc", pred_train[1]) # pred_train[0] contains predictions
        print("Pred Test  Acc", pred_test[1]) #pred_test[0] contains predictions

        test_x_orig , test_y  = utils.load("tdataset/prepared/test")
        # Reshape the training and test examples 
        test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
        # Standardize data to have feature values between 0 and 1.
        test_x = test_x_flatten/255.
        pred_test = model.predict(test_x, test_y, parameters)
        print("Pred Dev   Acc", pred_test[1]) #pred_test[0] contains predictions




        





    