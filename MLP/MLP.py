import __future__
import sys
import numpy as np
import math
import random
import pickle

inputLayerSize, hiddenLayerSize, outputLayerSize = 4, 4, 1

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def init_weights(rows,cols):
    return np.random.uniform(-1,1,(rows,cols))

def forward_flow(n,Xi,Yj,Dk,Wji,Wkj):
    Vj = np.dot(Wji,Xi[n,:]).reshape(4,1)
    Yj[1:,:] = sigmoid(Vj)
    Vk = np.dot(Wkj,Yj)
    Yk = sigmoid(Vk)
    err = (Dk[n] - Yk)
    return [err,Yj,Yk]

def back_flow_with_momentum(n,Xi,Yj,Yk,err,Wji,Wkj,learningRate,del_Wkj,del_Wji,alpha):
    delK = Yk * (1-Yk) * err
    del_Wkj = (learningRate * delK * Yj).transpose() + (alpha * del_Wkj)
    new_Wkj = Wkj + del_Wkj

    delJ = Yj * (1-Yj) * (Wkj.transpose()*delK)
    X = Xi[n,:].reshape((1,5))
    del_Wji = (learningRate * np.dot(delJ[1:,:],X)) + (alpha * del_Wji)
    new_Wji = Wji + del_Wji
    return [new_Wji,new_Wkj,del_Wkj,del_Wji]

def train_neural_net(Xi,Yj,Dk,Wji,Wkj,learningRate,del_Wkj,del_Wji,alpha):
    epoch = 0
    old_epoch = 0
    check = True
    while(check):
        check = False
        for i in range(0,16):
            [err,Yj,Yk] = forward_flow(i,Xi,Yj,Dk,Wji,Wkj)
            if(abs(err)>0.05):
                check = True
                [Wji,Wkj,del_Wkj,del_Wji] = back_flow_with_momentum(i,Xi,Yj,Yk,err,Wji,Wkj,learningRate,del_Wkj,del_Wji,alpha)

        epoch = epoch + 1

    print ("learningRate",learningRate,"momentum",alpha,"epoch",epoch)
    for i in range(0,16):
        [err,Yj,Yk] = forward_flow(i,Xi,Yj,Dk,Wji,Wkj)
        print (Xi[i,1:],Yk)
        print

if __name__ == "__main__":
    # Weights of the neural network
    Wji = init_weights(hiddenLayerSize,inputLayerSize+1)
    Wkj = init_weights(outputLayerSize,hiddenLayerSize+1)
    print("initial_weight_Wji",Wji)
    print("initial_weight_Wkj",Wkj)

    # Read the inputs
    Ip = np.genfromtxt('TrainInput.txt',delimiter = ',',dtype=np.float)
    # Nodes of the neural network
    Xi = Ip[:,:inputLayerSize+1]
    Yj = np.ones((hiddenLayerSize+1,1))
    # Expected Output
    Dk = Ip[:,inputLayerSize+1]

    del_Wkj= 0
    del_Wji= 0

    alpha = 0.9

    for l in range(1,11,1):
        learningRate = l * 0.05
        train_neural_net(Xi,Yj,Dk,Wji,Wkj,learningRate,del_Wkj,del_Wji,alpha)

    alpha = 0

    for l in range(1,11,1):
        learningRate = l * 0.05
        train_neural_net(Xi,Yj,Dk,Wji,Wkj,learningRate,del_Wkj,del_Wji,alpha)
