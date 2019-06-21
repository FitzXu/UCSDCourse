#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 12:54:49 2019
@author: qiminchen
"""

import numpy as np
from utils import *


def logistic_loss(W, X, Y):
    n = X.shape[0]
    yn = sigmoid(np.dot(X, W))
    loss = (-1.0/n) * np.sum(Y * np.log(yn+0.00000000001) + (1 - Y) * np.log(1 - yn+0.00000000001))  # calculate cost E
    return loss


def softmax_loss(W, X, Y):
    # W: class x features
    n = X.shape[0]
    an = np.exp(np.dot(X, W))
    yn = (an)/np.expand_dims(np.sum(an, axis=1),axis=1)
    return -(1.0/(n*Y.shape[1]))*np.sum(Y * np.log(yn+0.00000000001))


def logistic_BGD(X, Y, pid, epochs, dims_rescaled_data, learning_rate=0.1):
    # initialize weight to 0
    acc = []
    total_train_loss = []
    total_val_loss = []
    for t in range(10):
        W = np.random.randn(dims_rescaled_data+1, 1)
        best_W = np.zeros((dims_rescaled_data+1, 1))
        train_losses = []
        val_losses = []
        tmp_loss = float('inf')
        train_pca, train_Y, val_pca, val_Y, test_pca, test_Y ,eigenvec= split_data(X, Y, pid, t, dims_rescaled_data)

        train_pca = np.hstack((train_pca, np.ones((train_pca.shape[0], 1))))
        val_pca = np.hstack((val_pca, np.ones((val_pca.shape[0], 1))))
        test_pca = np.hstack((test_pca, np.ones((test_pca.shape[0], 1))))

        for epo in range(epochs):
            Yn = sigmoid(np.dot(train_pca, W))
            dW = np.dot(train_pca.T, train_Y - Yn)
            W = W + learning_rate * dW
            train_loss = logistic_loss(W, train_pca, train_Y)
            val_loss = logistic_loss(W, val_pca, val_Y)
            if val_loss < tmp_loss:
                best_W = W
                tmp_loss = val_loss
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print("Time:{}, Epoch:{}, train loss:{}, val_loss:{}".format(t, epo, train_loss, val_loss))
        total_train_loss.append(train_losses)
        total_val_loss.append(val_losses)
        one_acc = calculate_accuracy_testset(test_pca, test_Y, best_W, mode="binary")
        acc.append(one_acc)
        print("Accuracy:{}".format(one_acc))
    print("Average Accuracy:{}".format(np.array(acc).mean()))
    print("Standard Deviation:{}".format(np.std(np.array(acc))))
    return np.array(total_train_loss), np.array(total_val_loss), np.array(acc)



def logistic_SGD(X, Y, pid, epochs, dims_rescaled_data, learning_rate=0.1):

    acc = []
    total_train_loss = []
    total_val_loss = []
    for t in range(10):
        W = np.random.randn(dims_rescaled_data+1, 1)
        best_W = np.zeros((dims_rescaled_data+1, 1))
        train_losses = []
        val_losses = []
        tmp_loss = float('inf')
        train_pca, train_Y, val_pca, val_Y, test_pca, test_Y ,eigenvec= split_data(X, Y, pid, t, dims_rescaled_data)
        train_pca = np.hstack((train_pca, np.ones((train_pca.shape[0], 1))))
        val_pca = np.hstack((val_pca, np.ones((val_pca.shape[0], 1))))
        test_pca = np.hstack((test_pca, np.ones((test_pca.shape[0], 1))))

        total = train_pca.shape[0]
        
        for epo in range(epochs):
            
            # randomize the order of the indices into the training set
            permutation = list(np.random.permutation(total))
            train_pca = train_pca[permutation, :]
            train_Y = train_Y[permutation, :]
            
            for i in range(total):
                Yi = sigmoid(np.dot(train_pca[i,:], W))
                dW = np.dot(np.expand_dims(train_pca[i],axis=0).T, np.expand_dims(train_Y[i] - Yi, axis=0))
                W += learning_rate * dW

            train_loss = logistic_loss(W, train_pca, train_Y)
            val_loss = logistic_loss(W, val_pca, val_Y)

            if val_loss < tmp_loss:
                best_W = W
                tmp_loss = val_loss

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print("Time:{}, Epoch:{}, train loss:{}, val_loss:{}".format(t, epo, train_loss, val_loss))
        total_train_loss.append(train_losses)
        total_val_loss.append(val_losses)
        one_acc = calculate_accuracy_testset(test_pca, test_Y, best_W, mode="binary")
        acc.append(one_acc)
        print("Accuracy:{}".format(one_acc))
    print("Average Accuracy:{}".format(np.array(acc).mean()))
    print("Standard Deviation:{}".format(np.std(np.array(acc))))
    return np.array(total_train_loss), np.array(total_val_loss), np.array(acc)


def softmax_BGD(X, Y, pid, epochs, dims_rescaled_data, learning_rate=0.002):
    """
    :param X:
    :param Y:
    :param pid:
    :param epochs:
    :param dims_rescaled_data:
    :param learning_rate:
    :return:
    """
    # initialize weight to 0
    acc = []
    total_train_loss = []
    total_val_loss = []
    confusionM = np.zeros((6, 6))

    for t in range(10):
        W = np.random.randn(dims_rescaled_data, 6)
        best_W = np.zeros((dims_rescaled_data, 6))
        train_losses = []
        val_losses = []
        tmp_loss = float('inf')
        train_pca, train_Y, val_pca, val_Y, test_pca, test_Y ,eigenvec= split_data(X, Y, pid, t, dims_rescaled_data)
        #train_pca = np.hstack((train_pca, np.ones((train_pca.shape[0], 1))))
        #val_pca = np.hstack((val_pca, np.ones((val_pca.shape[0], 1))))
        #test_pca = np.hstack((test_pca, np.ones((test_pca.shape[0], 1))))
        for epo in range(epochs):
            an = np.exp(np.dot(train_pca, W))
            Yn = (an)/np.expand_dims(np.sum(an, axis=1),axis=1)
            dW = np.dot(train_pca.T, train_Y - Yn)
            W += learning_rate * dW
            
            train_loss = softmax_loss(W, train_pca, train_Y)
            val_loss = softmax_loss(W, val_pca, val_Y)
            
            if val_loss < tmp_loss:
                best_W = W
                tmp_loss = val_loss
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print("Time:{}, Epoch:{}, train loss:{}, val_loss:{}".format(t, epo, train_loss, val_loss))
        total_train_loss.append(train_losses)
        total_val_loss.append(val_losses)
        one_acc = calculate_accuracy_testset(test_pca, test_Y, best_W, mode="multiple")
        confusionM += confusion(test_pca, test_Y, best_W)
        acc.append(one_acc)
        print("Accuracy:{}".format(one_acc))
        if t == 8:
           visual_weights(best_W,eigenvec)
    print("Average Accuracy:{}".format(np.array(acc).mean()))
    print("Standard Deviation:{}".format(np.std(np.array(acc))))
    plot_confusion_matrix(confusionM)
    return np.array(total_train_loss), np.array(total_val_loss), np.array(acc)


def softmax_SGD(X, Y, pid, epochs, dims_rescaled_data, learning_rate=0.002):
    
    acc = []
    total_train_loss = []
    total_val_loss = []
    confusionM = np.zeros((6, 6))
    for t in range(10):

        W = np.random.randn(dims_rescaled_data, 6)
        best_W = np.zeros((dims_rescaled_data, 6))
        train_losses = []
        val_losses = []
        tmp_loss = float('inf')
        train_pca, train_Y, val_pca, val_Y, test_pca, test_Y ,eigenvec = split_data(X, Y, pid, t, dims_rescaled_data)
        total = train_pca.shape[0]
        for epo in range(epochs):
            # randomize the order of the indices into the training set
            permutation = list(np.random.permutation(total))
            train_pca = train_pca[permutation, :]
            train_Y = train_Y[permutation, :]
            for i in range(total):
                an = np.exp(np.dot(train_pca[i], W))
                Yn = (an)/np.sum(an)
                dW = np.dot(np.expand_dims(train_pca[i],axis=0).T, np.expand_dims(train_Y[i] - Yn,axis=0))
                W += learning_rate * dW
                
            train_loss = softmax_loss(W, train_pca, train_Y)
            val_loss = softmax_loss(W, val_pca, val_Y)
            if val_loss < tmp_loss:
                best_W = W
                tmp_loss = val_loss
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print("Time:{}, Epoch:{}, train loss:{}, val_loss:{}".format(t, epo, train_loss, val_loss))
        
        total_train_loss.append(train_losses)
        total_val_loss.append(val_losses)
        one_acc = calculate_accuracy_testset(test_pca, test_Y, best_W, mode="multiple")
        confusionM += confusion(test_pca, test_Y, best_W)
        acc.append(one_acc)
        print("Accuracy:{}".format(one_acc))
        if t == 9:
            plot_confusion_matrix(confusionM)
    print("Average Accuracy:{}".format(np.array(acc).mean()))
    print("Standard Deviation:{}".format(np.std(np.array(acc))))
    plot_confusion_matrix(confusionM)
    return np.array(total_train_loss), np.array(total_val_loss), np.array(acc)


def identity_classification(X, Y, pid, epochs, dims_rescaled_data, learning_rate=0.002):
    """
    :param X:
    :param Y:
    :param pid:
    :param epochs:
    :param dims_rescaled_data:
    :param learning_rate:
    :return:
    """
    # initialize weight to 0
    acc = []
    total_train_loss = []
    total_val_loss = []
    confusionM = np.zeros((10, 10))
    for t in range(6):
        W = np.random.randn(dims_rescaled_data, 10)
        best_W = np.zeros((dims_rescaled_data, 10))
        train_losses = []
        val_losses = []
        tmp_loss = float('inf')
        train_pca, train_Y, val_pca, val_Y, test_pca, test_Y, eigenvec = split_for_identity(X, Y, pid, t, dims_rescaled_data)
        #print()
        for epo in range(epochs):
            an = np.exp(np.dot(train_pca, W))
            Yn = (an) / np.expand_dims(np.sum(an, axis=1), axis=1)
            dW = np.dot(train_pca.T, train_Y - Yn)
            W += learning_rate * dW

            train_loss = softmax_loss(W, train_pca, train_Y)
            val_loss = softmax_loss(W, val_pca, val_Y)

            if val_loss < tmp_loss:
                best_W = W
                tmp_loss = val_loss
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print("Time:{}, Epoch:{}, train loss:{}, val_loss:{}".format(t, epo, train_loss, val_loss))
        total_train_loss.append(train_losses)
        total_val_loss.append(val_losses)
        one_acc = calculate_accuracy_testset(test_pca, test_Y, best_W, mode="multiple")
        confusionM += confusion(test_pca, test_Y, best_W)
        acc.append(one_acc)
        if t == 4:
            visual_weights(best_W,eigenvec)
        print("Accuracy:{}".format(one_acc))
    print("Average Accuracy:{}".format(np.array(acc).mean()))
    print("Standard Deviation:{}".format(np.std(np.array(acc))))
    plot_confusion_matrix(confusionM)
    return np.array(total_train_loss), np.array(total_val_loss), np.array(acc)