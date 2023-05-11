import torch
import numpy as np
import random
from matplotlib import pyplot as plt
from preprocess import hospital_loader
from model import NN, train, test, random_forest
from torch import nn, optim

def graph(epoch, y, xlab, ylab):
    x = []
    for i in range(epoch):
        x.append(i + 1)
    plt.plot(x, y)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.show()


def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    neural_network = True # Set to True to use neural network, false for Random Forest
    one_hot = True # Set to True for one hot encoding, false for ordinal encoding
    expand_range = False # Set to True for expanded correct prediction range, False for standard 10 day range

    if(one_hot): # if one hot features is 51
        model = NN(input_features = 51)
    else: # if ordinal features is 16
        model = NN(input_features= 16)

    batch_size = 32
    epochs = 10
    learning_rate = 0.2
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), learning_rate)


    if(neural_network):
        dataloader_train, dataloader_test = hospital_loader(batch_size = batch_size, method = neural_network, one_hot=one_hot)
        epoch_correct, epoch_loss = train(model, dataloader_train, loss_func, optimizer, epochs, expand_range)
        #loss_train, accuracy_train = test(model, dataloader_train, loss_func)
        loss_test, accuracy_test = test(model, dataloader_test, loss_func, expand_range)
        #print('Average Training Loss: {:.4f} | Average Training Accuracy: {:.4f}%'.format(loss_train, accuracy_train*100))
        print('Average Testing Loss: {:.4f} | Average Testing Accuracy: {:.4f}%'.format(loss_test, accuracy_test*100))

        
        graph(epochs, epoch_correct, "Epoch", "Accuracy")
        graph(epochs, epoch_loss, "Epoch", "Loss")
    else:
        X_tr, X_te, Y_tr, Y_te = hospital_loader(batch_size=batch_size, method = neural_network, one_hot=one_hot)
        random_forest(X_tr, X_te, Y_tr, Y_te)


if __name__ == "__main__":
    main()

