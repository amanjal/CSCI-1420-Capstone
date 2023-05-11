import torch
import numpy as np
from torch import nn 
from sklearn.ensemble import RandomForestRegressor

class NN(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.h1 = 64
        self.h2 = 128
        self.h3 = 32
        class_num = 12

        self.w1 = torch.nn.Linear(input_features, self.h1)
        self.w2 = torch.nn.Linear(self.h1, self.h2)
        self.w3 = torch.nn.Linear(self.h2, self.h2)
        self.w4 = torch.nn.Linear(self.h2, self.h1)
        self.w5 = torch.nn.Linear(self.h1, self.h3)
        self.w6 = torch.nn.Linear(self.h3, class_num)

        self.RELU1 = torch.nn.ReLU()
        self.RELU2 = torch.nn.ReLU()
        self.RELU3 = torch.nn.ReLU()
        self.RELU4 = torch.nn.ReLU()
        self.RELU5 = torch.nn.ReLU()


    def forward(self, X):
        x = self.RELU1(self.w1(X))
        x = self.RELU2(self.w2(x))
        x = self.RELU3(self.w3(x))
        x = self.RELU4(self.w4(x))
        x = self.RELU5(self.w5(x))

        return self.w6(x)
    
# Method to train passed in model
def train(model, dataloader_train, loss_func, optimizer, epochs, expand_range):

    model.train()
    epoch_loss_sum_list = []
    epoch_correct_num_list = []

    # Iterate over epochs
    for e in range(epochs):
        epoch_loss_sum = 0
        epoch_correct_num = 0
        print(e)

        # Iterate over batches in dataloader
        for X, Y in dataloader_train:

            output = model.forward(X)
            optimizer.zero_grad()
            loss = loss_func(output, Y)
            loss.backward()
            optimizer.step()
            
            calculated_loss = loss.item() # Total loss of the current batch
            epoch_loss_sum += calculated_loss # total loss of epoch


            correct_predictions = predict(output, Y, expand_range) # get correct number predictions
            epoch_correct_num += correct_predictions

        epoch_correct_num /= len(dataloader_train.dataset)
        epoch_correct_num_list.append(epoch_correct_num)
        print(len(epoch_correct_num_list))
        epoch_loss_sum_list.append(epoch_loss_sum)      
        print('Epoch Training Loss: {:.4f} | Epoch Training Accuracy: {:.4f}%'.format(epoch_loss_sum, epoch_correct_num*100))
    return epoch_correct_num_list, epoch_loss_sum_list       

# Method to test and get accuracy of trained model
def test(model, dataloader_test, loss_func, expand_range):
    loss_sum = 0
    correct_predictions = 0
    model.eval()
    with torch.no_grad():
        for X, Y in dataloader_test:
            output = model.forward(X)
            loss = loss_func(output, Y)

            loss_sum += loss.item() * X.shape[0]

            correct_predictions += predict(output, Y, expand_range)
    loss = loss_sum/len(dataloader_test.dataset)
    return loss, correct_predictions/len(dataloader_test.dataset)

# Method to return number of correct predictions
def predict(logit, target, double):
    predictions = torch.argmax(logit, 1)
    if(double):
        for i in range(np.shape(predictions)[0]):
            logit[i][predictions[i]] = 0
        predictions2 = torch.argmax(logit, 1)
        return torch.sum((predictions == target).long()).item() + torch.sum((predictions2 == target).long()).item()
    else:
        return torch.sum((predictions == target).long()).item()

# Method fo rrandom forest regressor
def random_forest(X_train, X_test, Y_train, Y_test):
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X_train, Y_train)
    predictions = np.around(rf.predict(X_test))
    correct = 0
    for i in range(np.shape(predictions)[0]):
        if(predictions[i] == Y_test[i]):
            correct += 1
    accuracy = (correct/np.shape(predictions)[0])*100

    importances = list(rf.feature_importances_)
    #for i in range(len(importances)):    ### for loop to print importances
    #   print(i, round(importances[i], 2))
    print(accuracy)

