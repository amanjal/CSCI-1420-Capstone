import os
import numpy as np
import torch
import math
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



class PatientDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = torch.LongTensor(Y)
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        return self.X[index], self.Y[index]

# Method to read CSV and preprocess data into proper format
def hospital_loader(dataset = "data/train_data.csv", batch_size = 128, method = True, one_hot = True):
    if not os.path.exists(dataset):
        raise FileNotFoundError('This file {} does not exist'.format(dataset))

    data = read_csv(dataset, header = None, low_memory = False).values # read csv data

    x_temp = data[:, 1:-1].astype(str) # get rid of first & last column for train data
    y_temp = data[:, -1].astype(str) # set Y to last column (labels)
    l_encoder = LabelEncoder() # encoder for labels



    Y = np.array(l_encoder.fit_transform(y_temp), dtype = np.float32) # encode labels, convert to np array
    Y = np.delete(Y, 0, 0) # Remove category names

    # For loop to encode and add new features to X_Copy
    if(one_hot):
        X = one_hot_encode([1, 3, 5, 6, 7, 11, 12, 14], np.array(x_temp))
    else:
        o_encoder = OrdinalEncoder() # encoder for features
        X = np.array(o_encoder.fit_transform(x_temp), dtype = np.float32)
        #X = np.delete(X, [0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 14], 1)  ##### IF we only want important features from RF model

    X = (X - np.mean(X, axis = 0))/ np.std(X, axis = 0) # normalize feature values
    # Since we don't have a test dataset, we split the test points from the ~320,000 training points
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.1)

    if(method): # if we are doing NN, use dataloader
        dataloader_train = DataLoader(PatientDataset(X_Train, Y_Train), batch_size = batch_size, shuffle = True)
        dataloader_test = DataLoader(PatientDataset(X_Test, Y_Test), batch_size = batch_size, shuffle = False)
        return dataloader_train, dataloader_test
    else: # if doing random forest, no need for dataloader
        return X_Train, X_Test, Y_Train, Y_Test

# Method to take in a column of categorical values and turn it into an np array mimicing one hot encoding
def encode(Category, Col):
    unique = {}
    row, col = np.shape(Category)
    counter = 0

    # For loop to find and count unique categorical values in passed in column
    for i in range(1, row):
        x = Category[i][Col]
        if x not in unique:
            unique[x] = counter
            counter += 1

    encoded = np.zeros((row-1, len(unique))) # encoded matrix of 0's

    # Fill encoded matrix with 1 in proper spots mimics one hot encoding
    for i in range(1, row):
        c = unique[Category[i][Col]]
        encoded[i-1][c] = 1

    return encoded

# Method to make all NaN 0
def remove_NaN(x):
    for i in range(np.shape(x)[0]):
            for j in range(np.shape(x)[1]):
                if(math.isnan(x[i][j])):
                    x[i][j] = 0
    return x

# one hot encode certain indices of input data
def one_hot_encode(indices_to_encode, X):
    X_copy = np.copy(X) # copy X
    X_copy = np.delete(X_copy, 0, 0) # delete the top row (the categories)
    X_copy = np.delete(X_copy, indices_to_encode, 1) # delete indices we will be encoding

    for i in indices_to_encode:
        X_copy = np.hstack((X_copy, encode(X, i)))
    
    x = remove_NaN(np.array(X_copy, dtype = np.float32)) # turn X copy in np array w/ proper dtype
    return x