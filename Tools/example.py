import torch
import torch.nn as nn    #neural network model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler

#Load datasets
dataset = pd.read_csv('test_100.csv')

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1:].values

X_scaler = MinMaxScaler()
Y_scaler = MinMaxScaler()
print(X_scaler.fit(X))
print(Y_scaler.fit(Y))
X = X_scaler.transform(X)
Y = Y_scaler.transform(Y)

x_temp_train = X[:79]
y_temp_train = Y[:79]
x_temp_test = X[80:]
y_temp_test = Y[80:]

X_train = torch.FloatTensor(x_temp_train)
Y_train = torch.FloatTensor(y_temp_train)
X_test = torch.FloatTensor(x_temp_test)
Y_test = torch.FloatTensor(y_temp_test)

D_in = 1 # D_in is input features
H = 24 # H is hidden dimension
D_out = 1 # D_out is output features.

#Define a Artifical Neural Network model
class Net(nn.Module):
#------------------Two Layers------------------------------
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()

        self.linear1 = nn.Linear(D_in, H)  
        self.linear2 = nn.Linear(H, D_out)
        
    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        prediction = self.linear2(h_relu)
        return prediction
model = Net(D_in, H, D_out)
print(model)

#Define a Loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.2) #2e-7

#Training
inputs = Variable(X_train)
outputs = Variable(Y_train)
inputs_val = Variable(X_test)
outputs_val = Variable(Y_test)
loss_values = []
val_values = []
epoch = []
for phase in ['train', 'val']:
    if phase == 'train':
        model.train()  # Set model to training mode
    else:
        optimizer.zero_grad() #zero the parameter gradients
        model.eval()   # Set model to evaluate mode
    for i in range(50):      #epoch=50
        if phase == 'train':
            model.train()
            prediction = model(inputs)
            loss = criterion(prediction, outputs) 
            print('train loss')
            print(loss)
            loss_values.append(loss.detach())
            optimizer.zero_grad() #zero the parameter gradients
            epoch.append(i)
            loss.backward()       #compute gradients(dloss/dx)
            optimizer.step()      #updates the parameters
        elif phase == 'val':
            model.eval()
            prediction_val = model(inputs_val)
            loss_val = criterion(prediction_val, outputs_val) 
            print('validation loss')
            print(loss)
            val_values.append(loss_val.detach())
            optimizer.zero_grad() #zero the parameter gradients
          
plt.plot(epoch,loss_values)
plt.plot(epoch, val_values)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper left')
plt.show()
