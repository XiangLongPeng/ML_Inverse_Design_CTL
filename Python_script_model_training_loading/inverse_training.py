
import torch
import torch.nn as nn
import torch.nn.functional as F0
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import math
import numpy as np



#forward ANN model parameters
n_hidden=2      #number of hidden layers
n_input=3       #number of inputs
n_hl=[128,128]  #number of neurons at each hidden layer
n_output=4      #number of outputs
class Net_f(nn.Module):
    def __init__(self):
        super(Net_f, self).__init__()
        self.hidden_layer1 = nn.Linear(n_input,n_hl[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(n_hidden-1):
          self.hidden_layers.append(nn.Linear(n_hl[i], n_hl[i+1])) 
        self.output_layer = nn.Linear(n_hl[i+1],n_output)

    def forward(self, x):
        inputs = x 
        output = F0.tanh(self.hidden_layer1(inputs))

        for layer in self.hidden_layers:
            output = layer(output)
            output = F0.tanh(output)         
        output = self.output_layer(output) 
        return output

#inverse ANN model parameters    
n_hidden_i=4            #number of hidden layers
n_input_i=4             #number of inputs
n_hl_i=[128,128,128,128]#number of neurons at each hidden layer
n_output_i=3            #number of outputs
class Net_i(nn.Module):
    def __init__(self):
        super(Net_i, self).__init__()
        self.hidden_layer1 = nn.Linear(n_input_i,n_hl_i[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(n_hidden_i-1):
          self.hidden_layers.append(nn.Linear(n_hl_i[i], n_hl_i[i+1])) 
        self.output_layer = nn.Linear(n_hl_i[i+1],n_output_i)

    def forward(self, x):
        inputs = x 
        output = F0.leaky_relu(self.hidden_layer1(inputs))

        for layer in self.hidden_layers:
            output = layer(output)
            output = F0.leaky_relu(output)         
        output = self.output_layer(output) 
        return output

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for s0 in range(6):
    id0=s0+1 #structure id
    net_f = Net_f()
    net_f.load_state_dict(torch.load("forward_"+str(id0)+".pt"))
    #Use the file path where the forward model "forward_id0" is located.
    
    net_f.to(device)
    net_f.eval()

    net_i = Net_i()
    net_i = net_i.to(device)
    mse_cost_function = torch.nn.MSELoss() # Mean squared error
    optimizer = torch.optim.Adam(net_i.parameters(),lr=0.005)

    x0= np.loadtxt(fname="input_training_"+str(id0)+".csv", delimiter=",")
    #Use the file path where inputs in the training dataset 
    #"input_training_id0.csv" is located. 
    x_training = torch.tensor(x0,dtype=torch.float,requires_grad=False).to(device)
    y0= np.loadtxt(fname="output_training_"+str(id0)+".csv", delimiter=",")
    #Use the file path where outputs in the training dataset 
    #"output_training_id0.csv" is located.

    y_training0 = net_f(x_training)
    y00=y_training0.data.cpu().numpy()
    y_training = torch.tensor(y00,dtype=torch.float,requires_grad=False).to(device)
    
    x0= np.loadtxt(fname="input_test_"+str(id0)+".csv", delimiter=",")
    #Use the file path where inputs in the test dataset 
    #"input_test_id0.csv" is located.    
    x_test = torch.tensor(x0,dtype=torch.float,requires_grad=False).to(device)
    y0= np.loadtxt(fname="output_test_"+str(id0)+".csv", delimiter=",")
    #Use the file path where outputs in the test dataset 
    #"output_test_id0.csv" is located.    
    y_test0 = net_f(x_test)
    y00=y_test0.data.cpu().numpy()
    y_test = torch.tensor(y00,dtype=torch.float,requires_grad=False).to(device)    

    BATCH_SIZE=500
    dataset = TensorDataset(y_training, x_training)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,drop_last=True, shuffle=True)###drop_last=True,
    dataset_size = len(dataloader.dataset)
    iterations = 200
    ep_num0=[]
    loss_training=[]
    loss_test=[]
    for epoch in range(iterations):
        loss_temp=0
        iii0=0
        for id_batch, (x_batch, y_batch) in enumerate(dataloader):
            optimizer.zero_grad()

            y_p=net_i(x_batch)
            mse0=mse_cost_function(net_f(y_p), x_batch)
            mse0_pred=mse_cost_function(y_p, y_batch)
            lambda0=0.2
            if epoch>39:
               lambda0=0                
            loss = mse0+lambda0*mse0_pred         
            loss.backward()
            optimizer.step()
            loss_temp=loss_temp+loss.item()
            iii0=iii0+1
            torch.save(net_i.state_dict(), "inverse_"+str(id0)+"_"+str(epoch)+".pt")
            #Save the model after each epoch. 
            #The one with smalllest test loss is selected as the final model.            
        ep_num0.append(epoch)        
        loss_training.append(loss_temp/(iii0*1.0))
        net_i.eval()
        y_p0=net_i(y_test)
        mse00=mse_cost_function(net_f(y_p0), y_test)
        mse00_pred=mse_cost_function(y_p0, x_test)
        val_loss = mse00+lambda0*mse00_pred
        val_loss = mse00  
        loss_test.append(val_loss.item())        
    ep_num1=np.array(ep_num0).reshape(-1,1)    
    loss1_training=np.array(loss_training).reshape(-1,1)  
    np.savetxt("loss_training_inverse_"+str(id0)+".csv", loss1_training, delimiter=",")
    #Save the training loss
    loss1_test=np.array(loss_test).reshape(-1,1)  
    np.savetxt("loss_test_inverse_"+str(id0)+".csv", loss1_test, delimiter=",")
    #Save the test loss
    #print(epoch)    





