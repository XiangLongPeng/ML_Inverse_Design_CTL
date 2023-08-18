
import torch
import torch.nn as nn
import torch.nn.functional as F0
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import math
import numpy as np

#ANN model parameters
n_hidden=2      #number of hidden layers
n_input=3       #number of inputs
n_hl=[128,128]  # number of neurons at each hidden layer
n_output=4      #number of outputs
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for s0 in range(6):
    id0=s0+1 #structure id
    net = Net()
    net = net.to(device)
    mse_cost_function = torch.nn.MSELoss() # Mean squared error
    optimizer = torch.optim.Adam(net.parameters(),lr=0.005)
    
    x0= np.loadtxt(fname="input_training_"+str(id0)+".csv", delimiter=",")
    #Use the file path where inputs in the training dataset 
    #"input_training_id0.csv" is located.    
    x_training = torch.tensor(x0,dtype=torch.float,requires_grad=False).to(device)
    
    y0= np.loadtxt(fname="output_training_"+str(id0)+".csv", delimiter=",")
    #Use the file path where outputs in the training dataset 
    #"output_training_id0.csv" is located.   
    y_training = torch.tensor(y0,dtype=torch.float,requires_grad=False).to(device)
    
    x1= np.loadtxt(fname="input_test_"+str(id0)+".csv", delimiter=",")
    #Use the file path where inputs in the test dataset 
    #"input_test_id0.csv" is located.    
    x_test = torch.tensor(x1,dtype=torch.float,requires_grad=False).to(device)
    
    y1= np.loadtxt(fname="output_test_"+str(id0)+".csv", delimiter=",")
    #Use the file path where outputs in the test dataset 
    #"output_test_id0.csv" is located.
    y_test = torch.tensor(y1,dtype=torch.float,requires_grad=False).to(device)    
    BATCH_SIZE=500
    dataset = TensorDataset(x_training, y_training)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,drop_last=True, shuffle=True)###drop_last=True,
    dataset_size = len(dataloader.dataset)
    iterations = 200
    ep_num0=[]
    loss_training=[]
    loss_test=[]
    print(s0+2)
    for epoch in range(iterations):

        loss_temp=0
        iii0=0
        for id_batch, (x_batch, y_batch) in enumerate(dataloader):
            optimizer.zero_grad()

            y_p=net(x_batch)
            mse0=mse_cost_function(y_p, y_batch)        
            loss = mse0         
            loss.backward()
            optimizer.step()
            loss_temp=loss_temp+loss.item()
            iii0=iii0+1
        ep_num0.append(epoch)        
        loss_training.append(loss_temp/(iii0*1.0))
        torch.save(net.state_dict(), "forward_"+str(id0)+"_"+str(epoch)+".pt") 
        #Save the model after each epoch. 
        #The one with smalllest test loss is selected as the final model.
        
        net.eval()
        y_test_p = net(x_test)
        val_loss = mse_cost_function(y_test_p, y_test)
        loss_test.append(val_loss.item())
    ep_num1=np.array(ep_num0).reshape(-1,1)    
    loss1_training=np.array(loss_training).reshape(-1,1)  
    np.savetxt("loss_training_"+str(id0)+".csv", loss1_training, delimiter=",")
    #Save the training loss
    
    loss1_test=np.array(loss_test).reshape(-1,1)  
    np.savetxt("loss_test_"+str(id0)+".csv", loss1_test, delimiter=",")
    #Save the test loss    
    #print(epoch)    
