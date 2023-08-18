import torch
import torch.nn as nn
import torch.nn.functional as F0
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

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

id0=1 #specifiy the structure id (from 1 to 6)        
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net_i = Net_i()
net_i.load_state_dict(torch.load("inverse_"+str(int(id0))+".pt",map_location=torch.device('cpu')))
#Use the file path where the inverse model "inverse_id0.pt" is located.
net_i.eval()

x0= np.loadtxt(fname="target.csv", delimiter=",")
#Use the file path where the target properties "target.csv" is located.

x1 = torch.tensor(x0,dtype=torch.float,requires_grad=False).to(device)
y0=net_i(x1)
y1=y0.data.cpu().numpy()
np.savetxt("inverse_geometry.csv", y1, delimiter=",")
#The predicted structure features are stored in "inverse_geometry.csv", 
#which can act as the inputs for the forward model to predict the corresponding
#effective elastic constants.

