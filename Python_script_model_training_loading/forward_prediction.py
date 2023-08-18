
import torch
import torch.nn as nn
import torch.nn.functional as F0
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
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

id0=1 #specifiy the structure id (from 1 to 6)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net()
net.load_state_dict(torch.load("forward_"+str(int(id0))+".pt",map_location=torch.device('cpu')))
#Use the file path where the forward model "forward_id0.pt" is located.

net.eval()
 	

x0= np.loadtxt(fname="structure.csv", delimiter=",")
#Use the file path where structure features "structure.csv" is located.

x1 = torch.tensor(x0,dtype=torch.float,requires_grad=False).to(device)
y0=net(x_vali)
y1=y0.data.cpu().numpy()
np.savetxt("property.csv", y1, delimiter=",")
#The predicted normalized effective elastic constants are stored in "property.csv".
#To get the real values, use the corresponding min and max values (see folder "E_min_max" in the dataset folder)
#and scaling scheme to de-normalize.

