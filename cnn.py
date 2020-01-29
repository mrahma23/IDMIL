import torch.nn as nn
import torch.nn.functional as F

class CNN_net(nn.Module):
    def __init__(self):
        super(CNN_net, self).__init__()
        
        self.pool = nn.MaxPool1d(2, stride = 2, ceil_mode = True)
        
        ######## Groups of bigger kernel #### 
        ### input size: 16384
        self.conv1 = nn.Conv1d(64, 64, 5)
        ### input size: 8190
        self.conv2 = nn.Conv1d(64, 64, 5)
        ### input size: 4093
        self.conv3 = nn.Conv1d(64, 64, 5)
        
        
        ######## Groups of smaller kernels ####
        ### input size: 2045
        self.conv4 = nn.Conv1d(64, 64, 3)
        ### input size: 1022
        self.conv5 = nn.Conv1d(64, 64, 3)
        ### input size: 510
        self.conv6 = nn.Conv1d(64, 64, 3)
        

        #### The fully-connected layer ######
        ### input size: 254  
        self.fc1 = nn.Linear(64 * 254, 64 * 254)
        self.fc2 = nn.Linear(64 * 254, 3)

    def forward(self, x):
        ##### bigger kernels, they remove noises 
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        ##### smaller kernel, they focus on fine-details selected by earlier kernels and poolings 
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = self.pool(F.relu(self.conv6(x)))
        
        #### fully connected layer
        x = x.view(-1, 64 * 254)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return F.log_softmax(x, dim = 1)