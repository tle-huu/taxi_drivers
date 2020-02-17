import torch
import torch.nn as nn
import torch.optim as optim

def dimensions_after_conv(grid_shape, channels_number):
    return channels_number * (grid_shape[0] * grid_shape[1])


class CarLeader(nn.Module):

    def __init__(self, grid_shape):

        super(CarLeader, self).__init__()


        self.convs1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size = 3, padding = 1),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(True))

        self.convs2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size = 3, padding = 1),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(True))
        
        self.convs3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(True))
        
        
        self.flatten = torch.nn.Flatten()

        features = int(dimensions_after_conv(grid_shape, 64))
        self.fc1 = nn.Sequential(nn.Linear(features, 32), nn.ReLU(True))
        self.fc2 = nn.Linear(32, 5)
        self.loss = nn.MSELoss()
        self.device = torch.device("cpu")
        self.optimizer = optim.RMSprop(self.parameters(), lr = 0.01)
        
    def forward(self, x):
        """
        X is the grid dimension should be (n,1, gridshape[0], gridshape[1]) with n being
        the number of grids in the batch
        output is a vector of size (n,4) containing q values for up,down,left,right for 
        each grid of the batch
        """
        x = self.convs1(x)
        x = self.convs2(x)
        x = self.convs3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
"""
Pour tester :
grille = torch.rand(5,5).unsqueeze(0)
grille = (grille<0.5).float()
grille =grille.unsqueeze(0)

traf = CarLeader([5,5])
traf(grille)

"""
