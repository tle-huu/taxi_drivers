import torch
import torch.nn as nn
import torch.optim as optim

def dimensions_after_conv(grid_shape, channels_number):
    return channels_number * (grid_shape[-1] * grid_shape[-2])


class CarLeader(nn.Module):

    def __init__(self, grid_shape, number_of_cars=1):

        super(CarLeader, self).__init__()


        self.convs1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size = 3, padding = 1),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(True))

        self.convs2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size = 3, padding = 1),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(True))
        
        self.convs3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(True))
        
        
        self.flatten = torch.nn.Flatten()
        self.number_of_cars = number_of_cars
        features = int(dimensions_after_conv(grid_shape, 64))
        self.fc1 = nn.Sequential(nn.Linear(features, 32), nn.ELU(True))
        self.fcout = [nn.Linear(32, 5) for i in range(self.number_of_cars)]
        self.loss = nn.MSELoss()
        self.device = torch.device("cpu")
        self.optimizer = optim.RMSprop(self.parameters(), lr = 0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                     mode='min',
                                                     factor=0.5,
                                                     patience=1000,
                                                     verbose=True,
                                                     threshold=0.0001,
                                                     threshold_mode='rel',
                                                     cooldown=10,
                                                     min_lr=0,
                                                     eps=1e-08)
    
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
        out = torch.tensor([])
        for layer in self.fcout:
            out = torch.cat((out,layer(x).unsqueeze(1)), dim = 1)
        return out
