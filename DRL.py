import torch
import torch.nn as nn


def dimensions_after_conv(grid_shape, channels_number):
    return channels_number * (grid_shape[0] * grid_shape[1])


class CarLeader(nn.Module):

    def __init__(self, cars_number, hidden_channels, grid_shape):

        super(CarLeader, self).__init__()

        self.cars_number = cars_number

        self.convs1 = nn.Sequential(nn.Conv2d(2, hidden_channels, kernel_size = 3, padding = 1),
                                    nn.MaxPool2d(hidden_channels),
                                    nn.ReLU(True))

        self.flatten = torch.nn.Flatten()

        features = int(dimensions_after_conv(grid_shape, hidden_channels))
        self.fc1 = nn.Sequential(nn.Linear(features, 32), nn.ReLU(True))
        self.fc2 = nn.Sequential(nn.Linear(32, 4), nn.ReLU(True))

    def forward(self, x):
        """
        x contains the grid and the positions of all cars
        size : (1, number of cars + 1,gridshape[0], gridshape[1]) the +1 comes
        from the fact that the grid is part of the input
        To parallelize computation, we the put it to the shape :
        (number of cars, 2, gridshape[0], gridshape[1])
        All cars are given as a batch of input each of them are together with the grid
        The output is a matrix of size (number of cars,4) every line representing the
        actions values for a car (left,right,up,down)
        nn.Softmax(1)(x1) will transform everything into probabilities
        """

        xsize = x.size()
        number_of_cars = xsize[1] - 1
        xreshaped = torch.zeros(number_of_cars, 2, xsize[2], xsize[3])
        grid = x[0][0]

        for i in range(number_of_cars):
            xreshaped[i][0] = grid
            xreshaped[i][1] = x[0][i + 1]

        x1 = self.convs1(xreshaped)
        x1 = self.flatten(x1)
        x1 = self.fc1(x1)
        x1 = self.fc2(x1)
        return x1
"""
Pour tester :
grille = torch.rand(5,5)
grille = (grille<0.5).float() ==>initialises the grid

car_pos = torch.zeros(5,5)
car_pos[0][1] = 1.    ==>puts the car at the 0 1 spot
traf = Car_Leader(1,5,[8,8])
traf(tor.stack((grille,car_pos)).unsqueeze())

"""
