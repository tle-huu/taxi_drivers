import torch
import torch.nn as nn


class Car_Leader(nn.Module):
    def __init__(self, cars, hidden_channels, grid_shape):
        super(Car_Leader, self).__init__()
        self.cars = cars
        self.convs1 = nn.Sequential(nn.Conv2d(2,hidden_channels, kernel_size=3,padding=1),nn.ReLU(True))
        self.flat = torch.nn.Flatten()

        f = lambda gshape,channs : channs*(gshape[0]*gshape[1])/4# chaque maxpool réduit le nombre de feature par 4
        features = int(f(grid_shape,hidden_channels))
        self.fc1 = nn.Sequential(nn.Linear(features, 32), nn.ReLU(True))
        self.fc2 = nn.Sequential(nn.Linear(32, 4), nn.ReLU(True))
        
    def forward(self, x):
        """
        x contains the gid and the positions of all cars
        size : (1,number of cars-1,gridshape[0], gridshape[1]) the -1 comes 
        from the fact that the grid is part of the input
        To parallelize computation, we the put it to the shape (number of cars, 2, gridshape[0], gridshape[1])
        All cars are given as a batch of input each of them are together with the grid
        The output is a matrix of size (number of cars,4) every line representing the 
        actions values for a car (left,right,up,down)
        nn.Softmax(1)(x1) will transform everything into probabilities
        """
        xsize = x.size()
        number_of_cars = xsize[1]-1
        xreshaped = torch.zeros(number_of_cars,2,xsize[2],xsize[3])
        grid = x[0][0]
        for i in range(number_of_cars):
            xreshaped[i][0] = grid
            xreshaped[i][1] = x[0][i+1]
        x1 = self.convs1(xreshaped)
        x1 = nn.MaxPool2d(2,stride=2)(x1)
        x1 = self.flat(x1)
        x1 = self.fc1(x1)
        x1 = self.fc2(x1)
        return x1

        