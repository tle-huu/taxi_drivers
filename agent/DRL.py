import torch
import torch.nn as nn
import torch.optim as optim

def dimensions_after_conv(grid_shape, channels_number):
    return channels_number * (grid_shape[-1] * grid_shape[-2])


class CarLeader(nn.Module):

    def __init__(self, grid_shape, number_of_cars):

        ## private variables
        self.number_of_cars = number_of_cars
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        super(CarLeader, self).__init__()


        ## Layers

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

        features = int(dimensions_after_conv(grid_shape, 64))
        # self.fc1 = nn.Sequential(nn.Linear(features, 32), nn.ELU(True))

        # One perceptron per car
        # self.fc_cars = [ nn.Linear(32, 5)  for _ in range(self.number_of_cars)]
        self.fc_cars = [ nn.Sequential(nn.Linear(features, 32), nn.ELU(True), nn.Linear(32, 5)) for _ in range(self.number_of_cars)]

        # self.fc2 = nn.Linear(32, 5)


        # Registering new layers to parameters
        for i, layer in enumerate(self.fc_cars):
          weight_1, bias_1, weight_2, bias_2 = layer.parameters()

          weight_1_name = 'weigh_1_perceptron_' + str(i)
          bias_1_name   = 'bias_1_perceptron_' + str(i)
          weight_2_name = 'weight_2_perceptron_' + str(i)
          bias_2_name   = 'bias_2_perceptron_' + str(i)

          self.register_parameter(weight_1_name, torch.nn.Parameter(weight_1))
          self.register_parameter(bias_1_name, torch.nn.Parameter(bias_1))
          self.register_parameter(weight_2_name, torch.nn.Parameter(weight_2))
          self.register_parameter(bias_2_name, torch.nn.Parameter(bias_2))

        self.loss = nn.MSELoss()
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
        self.to(self.device)
    
    def forward(self, x):
        """
        X is the grid dimension should be (n,1, gridshape[0], gridshape[1]) with n being
        the number of grids in the batch
        output is a vector of size (n,4) containing q values for up,down,left,right for 
        each grid of the batch
        """

        print(x)
        x = self.convs1(x)
        x = self.convs2(x)
        x = self.convs3(x)
        x = self.flatten(x)

        x= torch.Tensor(x)

        out = self.fc_cars[0](x).unsqueeze(1)
        # out = self.fc2(x).unsqueeze(1)
        for index in range(1, self.number_of_cars):

          out_local_car = self.fc_cars[index](x).unsqueeze(1)
          out = torch.cat( (out ,out_local_car) , dim = 1 )

        return out
