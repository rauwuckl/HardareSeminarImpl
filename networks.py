import torch
import torch.nn as nn
from torch.autograd import Variable
from torchdiffeq import odeint_adjoint as odeint


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class ResidualBlock(nn.Module):
    def norm(self):
        if self.norm_type=="batch":
            return nn.BatchNorm2d(self.n_channels)
        elif self.norm_type=="group":
            return nn.GroupNorm(num_groups=32, num_channels=self.n_channels)
            # 32 is the default group number suggested in https://arxiv.org/pdf/1803.08494.pdf
        else:
            raise ValueError("unknown normalisation type")

    def conv(self):
        return nn.Conv2d(self.n_channels, self.n_channels, kernel_size=self.kernel_size, stride=1, padding=True)

    def __init__(self, n_channels, kernel_size=3, cache_last_activation=False, norm_type="batch"):

        super(ResidualBlock, self).__init__()

        self.n_channels = n_channels
        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.cache_last_activation = cache_last_activation

        self.norm_layer1 = self.norm()
        self.relu_layer1 = nn.ReLU(inplace=True)
        self.conv_layer1 = self.conv()

        self.norm_layer2 = self.norm()
        self.relu_layer2 = nn.ReLU(inplace=True)
        self.conv_layer2 = self.conv()

    def forward(self, x):
        skip = x

        x = self.norm_layer1(x)
        x = self.relu_layer1(x)
        x = self.conv_layer1(x)

        x = self.norm_layer2(x)
        x = self.relu_layer2(x)
        x = self.conv_layer2(x)

        out =  x + skip

        if self.cache_last_activation:
            self.last_activation = out

        return out

class TimeDependentConv(nn.Module):
    # a convolution that also depends on the t parameter (by appending a constant channel with value t in all pixels)

    def __init__(self, channels, kernel_size, time_dependent=True):
        """
        :param channels:
        :param kernel_size:
        :param time_dependent: if False, t gets ignored in the forward pass
        """
        super(TimeDependentConv, self).__init__()

        self.time_dependent = time_dependent

        self.conv_layer = nn.Conv2d(channels + self.time_dependent, channels, kernel_size=kernel_size, padding=True)

    def forward(self, x, t):

        if self.time_dependent:
            time_channel = torch.ones_like(x[:, :1, :, :]) * t # trick to get right shape, data type and device
            with_time = torch.cat((x, time_channel), 1)
        else:
            with_time = x

        return self.conv_layer(with_time)

class ConvolutionalDynamicsFunction(nn.Module):

    def norm(self):
        if self.norm_type=="batch":
            return nn.BatchNorm2d(self.n_channels)
        elif self.norm_type=="group":
            return nn.GroupNorm(num_groups=32, num_channels=self.n_channels)
            # 32 is the default group number suggesting in https://arxiv.org/pdf/1803.08494.pdf
        else:
            raise ValueError("unknown normalisation type")

    def conv(self):
        return TimeDependentConv(self.n_channels, kernel_size=self.kernel_size, time_dependent=self.time_dependent)

    def reset_function_evaluations(self):
        self.n_function_evaluations = 0

    def get_function_evaluations(self):
        return self.n_function_evaluations

    def __init__(self, n_channels, kernel_size=3, time_dependent=True, norm_type="batch"):
        """
        :param n_channels:
        :param kernel_size:
        :param time_dependent: wether the convolution should behave differently at different times t within the ODE Block
        """
        super(ConvolutionalDynamicsFunction, self).__init__()
        self.n_channels = n_channels
        self.kernel_size=kernel_size
        self.time_dependent = time_dependent
        self.norm_type = norm_type

        self.reset_function_evaluations()

        self.norm_layer1 = self.norm()
        self.relu_layer1 = nn.ReLU(inplace=True)
        self.conv_layer1 = self.conv()

        self.norm_layer2 = self.norm()
        self.relu_layer2 = nn.ReLU(inplace=True)
        self.conv_layer2 = self.conv()

        self.norm_layer3 = self.norm()

    def forward(self, t, x):
        self.n_function_evaluations += 1

        x = self.norm_layer1(x)
        x = self.relu_layer1(x)
        x = self.conv_layer1(x, t)

        x = self.norm_layer2(x)
        x = self.relu_layer2(x)
        x = self.conv_layer2(x, t)

        x = self.norm_layer3(x)

        return x

class ODEBlock(nn.Module):

    def __init__(self, dynamics_function, intermediate_values_to_compute=None, atol=1e-3, rtol=1e-3, device='cpu'):
        """

        :param dynamics_function: a function that maps the current state and time to a tensor of the same shape as the current state
        :param intermediate_values_to_compute: (optional) a numpy array of time values between 0, 1 for which the state is computed
        :param atol: absolute tolerance of ODE solution
        :param rtol: relative tolerance of ODE solution
        """
        super(ODEBlock, self).__init__()

        self.dynamics_function = dynamics_function
        self.intermediate_values_to_compute = intermediate_values_to_compute
        self.atol = atol
        self.rtol = rtol

        if intermediate_values_to_compute is None:
            self.integration_time = torch.tensor([0.0, 1.0]).to(device)
        else:
            assert(intermediate_values_to_compute[0] == 0)
            assert(intermediate_values_to_compute[-1] == 1)
            self.integration_time = torch.tensor(intermediate_values_to_compute).to(device)

    def forward(self, x):

        out = odeint(self.dynamics_function, x, self.integration_time, atol=self.atol, rtol=self.rtol)
        # the backward path is already implemented for odeint in the package torchdiffeq

        if self.intermediate_values_to_compute is not None:
            self.intermediate_values_of_last_batch = out

        return out[-1]


def get_residual_blocks(n_blocks, cache_last_activation=False):
    layers = [ResidualBlock(64, cache_last_activation=cache_last_activation) for i in range(n_blocks)]
    return layers


def get_downsampling_layers():
    layers = [
        nn.Conv2d(1, 64, kernel_size=3, stride=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
    ]
    return layers

def get_final_layers():
    layers = [
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1,1)),
        Flatten(),
        nn.Linear(64, 10)
    ]
    return layers

def model_to_onnx(model, batch_size, file_path, channels=1, height=64, width=64):
    dummy_input = Variable(torch.randn(batch_size, channels, height, width))

    torch.onnx.export(model, dummy_input, file_path, verbose=True)
    return file_path

