import torch
import torch.nn as nn
from torch.autograd import Variable
from torchdiffeq import odeint_adjoint as odeint


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class ResidualBlock(nn.Module):
    def norm(self):
        return nn.BatchNorm2d(self.n_channels)

    def conv(self):
        return nn.Conv2d(self.n_channels, self.n_channels, kernel_size=self.kernel_size, stride=1, padding=True)


    def __init__(self, n_channels, kernel_size=3, cache_last_activation=False):

        super(ResidualBlock, self).__init__()

        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.cache_last_activation = cache_last_activation

        # self.relu_layer1 = nn.ReLU(inplace=True) #TODO
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
            self.cache_last_activation = out

        return out

class TimeDependentConv(nn.Module):
    # a convolution that also depends on the t parameter, (by appending a constant channel with value t in all pixels)

    def __init__(self, channels, kernel_size, time_dependent=True):
        """

        :param channels:
        :param kernel_size:
        :param time_dependent: if False, the t gets ignored in the forward pass
        """
        super(TimeDependentConv, self).__init__()
        # time_dependent = True
        # self.time_independent_debug = False

        self.time_dependent = time_dependent #or self.time_independent_debug

        self.conv_layer = nn.Conv2d(channels + self.time_dependent, channels, kernel_size=kernel_size, padding=True)

    def forward(self, x, t):
        # if self.time_independent_debug:
        #     t = 0

        if self.time_dependent:
            # print(t)
            time_channel = torch.ones_like(x[:, :1, :, :]) * t # trick to get right shape, data type and device
            with_time = torch.cat((x, time_channel), 1)
        else:
            with_time = x

        return self.conv_layer(with_time)


class ConvolutionalDynamicsFunction(nn.Module):
    def norm(self):
        return nn.BatchNorm2d(self.n_channels)

    def conv(self):
        return TimeDependentConv(self.n_channels, kernel_size=self.kernel_size, time_dependent=self.time_dependent)

    def __init__(self, n_channels, kernel_size=3, time_dependent=True):
        """

        :param n_channels:
        :param kernel_size:
        :param time_dependent: wether the convlution should behave differently at different times t within the ODE Block
        """
        super(ConvolutionalDynamicsFunction, self).__init__()
        self.n_channels = n_channels
        self.kernel_size=kernel_size
        self.time_dependent = time_dependent

        self.norm_layer1 = self.norm()
        self.relu_layer1 = nn.ReLU(inplace=True)
        self.conv_layer1 = self.conv()

        self.norm_layer2 = self.norm()
        self.relu_layer2 = nn.ReLU(inplace=True)
        self.conv_layer2 = self.conv()

        self.norm_layer3 = self.norm() # TODO <- is not applied

    def forward(self, t, x):
        x = self.norm_layer1(x)
        x = self.relu_layer1(x)
        x = self.conv_layer1(x, t)

        x = self.norm_layer2(x)
        x = self.relu_layer2(x)
        x = self.conv_layer2(x, t)

        return x

class ODEBlock(nn.Module):

    def __init__(self, dynamics_function, intermediate_values_to_compute=None, atol=1e-3, rtol=1e-3, device='cpu'):
        """

        :param dynamics_function:
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
            self.integration_time = torch.tensor([0.0 , 1.0]).to(device)
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


def get_residual_blocks(n_blocks):
    layers = [ResidualBlock(64) for i in range(n_blocks)]
    return layers


def get_downsampling_layers():
    layers = [
        nn.Conv2d(1, 64, kernel_size=3, stride=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64,64,2,1)
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

if __name__ == "__main__":
    print('hello')

    # Get dataset size
    mnist_shape = d5ds.dataset_shape('mnist')
    print(mnist_shape)
    classes, c, h, w = mnist_shape
    BATCH_SIZE = 128

    # model = nn.Sequential(*get_downsamplint_layers(), *get_final_layers())

    # onnx_file = model_to_onnx(model, 128, "test.onnx")

    onnx_file = d5nt.export_network("simple_cnn", BATCH_SIZE, classes=classes,
                                 channels=c, height=h, width=w)

    d5_model = d5.parser.load_and_parse_model(onnx_file)

    INPUT_NODE = d5_model.get_input_nodes()[0].name
    OUTPUT_NODE = d5_model.get_output_nodes()[0].name

    train_set, test_set = d5ds.load_dataset('mnist', INPUT_NODE, 'labels')
    d5_model.add_operation(d5.ops.LabelCrossEntropy([OUTPUT_NODE, 'labels'], 'loss'))

    train_sampler = d5.ShuffleSampler(train_set, BATCH_SIZE)
    test_sampler = d5.ShuffleSampler(test_set, BATCH_SIZE)

    executor = d5pt.from_model(d5_model)

    optimizer = d5ref.GradientDescent(executor, 'loss', 0.1)

    EPOCHS = 2
    d5.test_training(executor, train_sampler, test_sampler, optimizer,
                     EPOCHS, BATCH_SIZE, OUTPUT_NODE)

    # executor = d5pt.PyTorchGraphExecutor(model)





    # onnx_file = d5nt.export_network('simple_cnn', BATCH_SIZE, classes=classes,
    #                                 channels=c, height=h, width=w)
    # model = d5.parser.load_and_parse_model(onnx_file)


    print()