import torch.nn as nn
from networks import *
from train_utils import *

n_epochs = 23
batch_size = 128
batch_size_test = 1028
device="cuda"

# Baseline ResNet

baseline_resnet = nn.Sequential(*get_downsampling_layers(), *get_residual_blocks(6), *get_final_layers())
train_model_cached(baseline_resnet, file_path="cached_models/baseline_resnet6.pth", batch_size=128, test_batch_size=batch_size_test, epochs=n_epochs, verbosity=2, device=device)

simple_resnet = nn.Sequential(*get_downsampling_layers(), *get_residual_blocks(1), *get_final_layers())
train_model_cached(simple_resnet, file_path="cached_models/simple_resnet1.pth", batch_size=128, test_batch_size=batch_size_test, epochs=n_epochs, verbosity=2, device=device)

standard_ode = nn.Sequential(*get_downsampling_layers(), ODEBlock(ConvolutionalDynamicsFunction(64, time_dependent=True)), *get_final_layers())
train_model_cached(standard_ode, file_path="cached_models/ode.pth", batch_size=128, test_batch_size=batch_size_test, epochs=n_epochs, verbosity=2, device=device)

no_time_ode = nn.Sequential(*get_downsampling_layers(), ODEBlock(ConvolutionalDynamicsFunction(64, time_dependent=False)), *get_final_layers())
train_model_cached(no_time_ode, file_path="cached_models/no_time_ode.pth", batch_size=128, test_batch_size=batch_size_test, epochs=n_epochs, verbosity=2, device=device)