import torch.nn as nn
from networks import *
from train_utils import *

n_epochs = 42
batch_size = 128
batch_size_test = 1028
device = "cuda"


standard_ode_group = nn.Sequential(*get_downsampling_layers(), ODEBlock(ConvolutionalDynamicsFunction(64, time_dependent=True, norm_type="group"), device=device), *get_final_layers())
train_model_cached(standard_ode_group, file_path="cached_models/ode_group.pth", batch_size=batch_size, test_batch_size=batch_size_test, epochs=n_epochs, verbosity=2, device=device, specialised_metric=ODEMetric())

standard_ode_batch = nn.Sequential(*get_downsampling_layers(), ODEBlock(ConvolutionalDynamicsFunction(64, time_dependent=True, norm_type="batch"), device=device), *get_final_layers())
train_model_cached(standard_ode_batch, file_path="cached_models/ode_batch.pth", batch_size=batch_size, test_batch_size=batch_size_test, epochs=n_epochs, verbosity=2, device=device, specialised_metric=ODEMetric() )

simple_resnet = nn.Sequential(*get_downsampling_layers(), *get_residual_blocks(1), *get_final_layers())
train_model_cached(simple_resnet, file_path="cached_models/simple_resnet1.pth", batch_size=batch_size, test_batch_size=batch_size_test, epochs=n_epochs, verbosity=2, device=device)

resnet_6 = nn.Sequential(*get_downsampling_layers(), *get_residual_blocks(6), *get_final_layers())
train_model_cached(resnet_6, file_path="cached_models/resnet6.pth", batch_size=batch_size, test_batch_size=batch_size_test, epochs=n_epochs, verbosity=2, device=device)

no_time_ode_group = nn.Sequential(*get_downsampling_layers(), ODEBlock(ConvolutionalDynamicsFunction(64, time_dependent=False, norm_type="group"), device=device), *get_final_layers())
train_model_cached(no_time_ode_group, file_path="cached_models/no_time_ode_group.pth", batch_size=batch_size, test_batch_size=batch_size_test, epochs=n_epochs, verbosity=2, device=device, specialised_metric=ODEMetric())

no_time_ode_batch = nn.Sequential(*get_downsampling_layers(), ODEBlock(ConvolutionalDynamicsFunction(64, time_dependent=False, norm_type="batch"), device=device), *get_final_layers())
train_model_cached(no_time_ode_batch, file_path="cached_models/no_time_ode_batch.pth", batch_size=batch_size, test_batch_size=batch_size_test, epochs=n_epochs, verbosity=2, device=device, specialised_metric=ODEMetric())
