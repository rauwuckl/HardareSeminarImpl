import torch.nn as nn
from networks import *
from train_utils import *

model = nn.Sequential(*get_downsampling_layers(), ODEBlock(ConvolutionalDynamicsFunction(64)), *get_final_layers())
train_model_cached(model, file_path="cached_models/ode1.pth", batch_size=128, epochs=10, verbosity=5)