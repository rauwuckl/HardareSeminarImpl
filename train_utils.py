import torch
import os
import traceback
import sys

from networks import *
import numpy as np

from torchvision import datasets, transforms

def get_data_loaders(batch_size, test_batch_size=None):
    if test_batch_size is None:
        test_batch_size = batch_size

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size, shuffle=True)

    return train_loader, test_loader

def compute_accuracy(model, data_loader, device):
    total_correct = 0
    total = 0

    with torch.no_grad():
        model.eval()
        for img, label in data_loader:
            img_device = img.to(device)


            output = model(img_device)
            prediction = torch.argmax(output, dim=-1)

            correct = torch.sum(prediction == label.to(device)).cpu().detach().float().item()

            total_correct += correct
            # print(len(label))
            total += len(label)

        model.train()

    return total_correct/float(total)

def train_model_cached(model, file_path=None, device="cpu", **kwargs):
    print("Experiment {}".format(file_path))
    print(model)
    kwargs["device"]=device

    if file_path and os.path.isfile(file_path):
        saved = torch.load(file_path, map_location=kwargs["device"])
        restored_args = saved['args']

        if restored_args != kwargs:
            print("some Parameters were different")
            # raise ValueError("For the given file_path, there already exists a cached network, that was produced with different parameters")

        state_dict = saved['state_dict']

        model.load_state_dict(state_dict) # now containing parameters
        return model, saved['stats']

    else:
        try:
            model, stats = train_model(model, **kwargs)
        except Exception as e:
            errormsg = traceback.format_exception(*sys.exc_info())
            stats = errormsg

        if file_path is None:
            return model, stats
        else:
            to_save = {'state_dict': model.state_dict(), 'args': kwargs, 'stats': stats }
            torch.save(to_save, file_path)



class SpecialisedMetric:
    def batch_forward(self, model):
        pass
    def batch_backward(self, model):
        pass
    def epoch(self, model):
        pass
    def get_summary_dict(self):
        return dict()

class ODEMetric(SpecialisedMetric):

    def __init__(self):
        self.function_evaluations_forward = list()
        self.function_evaluations_backward = list()

        self.all_epochs_forward = list()
        self.all_epochs_backward = list()

    def find_ode_dynamics(self, model):
        for layer in model.children():
            if isinstance(layer, ODEBlock):
                return layer.dynamics_function

    def batch_forward(self, model):
        """ will be called after a batch went through the forward pass"""
        dyn_function = self.find_ode_dynamics(model)

        n = dyn_function.get_function_evaluations()
        dyn_function.reset_function_evaluations()
        self.function_evaluations_forward.append(n)
        # print("{} function evaluations, forward".format(n))

    def batch_backward(self, model):
        """ will be called after a batch went through the backward pass"""
        dyn_function = self.find_ode_dynamics(model)

        n = dyn_function.get_function_evaluations()
        dyn_function.reset_function_evaluations()
        self.function_evaluations_backward.append(n)
        # print("{} function evaluations, backward".format(n))

    def epoch(self, model):
        """"""
        self.all_epochs_forward.append(self.function_evaluations_forward)
        self.function_evaluations_forward = list()

        self.all_epochs_backward.append(self.function_evaluations_backward)
        self.function_evaluations_backward = list()

    def get_summary_dict(self):
        return dict(function_evals_forward = self.all_epochs_forward, function_evals_backward = self.all_epochs_backward)



def train_model(model, batch_size, epochs, test_batch_size = None, device="cpu", learning_rate=0.1, loss_op = nn.CrossEntropyLoss, specialised_metric=None, verbosity=1, n_prints_per_epoch=100):
    if specialised_metric is None:
        specialised_metric = SpecialisedMetric()

    train_loader, test_loader = get_data_loaders(batch_size=batch_size, test_batch_size=test_batch_size)
    epoch_length = len(train_loader)
    print_every_n = epoch_length // n_prints_per_epoch

    model.to(device)
    loss_function = loss_op().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    epoch_test_accuracies = np.zeros(epochs)
    epoch_train_accuracies = np.zeros(epochs)

    for epoch_nr in range(epochs):

        print("Epoch {}".format(epoch_nr))
        model.train()
        for batch_id, batch in enumerate(train_loader):
            optimizer.zero_grad()

            imgs, label = batch

            imgs_device = imgs.to(device)
            label_device = label.to(device)

            output = model(imgs_device)
            specialised_metric.batch_forward( model=model )

            loss = loss_function(output, label_device)
            loss.backward()
            optimizer.step()
            specialised_metric.batch_backward(model)
            if verbosity>=3:
                print(loss)

            if verbosity==2 and (batch_id % print_every_n) == 0:
                # if verbosity >= 3:
                #     print(loss)
                # print('hallo')
                print("\r {:2}% done <> Current batch loss: {:1.4}".format( batch_id*100//epoch_length, loss.detach().float()), end='')


        test_acc = compute_accuracy(model, test_loader, device=device)
        epoch_test_accuracies[epoch_nr] = test_acc

        train_acc = compute_accuracy(model, train_loader, device=device)
        epoch_train_accuracies[epoch_nr] = train_acc
        if verbosity >= 1:
            print("Epoch {}>> Train Accuracy: {} | Test Accuracy: {}".format(epoch_nr, train_acc, test_acc))

            # print()
        specialised_metric.epoch(model)

    return model, dict(train_accuracy=epoch_train_accuracies, test_accuracy=epoch_test_accuracies, specialised_metric=specialised_metric.get_summary_dict())

if __name__=="__main__":
    model = nn.Sequential(*get_downsampling_layers(), ODEBlock(ConvolutionalDynamicsFunction(64, time_dependent=False)), *get_final_layers())
    train_model_cached(model, batch_size=128, epochs=10, verbosity=3, specialised_metric=ODEMetric())