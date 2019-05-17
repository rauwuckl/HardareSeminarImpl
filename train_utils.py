import torch
import os

from networks import *
import numpy as np


import torchvision.datasets as datasets
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

            correct = np.count_nonzero(prediction == label.to(device))

            total_correct += correct
            total += len(label)

        model.train()

    return total_correct/total

def train_model_cached(model, file_path=None, **kwargs):
    if file_path and os.path.isfile(file_path):
        saved = torch.load(file_path)
        restored_args = saved['args']

        if restored_args != kwargs:
            print("some Parameters were different")
            # raise ValueError("For the given file_path, there already exists a cached network, that was produced with different parameters")

        state_dict = saved['state_dict']

        model.load_state_dict(state_dict) # now containing parameters
        return model, saved['accuracies']

    else:
        model, accuracies = train_model(model, **kwargs)

        if file_path is None:
            return model, accuracies
        else:
            to_save = {'state_dict': model.state_dict(), 'args': kwargs, 'accuracies': accuracies }


def train_model(model, batch_size, epochs, test_batch_size = None, device="cpu", learning_rate=0.1, loss_op = nn.CrossEntropyLoss, verbosity=1, n_prints_per_epoch=100):

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
        for batch_id, batch in enumerate(test_loader):
            optimizer.zero_grad()

            imgs, label = batch

            imgs_device = imgs.to(device)
            label_device = label.to(device)

            output = model(imgs_device)
            loss = loss_function(output, label_device)
            loss.backward()
            optimizer.step()

            if verbosity>=3:
                print(loss)

            if verbosity>=2 and (batch_id % print_every_n) == 0:
                # if verbosity >= 3:
                #     print(loss)
                # print('hallo')
                print("\r {:2}% done <> Current batch loss: {:1.4}".format( batch_id*100//epoch_length, loss.detach().float()), end='')


        test_acc = compute_accuracy(model, test_loader, device=device)
        epoch_test_accuracies[epoch_nr] = test_acc

        train_acc = compute_accuracy(model, train_loader)
        epoch_train_accuracies[epoch_nr] = train_acc
        if verbosity >= 1:
            print("Epoch {}>> Train Accuracy: {} | Test Accuracy: {}")

            # print()

    return model, dict(train=epoch_train_accuracies, test=epoch_test_accuracies)

if __name__=="__main__":
    model = nn.Sequential(*get_downsampling_layers(),*get_residual_blocks(1), *get_final_layers())
    train_model_cached(model, batch_size=128, epochs=10, verbosity=3)