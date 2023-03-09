import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd
from pylab import *


# load dataset
train_file = datasets.MNIST(
    root='./dataset/',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)
test_file = datasets.MNIST(
    root='./dataset/',
    train=False,
    transform=transforms.ToTensor()
)
train_data = train_file.data
train_targets = train_file.targets
test_data = test_file.data
test_targets = test_file.targets


# Defining the loss function
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, x, y, z, model):
        regularization_loss = 0
        for param in model.parameters():
                regularization_loss += torch.sum(torch.abs(param))
        mse_loss = torch.mean(torch.pow((x - y), 2)) + 0.1/torch.sum(torch.var(z.T,1,False)) + 0.0000001*regularization_loss
        return mse_loss


batch_size = 512

lv_number = 10  # number of the Latent Variables


class AutoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 392),
            nn.ReLU(inplace=True),
            nn.Linear(392, lv_number),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(lv_number, 392),
            nn.ReLU(inplace=True),
            nn.Linear(392, 784),
            nn.Sigmoid(),
        )
        # Xavier initialisation
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        Lv = self.encoder(x)
        x = self.decoder(Lv)
        return x, Lv


# number is the digit
for number in range(0, 10):
    # Data import and pre-processing
    train = train_data[np.where(train_targets == number)[0]] / 255
    test = test_data[np.where(test_targets == number)[0]] / 255

    train_set = np.array(np.float32(train)).reshape(train.shape[0], -1)
    test_set = np.array(np.float32(test)).reshape(test.shape[0], -1)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=False)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=False)

    device = torch.device('cpu')
    model = AutoEncoder().to(device)
    criteon = CustomLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    Mse_train = []
    Mse_test = []
    Var_train = []
    Var_test = []
    final_auc = []

    for epoch in range(400):
        for batch_idx, x in enumerate(train_loader):
            x = x.to(device)
            x_reconstruct, lv = model(x)
            loss = criteon(x_reconstruct, x, lv, model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Train
        train_set = torch.tensor(train_set, dtype=float).to(device)
        train_reconstruct, lv_train = model(torch.tensor(np.float32(train_set)))

        var_train = sum(np.var(lv_train.detach().numpy(), axis=0))
        Var_train.append(var_train.item())

        mse_train = torch.mean(torch.pow((train_reconstruct - train_set), 2))
        Mse_train.append(mse_train.item())

        # Test
        test_set = torch.tensor(test_set, dtype=float).to(device)
        test_reconstruct, lv_test = model(torch.tensor(np.float32(test_set)))

        var_test = sum(np.var(lv_test.detach().numpy(), axis=0))
        Var_test.append(var_test.item())
        mse_test = torch.mean(torch.pow((test_reconstruct - test_set), 2))
        Mse_test.append(mse_test.item())
        print(epoch, 'mse_train:', mse_train.item(), 'var_train', var_train, 'mse_test', mse_test.item(), 'var_test',
              var_test)

    result = pd.concat([pd.DataFrame(Mse_train), pd.DataFrame(Var_train),
                        pd.DataFrame(Mse_test), pd.DataFrame(Var_test)], axis=1, )
    result.columns = ['mse_train', 'var_train', 'mse_test', 'var_test']
    result.to_csv("******", header=True, index=False)
    torch.save(model.state_dict(), "******.pkl")