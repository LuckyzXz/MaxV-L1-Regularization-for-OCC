import torch
import torch.nn as nn
from torchvision import datasets,transforms
import pandas as pd
from pylab import *
from sklearn.metrics import roc_curve, auc
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

lv_number = 10

class AutoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784,392),
            nn.ReLU(inplace=True),
            nn.Linear(392,lv_number),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(lv_number,392),
            nn.ReLU(inplace=True),
            nn.Linear(392,784),
            nn.Sigmoid(),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
#                 print(1)
    def forward(self, x):
        Lv = self.encoder(x)
        x = self.decoder(Lv)
        return x, Lv

#dataset
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

device = torch.device('cpu')
model = AutoEncoder().to(device)

batch_size = 512
#number is the digit from 0 to 9
for number in range(0,10):
    final_auc = []
    # Calculate auc when each digit
    for negative_number in range(0,10):
        # load the model saved by training
        model.load_state_dict(torch.load("******.pkl")) # AE model with different regularization method can be loaded
        part_auc = []
        for positive_number in range(0,10):
            test = test_data[np.where((test_targets==negative_number)|(test_targets==positive_number))[0]]/255
            test_reconstruct,_ = model(test.reshape(test.shape[0],-1))
            truth = test_targets[np.where((test_targets==negative_number)|(test_targets==positive_number))[0]].detach().numpy()
            if positive_number == 0 and negative_number== 1:
                truth[truth==positive_number] = 2
                truth[truth==negative_number] = 0
                truth[truth==2] = 1
            elif positive_number == 0 and negative_number!= 1:
                truth[truth==positive_number] = 1
                truth[truth==negative_number] = 0
            else:
                truth[truth==negative_number] = 0
                truth[truth==positive_number] = 1
            prob = torch.sum(torch.pow((test_reconstruct-test.reshape(test.shape[0],-1)),2),1).detach().numpy()
            fpr, tpr, _ = roc_curve(truth, prob)     # ROC
            roc_auc = auc(fpr, tpr)   # Calculate auc
            part_auc.append(roc_auc)
        final_auc.append(part_auc)
    pd.DataFrame(final_auc).to_csv("******-auc.csv",header = True, index = False) #save the AUC result
