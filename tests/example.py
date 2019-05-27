import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from lr_range_test import lr_range_test


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def main():
    # training settings
    batch_size = 64
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 16, 'pin_memory': True} if use_cuda else {}

    # create the loader for MNIST data
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    # define model, optimizer and loss
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.NLLLoss()

    matplotlib.use('TkAgg')  # this is needed so we are able to interact with the plot
    result = lr_range_test(model=model, optimizer=optimizer, loss_fn=loss_fn,
                           lr_min=1e-7, lr_max=1e1, train_loader=train_loader,
                           num_steps=200, automatic=False, pbar=True, wd_values=[0.0, 1e-6])
    print(result)


if __name__ == '__main__':
    main()
