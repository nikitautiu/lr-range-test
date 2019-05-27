.. lr-range-test documentation master file, created by
   sphinx-quickstart on Wed May 22 16:44:17 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

lr-range-test's documentation
=========================================
.. toctree::
   :hidden:
   :maxdepth: 3

   index

This is a library for LR range tuning, implementing the method proposed in
`Cyclical Learning Rates for Training Neural Networks <https://arxiv.org/pdf/1506.01186.pdf>`_. It can
be used with any combination of pytorch models and optimizers and supports searching for good  values of weight decay.

Usage
-----
Although the library provides a lower-level interface through the
:class:`lr_range_test.lr_range.InteractiveLRRangeTest` and
:class:`lr_range_test.lr_range.AutomaticLRRangeTest` classes, a simpler and easier to use interface is provided
via :meth:`lr_range_test.lr_range_test`.


Sample usage for LR values between 1e-7 and 1e1. The LR is varied over the course of 200 steps and the test is ran
2 times, with two different values of weight decay.
::
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
    # run the training with a progress bar for 200 steps, with values of the
    result = lr_range_test(model=model, optimizer=optimizer, loss_fn=loss_fn,
                           lr_min=1e-7, lr_max=1e1, train_loader=train_loader,
                           num_steps=200, automatic=False, pbar=True, wd_values=[0.0, 1e-6])
    print(result)


For automatic mode simply change thq ``automatic`` flag to ``True``.
::
   result = lr_range_test(model=model, optimizer=optimizer, loss_fn=loss_fn,
                              lr_min=1e-7, lr_max=1e1, train_loader=train_loader,
                              num_steps=200, automatic=True, pbar=True, wd_values=[0.0, 1e-6])



Interactive mode
^^^^^^^^^^^^^^^^

In the interactive fashion, the loss is plotted with respect to the learning rate.
A vertical line is drawn tpo indicate the point of steepest improvement  in the metric.
The user can then drag and select an interval for the desired learning rate which will be inputted
in the textboxes above the plot.

If the user wants to redo the plot with different minimum/maximum LR values, or with a different value for
weight decay, they can use the boxes in the top-left of the corner to input these values and click
**PLOT**. After they have selected satisfactory values , they can return those values using the
**SAVE** button.

.. note::
   The matplotlib backend should be set to an interactive one in order for interactive mode
   to work (ie. ``TkAgg``) To do this, simply use ``matplotlib.use('TkAgg')`` before calling the lr test function.


Automatic mode
^^^^^^^^^^^^^^
In this mode, no plot is displayed and the ``lr_max`` value returned by the function is
the value corresponding to the steepest improvement in the metric used. This is equivalent
to the x coordinate of the red line displayed in interactive mode.

If multiple weight decay values are used, the one for which the optimal LR value is the greatest,
is returned.


API
---
Simple
^^^^^^
.. automodule:: lr_range_test
   :members:

Low-level
^^^^^^^^^
.. autoclass:: lr_range_test.lr_range.ModelOrEngineLRRangeTest
.. autoclass:: lr_range_test.lr_range.InteractiveLRRangeTest
   :show-inheritance:
   :members:
.. autoclass:: lr_range_test.lr_range.AutomaticLRRangeTest
   :show-inheritance:
   :members:


