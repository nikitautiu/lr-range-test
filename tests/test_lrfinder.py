from unittest import TestCase

import torch
from ignite import ignite
from ignite.metrics import Loss

from lr_range_test.finder import LRFinderIgnite
from tests.utils import set_reproducible, LogisticRegression, get_loaders, DEVICE


class TestLRFinder(TestCase):
    def test_train_loss_finder(self):
        """Test whether th LR finder works"""
        device = DEVICE
        set_reproducible()  # set the seeds

        # get the datasets and loaders
        train_loader, test_loader = get_loaders()
        sample_size = next(iter(test_loader))[0].shape[1]

        # create engine
        lr_finder = LRFinderIgnite()  # get a lr finder
        model = LogisticRegression(sample_size)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
        loss_fn = torch.nn.BCELoss()
        train_engine = ignite.engine.create_supervised_trainer(model=model, optimizer=optimizer,
                                                               loss_fn=loss_fn, device=device)
        test_engine = ignite.engine.create_supervised_evaluator(model=model, metrics={'loss': Loss(loss_fn)},
                                                                device=device)

        # test train loss
        values = lr_finder.run(optimizer=optimizer, train_engine=train_engine, train_loader=train_loader,
                               lr_min=1e-7, lr_max=1e2, num_steps=200, pbar=False)

        # test on test dataset with given engine
        values = lr_finder.run(optimizer=optimizer, train_engine=train_engine, train_loader=train_loader,
                               test_engine=test_engine, test_loader=test_loader, lr_min=1e-7,
                               lr_max=1e2, num_steps=200, pbar=False)

        # test on test dataset without given engine
        values = lr_finder.run(optimizer=optimizer, train_engine=train_engine, train_loader=train_loader,
                               test_loader=test_loader, lr_min=1e-7, lr_max=1e2, num_steps=200, pbar=False)
