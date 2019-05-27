from typing import Optional, List, Dict
from unittest import TestCase

import ignite
import torch
from ignite.metrics import Loss, Accuracy

from lr_range_test.finder import LRFinderIgnite
from lr_range_test.lr_range import ModelOrEngineLRRangeTest, AutomaticLRRangeTest
from tests.utils import get_loaders, LogisticRegression, set_reproducible, DEVICE


class DummyLRTester(ModelOrEngineLRRangeTest):
    def run(self, lr_min: float = 1e-7, lr_max: float = 1e1, num_steps: int = 50, smooth_f: float = .05,
            diverge_th: float = 5., wd_values: Optional[List[float]] = None,
            pbar: bool = False) -> Dict['str', float]:
        if wd_values is None:
            wd_values = [0.0]

        all_values = []
        lr_finder = LRFinderIgnite()
        for wd in wd_values:
            # get the classes generated
            results = self.build_optimizer_trainers_loaders()
            optimizer, train_engine, train_loader, test_engine, test_loader = results

            # update the weight_decay
            for param_group in optimizer.param_groups:
                param_group['weight_decay'] = wd

            # do a lr finding run
            run = lr_finder.run(optimizer=optimizer, train_engine=train_engine, train_loader=train_loader,
                                test_engine=test_engine, test_loader=test_loader, lr_min=lr_min, lr_max=lr_max,
                                num_steps=num_steps, smooth_f=smooth_f, diverge_th=diverge_th, pbar=False)
            value_list = run
            all_values.append((list(value_list), wd))

        return all_values


class TestModelOrEngineLRRangeTest(TestCase):
    def test_valid_testers(self):
        """Test whether correct initialization works for the lr range test and
        whether the test runs"""
        # add reproducibility
        set_reproducible()

        train_loader, test_loader = get_loaders()
        sample_size = next(iter(test_loader))[0].shape[1]
        model = LogisticRegression(sample_size)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
        loss_fn = torch.nn.BCELoss()
        device = DEVICE

        # create with default evaluator and no test set
        tester = DummyLRTester(optimizer=optimizer, model=model, loss_fn=loss_fn, train_loader=train_loader,
                               device=device)
        all_values = tester.run()

        # create with default evaluator and given test set
        tester = DummyLRTester(optimizer=optimizer, model=model, loss_fn=loss_fn,
                               train_loader=train_loader, test_loader=test_loader, device=device)
        all_values = tester.run()

        # create with default evaluator, given test set and custom metric
        # this is binary classification so we cannot use logits
        tester = DummyLRTester(optimizer=optimizer, model=model, loss_fn=loss_fn,
                               eval_metric=Accuracy(output_transform=lambda x: (x[0] >= 0.5, x[1])),
                               train_loader=train_loader, test_loader=test_loader, descending=False, device=device)
        all_values = tester.run()

        # create with custom trainer and no test set
        train_engine = ignite.engine.create_supervised_trainer(model=model, optimizer=optimizer,
                                                               loss_fn=loss_fn, device=device)
        tester = DummyLRTester(model=model, optimizer=optimizer, train_engine=train_engine,
                               train_loader=train_loader, device=device)
        all_values = tester.run()

        # create with custom trainer and default evaluator and given test set
        train_engine = ignite.engine.create_supervised_trainer(model=model, optimizer=optimizer,
                                                               loss_fn=loss_fn, device=device)
        tester = DummyLRTester(optimizer=optimizer, model=model, loss_fn=loss_fn, train_engine=train_engine,
                               train_loader=train_loader, test_loader=test_loader, device=device)
        all_values = tester.run()

        # create with custom trainer and default evaluator and given test set
        # and custom metric instead of loss function
        train_engine = ignite.engine.create_supervised_trainer(model=model, optimizer=optimizer,
                                                               loss_fn=loss_fn, device=device)
        tester = DummyLRTester(optimizer=optimizer, model=model, train_engine=train_engine,
                               eval_metric=Accuracy(output_transform=lambda x: (x[0] >= 0.5, x[1])),
                               train_loader=train_loader, test_loader=test_loader,
                               descending=False, device=device)
        all_values = tester.run()

        # create with custom trainer and custom evaluator
        train_engine = ignite.engine.create_supervised_trainer(model=model, optimizer=optimizer,
                                                               loss_fn=loss_fn, device=device)
        test_engine = ignite.engine.create_supervised_evaluator(model=model, metrics={'loss': Loss(loss_fn)},
                                                                device=device)
        tester = DummyLRTester(model=model, optimizer=optimizer, train_engine=train_engine, test_engine=test_engine,
                               train_loader=train_loader, test_loader=test_loader, device=device)
        all_values = tester.run()

        # create with custom tester and default trainer
        test_engine = ignite.engine.create_supervised_evaluator(model=model, metrics={'loss': Loss(loss_fn)},
                                                                device=device)
        tester = DummyLRTester(optimizer=optimizer, model=model, loss_fn=loss_fn, test_engine=test_engine,
                               train_loader=train_loader, test_loader=test_loader, device=device)
        all_values = tester.run()

    def test_invalid_testers(self):
        """Test whether the lr range test class properly fails initialization with bad parameters"""
        # add reproducibility
        set_reproducible()

        train_loader, test_loader = get_loaders()
        sample_size = next(iter(test_loader))[0].shape[1]
        model = LogisticRegression(sample_size)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
        loss_fn = torch.nn.BCELoss()
        device = DEVICE

        # no model or train_engine
        with self.assertRaisesRegex(TypeError, r'.*(model|engine).*'):
            tester = DummyLRTester(optimizer=optimizer, train_loader=train_loader, device=device)

        # no loss_fn
        with self.assertRaisesRegex(TypeError, r'.*loss_fn.*'):
            tester = DummyLRTester(optimizer=optimizer, model=model, train_loader=train_loader, device=device)

        # train_engine specified, but no model for default evaluator
        with self.assertRaisesRegex(TypeError, r'.*model.*'):
            train_engine = ignite.engine.create_supervised_trainer(model=model, optimizer=optimizer,
                                                                   loss_fn=loss_fn, device=device)
            tester = DummyLRTester(optimizer=optimizer, loss_fn=loss_fn, train_engine=train_engine,
                                   train_loader=train_loader, test_loader=test_loader, device=device)

        # train_engine specified, but no loss_fn or eval_metric for evaluator
        with self.assertRaisesRegex(TypeError, r'.*loss_fn.*'):
            train_engine = ignite.engine.create_supervised_trainer(model=model, optimizer=optimizer,
                                                                   loss_fn=loss_fn, device=device)
            tester = DummyLRTester(optimizer=optimizer, model=model, train_engine=train_engine,
                                   train_loader=train_loader, test_loader=test_loader, device=device)

        # NOW test for bad train_engine or test_engines which do not output a loss
        train_engine = ignite.engine.create_supervised_trainer(model=model, optimizer=optimizer,
                                                               loss_fn=loss_fn, device=device,
                                                               output_transform=lambda x, y, y_, loss: None)
        tester = DummyLRTester(model=model, optimizer=optimizer, train_engine=train_engine,
                               train_loader=train_loader, device=device)
        with self.assertRaisesRegex(TypeError, r'.*metrics.*'):
            tester.run()  # this should fail because we're returning no loss

        # this one should fail because we're using a test engine which has no loss metric
        train_engine = ignite.engine.create_supervised_trainer(model=model, optimizer=optimizer,
                                                               loss_fn=loss_fn, device=device)
        test_engine = ignite.engine.create_supervised_evaluator(model=model, device=device)
        tester = DummyLRTester(model=model, optimizer=optimizer, train_engine=train_engine, test_engine=test_engine,
                               train_loader=train_loader, test_loader=test_loader, device=device)
        with self.assertRaisesRegex(TypeError, r'.*metrics.*'):
            tester.run()  # this should fail because we're returning no loss


class TestAutomaticLRRangeTest(TestCase):
    def test_automatic_lrrange(self):
        """Reproducible test for the automatic range finder."""
        device = DEVICE
        # add reproducibility
        set_reproducible()

        train_loader, test_loader = get_loaders()
        sample_size = next(iter(test_loader))[0].shape[1]
        model = LogisticRegression(sample_size)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
        loss_fn = torch.nn.BCELoss()

        set_reproducible()
        tester = AutomaticLRRangeTest(optimizer=optimizer, model=model, loss_fn=loss_fn,
                                      train_loader=train_loader, test_loader=test_loader, device=device)
        results = tester.run(num_steps=200)
        self.assertAlmostEqual(results['lr_max'], 0.1867, delta=1e-4)

        set_reproducible()
        tester = AutomaticLRRangeTest(optimizer=optimizer, model=model, loss_fn=loss_fn,
                                      eval_metric=Accuracy(output_transform=lambda x: (x[0] >= 0.5, x[1])),
                                      train_loader=train_loader, test_loader=test_loader, descending=False,
                                      device=device)
        results = tester.run(num_steps=200)
        self.assertAlmostEqual(results['lr_max'], 0.0674, delta=1e-4)

        set_reproducible()
        tester = AutomaticLRRangeTest(optimizer=optimizer, model=model, loss_fn=loss_fn,
                                      train_loader=train_loader, device=device)
        results = tester.run(num_steps=200)
        self.assertAlmostEqual(results['lr_max'], 0.3255, delta=1e-4)
