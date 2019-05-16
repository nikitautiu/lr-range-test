from typing import Optional, List, Dict
from unittest import TestCase

import ignite
import torch
from ignite.metrics import Loss, Accuracy

from lr_range import ModelOrEngineLRRangeTest, LRFinderIgnite
from tests.utils import get_loaders, LogisticRegression


class DummyLRTester(ModelOrEngineLRRangeTest):
    def run(self, lr_min: float = 1e-7, lr_max: float = 1e1, num_steps: int = 50, smooth_f: float = .05,
            diverge_th: float = 5., initial_wd_values: Optional[List[float]] = None) -> Dict['str', float]:
        if initial_wd_values is None:
            initial_wd_values = [0.0]

        all_values = []
        lr_finder = LRFinderIgnite()
        for wd in initial_wd_values:
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

        train_loader, test_loader = get_loaders()
        sample_size = next(iter(test_loader))[0].shape[1]
        model = LogisticRegression(sample_size)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
        loss_fn = torch.nn.BCELoss()
        device = 'cuda'

        # create with default evaluator and no test set
        tester = DummyLRTester(optimizer=optimizer, model=model, loss_fn=loss_fn, train_loader=train_loader)
        all_values = tester.run()

        # create with default evaluator and given test set
        tester = DummyLRTester(optimizer=optimizer, model=model, loss_fn=loss_fn,
                               train_loader=train_loader, test_loader=test_loader)
        all_values = tester.run()

        # create with default evaluator, given test set and custom metric
        # this is binary classification so we cannot use logits
        tester = DummyLRTester(optimizer=optimizer, model=model, loss_fn=loss_fn,
                               eval_metric=Accuracy(output_transform=lambda x: (x[0] >= 0.5, x[1])),
                               train_loader=train_loader, test_loader=test_loader)
        all_values = tester.run()

        # create with custom trainer and no test set
        train_engine = ignite.engine.create_supervised_trainer(model=model, optimizer=optimizer,
                                                               loss_fn=loss_fn, device=device)
        tester = DummyLRTester(optimizer=optimizer, train_engine=train_engine,
                               train_loader=train_loader)
        all_values = tester.run()

        # create with custom trainer and default evaluator and given test set
        train_engine = ignite.engine.create_supervised_trainer(model=model, optimizer=optimizer,
                                                               loss_fn=loss_fn, device=device)
        tester = DummyLRTester(optimizer=optimizer, model=model, loss_fn=loss_fn, train_engine=train_engine,
                               train_loader=train_loader, test_loader=test_loader)
        all_values = tester.run()

        # create with custom trainer and default evaluator and given test set
        # and custom metric instead of loss function
        train_engine = ignite.engine.create_supervised_trainer(model=model, optimizer=optimizer,
                                                               loss_fn=loss_fn, device=device)
        tester = DummyLRTester(optimizer=optimizer, model=model, train_engine=train_engine,
                               eval_metric=Accuracy(output_transform=lambda x: (x[0] >= 0.5, x[1])),
                               train_loader=train_loader, test_loader=test_loader)
        all_values = tester.run()

        # create with custom trainer and custom evaluator
        train_engine = ignite.engine.create_supervised_trainer(model=model, optimizer=optimizer,
                                                               loss_fn=loss_fn, device=device)
        test_engine = ignite.engine.create_supervised_evaluator(model=model, metrics={'loss': Loss(loss_fn)},
                                                                device=device)
        tester = DummyLRTester(optimizer=optimizer, train_engine=train_engine, test_engine=test_engine,
                               train_loader=train_loader, test_loader=test_loader)
        all_values = tester.run()

        # create with custom tester and default trainer
        test_engine = ignite.engine.create_supervised_evaluator(model=model, metrics={'loss': Loss(loss_fn)},
                                                                device=device)
        tester = DummyLRTester(optimizer=optimizer, model=model, loss_fn=loss_fn, test_engine=test_engine,
                               train_loader=train_loader, test_loader=test_loader)
        all_values = tester.run()

    def test_invalid_testers(self):
        """Test whether the lr range test class properly fails initialization with bad parameters"""
        train_loader, test_loader = get_loaders()
        sample_size = next(iter(test_loader))[0].shape[1]
        model = LogisticRegression(sample_size)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
        loss_fn = torch.nn.BCELoss()
        device = 'cuda'

        # no model or train_engine
        with self.assertRaisesRegex(TypeError, r'.*(model|engine).*(model|engine).*'):
            tester = DummyLRTester(optimizer=optimizer, train_loader=train_loader)

        # no loss_fn
        with self.assertRaisesRegex(TypeError, r'.*loss_fn.*'):
            tester = DummyLRTester(optimizer=optimizer, model=model, train_loader=train_loader)

        # train_engine specified, but no model for default evaluator
        with self.assertRaisesRegex(TypeError, r'.*model.*'):
            train_engine = ignite.engine.create_supervised_trainer(model=model, optimizer=optimizer,
                                                                   loss_fn=loss_fn, device=device)
            tester = DummyLRTester(optimizer=optimizer, loss_fn=loss_fn, train_engine=train_engine,
                                   train_loader=train_loader, test_loader=test_loader)

        # train_engine specified, but no loss_fn or eval_metric for evaluator
        with self.assertRaisesRegex(TypeError, r'.*loss_fn.*'):
            train_engine = ignite.engine.create_supervised_trainer(model=model, optimizer=optimizer,
                                                                   loss_fn=loss_fn, device=device)
            tester = DummyLRTester(optimizer=optimizer, train_engine=train_engine,
                                   train_loader=train_loader, test_loader=test_loader)

        # NOW test for bad train_engine or test_engines which do not output a loss
        train_engine = ignite.engine.create_supervised_trainer(model=model, optimizer=optimizer,
                                                               loss_fn=loss_fn, device=device,
                                                               output_transform=lambda x, y, y_, loss: None)
        tester = DummyLRTester(optimizer=optimizer, train_engine=train_engine,
                               train_loader=train_loader)
        with self.assertRaisesRegex(TypeError, r'.*metrics.*'):
            tester.run()  # this should fail because we're returning no loss

        # this one should fail because we're using a test engine which has no loss metric
        train_engine = ignite.engine.create_supervised_trainer(model=model, optimizer=optimizer,
                                                               loss_fn=loss_fn, device=device)
        test_engine = ignite.engine.create_supervised_evaluator(model=model, device=device)
        tester = DummyLRTester(optimizer=optimizer, train_engine=train_engine, test_engine=test_engine,
                               train_loader=train_loader, test_loader=test_loader)
        with self.assertRaisesRegex(TypeError, r'.*metrics.*'):
            tester.run()  # this should fail because we're returning no loss
