import copy
from typing import List, Optional, Dict

import ignite
import ignite.contrib.handlers
import numpy as np
import torch
from ignite.metrics import Loss

from lr_range_test.finder import LRFinderIgnite
from lr_range_test.plot import InteractiveLrPlot
from lr_range_test.type_aliases import (OptimizerEngineLoaderTupleType, OptimizerType,
                                        DataLoaderType, LossFnType)


class BaseLRRangeTest(object):
    def build_optimizer_trainers_loaders(self) -> OptimizerEngineLoaderTupleType:
        """Method that builds the optimizer, the engines and data loaders.
        Should be overriden by any subclass."""
        raise NotImplementedError

    def run(self, lr_min: float = 1e-7, lr_max: float = 1e1, num_steps: int = 50,
            smooth_f: float = .05, diverge_th: float = 5.,
            wd_values: Optional[List[float]] = None) -> Dict['str', float]:
        raise NotImplementedError


class ModelOrEngineLRRangeTest(BaseLRRangeTest):
    """An LR range test base class that simplifies initialization of the engines Provides several
    parameters setups in which it can be run. For some of these, the class automatically
    creates trainers and evaluators, and for others, the users can provide their own.

    Allowed parameter combinations:

    (model, optimizer, loss_fn, train_loader)
        The data is taken from `train_loader` and ``model`` is trained
        with ``optimizer``. The loss value is taken at the end of each iteration from the output of the trainer.
        The output should have a key called "loss".

    (model, optimizer, loss_fn, train_loader, test_loader)
        The same a the last one, but the loss is computed
        with data from the ``test_loader`` after each iteration of ``train_loader``.

    (model, optimizer, loss_fn, test_engine, train_loader, test_loader) -
        A default trainer will be created, but ``test_engine`` will be used as an evaluator.

    (model, train_engine, optimizer, train_loader)
        The ``optimizer`` should belong to the ``train_engine``. The loss is taken
        from the output of the ``train_engine``.

    (model, train_engine, model, optimizer, train_loader, test_loader)
        A default evaluator is build using ``model`` and ``loss_fn`` and the loss is computed on the test set.

    (model, train_engine, test_engine, train_loader, test_loader)
        Computes the loss using ``test_engine`` on data from ``test_loader``.

    The model and the optimizer always have to be specified, even if they are trained by proxy using a train engine.
    This is because they have to be reset at the end of the run and when restarting the run for a new plot (in the
    interactive case).

    The train engine should have key called "loss" in the output. The test engine should have a metric
    called "loss" if used. (*Note*: The metric does not have to necessarily represent "the loss". I can be accuracy or
    anything the user wants to use). If using the default tests engine, ``loss_fn`` will be used as a metric.
    However, any loss will be overriden, if ``eval_metric`` is provided.

    :param eval_metric: An ignite metric to use when evaluating the test_loader.
    :param optimizer: The optimizer to use for the LR range test.
    :param train_loader: An iterable to load data from and feed to the trainer.
    :param model: A torch module receiving inputs and outputting predictions
    :param train_engine: An alternative to `model`. Used for training. Must output 'loss'.
    :param test_engine: An alternative to the default evaluator. Must output a metric called 'loss'
    :param test_loader: An iterable to load data from and feed to the evaluator,
    :param loss_fn: An objective function taking outputs and predictions and returning a metric.
    :param device: the device to do the training/evaluation on (default: cuda)
    :param descending: whether the metric/loss chosen should descend or not (ie. accuracy should not)
    """

    def __init__(self, optimizer: OptimizerType, train_loader: DataLoaderType,
                 model: torch.nn.Module,
                 train_engine: Optional[ignite.engine.Engine] = None,
                 test_engine: Optional[ignite.engine.Engine] = None, test_loader: Optional[DataLoaderType] = None,
                 loss_fn: Optional[LossFnType] = None, eval_metric: Optional[ignite.metrics.Metric] = None,
                 descending: bool = True, device: str = 'cuda') -> None:

        super().__init__()
        self.descending = descending
        self.optimizer: OptimizerType = optimizer
        self.model: Optional[torch.nn.Module] = model
        self.train_engine: ignite.engine.Engine
        self.train_loader: DataLoaderType = train_loader
        self.test_loader: Optional[DataLoaderType] = test_loader
        self.test_engine: Optional[ignite.engine.Engine]

        # create the train engine if necessary
        # if so, build it from  the model and loss_fn
        if train_engine is None and model is None:
            raise TypeError('either train_engine or model have to be provided')
        if train_engine is not None:
            self.train_engine = train_engine  # directly use it
        elif model is not None:
            if loss_fn is None:
                raise TypeError('loss_fn has to be provided if passing a plain pytorch model')
            self.train_engine = ignite.engine.create_supervised_trainer(model, optimizer,
                                                                        loss_fn=loss_fn, device=device,
                                                                        non_blocking=True)

        # get the metric to use
        new_metric = None
        if eval_metric is not None:
            new_metric = eval_metric
        elif loss_fn is not None:
            # use the given eval_metric if provided, but fallback
            # to using the loss averaged over the entire epoch
            new_metric = Loss(loss_fn)

        # if the test loader is present, then we need an engine for training
        if test_loader is not None:
            # test engine is needed only if we have a test loader
            if test_engine is None:
                if eval_metric is None:
                    if loss_fn is None:
                        # error if no metric or loss_fn
                        raise TypeError('loss_fn has to be provided if using the default evaluator and not '
                                        'providing a metric')
                if model is None:
                    raise TypeError('model must be provided if using the default evaluator')

                # create a default test engine
                self.test_engine = ignite.engine.create_supervised_evaluator(model,
                                                                             metrics={'loss': new_metric},
                                                                             device=device, non_blocking=True)
            else:
                self.test_engine = test_engine  # use the specified engine
                # attach a new metric if present
                if new_metric is not None:
                    new_metric.attach(self.test_engine, 'loss')
        else:
            self.test_engine = None  # no need for a test engine if no test loader specified

    def run(self, lr_min: float = 1e-7, lr_max: float = 1e1, num_steps: int = 50, smooth_f: float = .05,
            diverge_th: float = 5., wd_values: Optional[List[float]] = None) -> Dict['str', float]:
        raise NotImplementedError  # this serves as a base class and doesn't implement run

    def build_optimizer_trainers_loaders(self) -> OptimizerEngineLoaderTupleType:
        return self.optimizer, self.train_engine, self.train_loader, self.test_engine, self.test_loader

    def save_optimizer_and_model(self):
        """Persist the optimizer and model data so we can restore it later."""
        self.optimizer_state_dict = copy.deepcopy(self.optimizer.state_dict())
        self.model_state_dict = copy.deepcopy(self.model.state_dict())

    def restore_optimizer_and_model(self):
        """Restore the model and optimizer given as arguments to the previously-saved state"""
        self.optimizer.load_state_dict(self.optimizer_state_dict)
        self.model.load_state_dict(self.model_state_dict)


class InteractiveLRRangeTest(ModelOrEngineLRRangeTest):
    def run(self, lr_min: float = 1e-7, lr_max: float = 1e1, num_steps: int = 50,
            smooth_f: float = .05, diverge_th: float = 5.,
            wd_values: Optional[List[float]] = None, pbar: bool = False) -> Dict['str', float]:
        """Perform an interactive  LR range test. The method constructs the loss plots for the given
        weight decays in the interval specified delimited by `lr_min` and `lr_max`. The lr is incremented exponentially
        over `num_steps` iterations. If no weight decay values are specified, the model will not use any,

        The plots are smoothed with and exponential moving average with an alpha of `smooth_f`. The
        training will stop prematurely if the smoothed metric worsens by a factor of at least `diverge_th` compared
        to the best metric recorded until now.

        The best interval is selected from the plot by dragging. On exit, the last interval selected is returned
        alongside the last entered weight decay value. The plot can be rerun with different values of `lr_min`, `lr_max`,
        `wd` and `num_steps` using the "PLOT" inputs.
        """
        self.save_optimizer_and_model()
        results = self.build_optimizer_trainers_loaders()
        optimizer, train_engine, train_loader, test_engine, test_loader = results

        # initialize the rerun_wd values
        if wd_values is None:
            wd_values = [0.0]

        all_values = []
        for wd in wd_values:
            # get the classes generated
            self.restore_optimizer_and_model()
            # update the weight_decay
            for param_group in optimizer.param_groups:
                param_group['weight_decay'] = wd

            # do a lr finding run
            lr_finder = LRFinderIgnite()
            run = lr_finder.run(optimizer=optimizer, train_engine=train_engine, train_loader=train_loader,
                                test_engine=test_engine, test_loader=test_loader, lr_min=lr_min, lr_max=lr_max,
                                num_steps=num_steps, smooth_f=smooth_f, diverge_th=diverge_th,
                                descending=self.descending, pbar=pbar)
            value_list = run
            all_values.append((list(value_list), wd))

        # rerun if use click 'PLOT'
        ok = False
        wd = wd_values[-1]
        plot = None
        while not ok:
            plot = InteractiveLrPlot(all_values, lr_max=lr_max)
            plot.run()
            ok = True
            if plot.is_rerun:
                ok = False
                lr_min = plot.rerun_lr_min
                lr_max = plot.rerun_lr_max
                wd = plot.rerun_wd
                lr_min, lr_max = min(lr_min, lr_max), max(lr_min, lr_max)

                # get the classes generated
                self.restore_optimizer_and_model()
                # update the weight_decay
                for param_group in optimizer.param_groups:
                    param_group['weight_decay'] = wd

                # do a lr finding run
                lr_finder = LRFinderIgnite()
                value_list = lr_finder.run(optimizer=optimizer, train_engine=train_engine, train_loader=train_loader,
                                           test_engine=test_engine, test_loader=test_loader, lr_min=lr_min,
                                           lr_max=lr_max, num_steps=num_steps, smooth_f=smooth_f,
                                           diverge_th=diverge_th, pbar=pbar)
                all_values.append((list(value_list), wd))

        self.restore_optimizer_and_model()
        return {'lr_min': plot.lr_min, 'lr_max': plot.lr_max, 'weight_decay': wd}


class AutomaticLRRangeTest(ModelOrEngineLRRangeTest):
    """Range test class that automatically selects the best values for the minimum and maximum LR
    and weight decay values, based on the approximate gradient of the loss with respect to the learning rate."""

    def run(self, lr_min: float = 1e-7, lr_max: float = 1e1, num_steps: int = 50,
            smooth_f: float = .05, diverge_th: float = 5.,
            wd_values: Optional[List[float]] = None,
            pbar: bool = False) -> Dict['str', float]:
        """Similar to the interactive test, but the values for lr are selected automatically.
        The maximum lr is selected as the steepest improvement value of the smoothed metric plot.
        The best weight decay is selected as the weight decay value for which the steepest improvement
        occurs at the greatest LR value.

        :param pbar: whether to print a progress bar during training
        :param wd_values: the weight decay values to test for
        :param diverge_th:  the coefficient by which the current metric must differ from the best recorded value
            to consider that the metric has diverged
        :param num_steps: the number of steps to increase LR over
        :param lr_max: the lr to end on
        :param lr_min: the lr to start from
        :param smooth_f: the alpha coefficient for the exponential moving average
        """
        # get the classes generated

        results = self.build_optimizer_trainers_loaders()
        optimizer, train_engine, train_loader, test_engine, test_loader = results

        # initialize the rerun_wd values
        if wd_values is None:
            wd_values = [0.0]

        all_values = []
        for wd in wd_values:
            self.restore_optimizer_and_model()
            # update the weight_decay
            for param_group in optimizer.param_groups:
                param_group['weight_decay'] = wd

            # do a lr finding run
            lr_finder = LRFinderIgnite()
            run = lr_finder.run(optimizer=optimizer, train_engine=train_engine, train_loader=train_loader,
                                test_engine=test_engine, test_loader=test_loader, lr_min=lr_min, lr_max=lr_max,
                                num_steps=num_steps, smooth_f=smooth_f, diverge_th=diverge_th,
                                descending=self.descending,
                                pbar=pbar)
            value_list = run
            all_values.append((list(value_list), wd))

        # find the best values for lr and weight decay
        best_values = []
        for values in all_values:
            values, wd = values
            x, y = list(zip(*sorted(values)))
            x, y = np.array(x), np.nan_to_num(np.array(y))
            # compute gradients
            grads = np.gradient(y)

            # find wither the steepest descent or ascent
            if self.descending:
                best_x = x[np.argmin(grads)]
            else:
                best_x = x[np.argmax(grads)]
            best_values.append((best_x, wd))

        # get teh weight decay with the largest learning rate corresponding to
        # its steepest descent
        best_values = sorted(best_values, reverse=self.descending)
        lr_max, wd = best_values[0]

        self.restore_optimizer_and_model()
        return {'lr_max': lr_max, 'weight_decay': wd}
