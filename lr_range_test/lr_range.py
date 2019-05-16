import copy
from typing import Sequence, List, Optional, Dict

import ignite
import ignite.contrib.handlers
import matplotlib
import numpy as np
import torch
from ignite.engine import Events
from ignite.metrics import Loss
from matplotlib import pyplot as plt, widgets as mwidgets
from matplotlib.gridspec import GridSpec

from lr_range_test.type_aliases import (OptimizerEngineLoaderTupleType, HistoryType, PlotDataType, OptimizerType,
                                        DataLoaderType, LossFnType)


class LRFinderIgnite(object):
    """LR finder using ignite engines. More flexible than simply using a model.
    Can support different training regimens, such s as unsupervised scenarios."""

    def _get_loss(self) -> float:
        if self.test_engine is not None:
            self.test_engine.run(self.train_loader)  # do test evaluation
            output = self.test_engine.state.metrics
        else:
            output = self.train_engine.state.output

        # try to fetch the loss element or consider it as as a single value
        try:
            loss = output['loss']
        except (KeyError, TypeError):
            loss = output  # if it doesn't have a 'loss' key, it means it should be a scalar

        if hasattr(loss, 'item'):
            loss = loss.item()  # convert to scalar if not already a scalar

        # finally check if we have a value
        if not np.isscalar(loss):
            raise TypeError('Either metrics or the output of the model must contain an element'
                            'called "loss" or the output must be a single scalar representing the loss.')
        return loss

    def _train_step(self, engine) -> None:
        # terminate. needed because we may have to incompletely consume the last epoch
        if self.current_ind < len(self.lr_values):
            # get the loss depending on whether we ha ve a test
            # set or not, from train_engine or test engine
            loss = self._get_loss()
            if self.current_ind == 0:
                self.best_loss = loss  # first iteration
            else:
                # update best loss recorded until now
                if self.descending and loss < self.best_loss:
                    self.best_loss = loss
                if not self.descending and loss > self.best_loss:
                    self.best_loss = loss
                if self.smooth_f > 0:
                    loss = self.smooth_f * loss + (1 - self.smooth_f) * self.history[-1][1]  # perform smoothing

            # append it to the history
            self.history.append((self.lr_values[self.current_ind], loss))

            # stop if loss diverges. this means that
            # either it grows more than a factor of diverge_th than the best loss if it's supposed to descend
            # or it drops diverge_th times
            if self.descending:
                diverging = loss > self.diverge_th * self.best_loss or np.isnan(self.history[-1][1])
            else:
                diverging = loss < self.best_loss / self.diverge_th or np.isnan(self.history[-1][1])
            if diverging:
                self.train_engine.terminate()
                print('Stopping early, the loss has diverged')

            # update the learning rate
            if self.current_ind < len(self.lr_values) - 1:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr_values[self.current_ind]

            self.current_ind += 1  # update the iteration counter
        else:
            self.train_engine.terminate()  # terminate after last epoch

    def run(self, optimizer: torch.optim.Optimizer, train_engine: ignite.engine.Engine,
            train_loader: DataLoaderType, lr_max: float, lr_min: float, num_steps: int,
            smooth_f: float = 0.05, diverge_th: float = 5., descending: bool = True,
            pbar: bool = True, test_engine: Optional[ignite.engine.Engine] = None,
            test_loader: Optional[Sequence] = None) -> HistoryType:
        # clone hte engines
        train_engine = copy.copy(train_engine)
        test_engine = copy.copy(test_engine)

        # get the model and loaders
        self.smooth_f = smooth_f
        self.diverge_th = diverge_th

        self.optimizer = optimizer
        self.test_engine = test_engine
        self.test_loader = test_loader
        self.train_engine = train_engine
        self.train_loader = train_loader
        self.descending = descending

        self.lr_values = np.geomspace(lr_min, lr_max, num=num_steps)
        self.current_ind = 0
        self.best_loss = None
        self.history = []

        # add progress bar
        if pbar:
            pbar_handler = ignite.contrib.handlers.ProgressBar(persist=False)
            pbar_handler.attach(train_engine)  # attach it

        # set the initial learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr_values[0]

        # add the event to evaluate the learning rate
        train_engine.on(Events.ITERATION_COMPLETED)(self._train_step)
        train_engine.run(data=train_loader, max_epochs=(num_steps // len(self.train_loader)) + 1)

        return self.history


class InteractiveLrPlot(object):
    def __init__(self, values: PlotDataType, lr_max: Optional[float] = None, suggest: bool = True):
        self.lr_min: Optional[float] = None
        self.lr_max: Optional[float] = None
        self.rerun_lr_min: Optional[float] = None
        self.rerun_lr_max: Optional[float] = None
        self.is_rerun: bool = False
        self.is_save: bool = False

        self.values: PlotDataType = values
        self.suggest: bool = suggest

        self.init_lr_max: float = lr_max

    def create_plot(self):
        for values, wd in self.values:
            x, y = list(zip(*sorted(values)))
            x, y = np.array(x), np.nan_to_num(np.array(y))
            self.plot_ax.set_xlabel('Learning rate')
            self.plot_ax.set_ylabel('Loss')
            self.plot_ax.plot(x, y, alpha=.6, label='wd={}'.format(wd))
            self.plot_ax.set_xscale('log')
            self.plot_ax.xaxis.set_major_formatter(matplotlib.ticker.LogFormatter())
            self.plot_ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10))
            self.plot_ax.grid(which='major', axis='x')
            self.plot_ax.grid(which='minor', linestyle=':', axis='x')

        # add the suggestion with the steepest descent
        if self.suggest:
            values, wd = self.values[-1]
            x, y = list(zip(*sorted(values)))
            x, y = np.array(x), np.nan_to_num(np.array(y))
            # compute gradients
            grads = np.gradient(y)
            best_x = x[np.argmin(grads)]
            self.plot_ax.axvline(x=best_x, linestyle=':', color='r')

        self.plot_ax.legend()

    def create_axes(self):
        self.fig, self.plot_ax = plt.subplots(1, 1, gridspec_kw={'top': 0.88})
        # define the axes
        grid = GridSpec(3, 2, left=0.20, right=0.45, bottom=0.92, top=0.98)
        self.plot_btn_ax = plt.subplot(grid[0, 1])
        self.plot_lr_min_ax = plt.subplot(grid[0, 0])
        self.plot_lr_max_ax = plt.subplot(grid[1, 0])
        self.plot_wd_ax = plt.subplot(grid[2, 0])

        grid = GridSpec(3, 2, left=0.55, right=0.8, bottom=0.92, top=0.98)
        self.btn_ax = plt.subplot(grid[0, 1])
        self.lr_min_ax = plt.subplot(grid[0, 0])
        self.lr_max_ax = plt.subplot(grid[1, 0])
        self.wd_ax = plt.subplot(grid[2, 0])

    def _on_interval_select(self, lr_min, lr_max):
        self.lr_min = lr_min
        self.lr_max = lr_max

        # update the input values
        self.lr_min_text = str(lr_min)
        self.lr_max_text = str(lr_max)
        # and the text
        self.lr_min_box.set_val(self.lr_min_text)
        self.lr_max_box.set_val(self.lr_max_text)

    def _on_rerun_click(self, args):
        try:
            new_lr_min = float(self.plot_lr_min_text)
            new_lr_max = float(self.plot_lr_max_text)
            wd = float(self.plot_wd_text)
        except ValueError as e:
            pass
        else:
            self.is_rerun = True
            self.rerun_lr_min = new_lr_min
            self.rerun_lr_max = new_lr_max
            self.rerun_wd = wd
            plt.close()

    def _on_plot_lr_min_change(self, text):
        """Rerun lr_min update"""
        self.plot_lr_min_text = text

    def _on_plot_lr_max_change(self, text):
        """Rerun lr_max update"""
        self.plot_lr_max_text = text

    def _on_plot_wd_change(self, text):
        """Rerun weight_decay update"""
        self.plot_wd_text = text

    def _on_lr_min_change(self, text):
        """lr_min update"""
        self.lr_min_text = text
        # persist the value immediately
        try:
            lr_min = float(self.lr_min_text)
        except ValueError as e:
            pass
        else:
            self.lr_min = lr_min

    def _on_lr_max_change(self, text):
        """lr_max update"""
        self.lr_max_text = text
        # persist the value immediately
        try:
            lr_max = float(self.lr_max_text)
        except ValueError as e:
            pass
        else:
            self.lr_max = lr_max

    def _on_wd_change(self, text):
        """weight decay update"""
        self.wd_text = text
        # persist the value immediately
        try:
            wd = float(self.wd_text)
        except ValueError as e:
            pass
        else:
            self.wd = wd

    def _on_save_click(self, args):
        try:
            lr_min = float(self.lr_min_text)
            lr_max = float(self.lr_max_text)
            wd = float(self.wd_text)
        except ValueError as e:
            pass
        else:
            self.is_rerun = False
            self.is_save = True
            self.lr_min = lr_min
            self.lr_max = lr_max
            self.wd = wd
            plt.close()

    def add_widgets(self):
        rect_props = dict(facecolor='blue', alpha=0.5)
        self.span = mwidgets.SpanSelector(self.plot_ax, self._on_interval_select, 'horizontal',
                                          rectprops=rect_props, useblit=True)

        all_lr, _ = list(zip(*sorted(self.values[-1][0])))
        init_lr_min = all_lr[0]
        init_lr_max = self.init_lr_max or all_lr[-1]
        init_wd = self.values[-1][1]

        # add the plot controls
        self.plot_lr_min_text: str = ''
        self.plot_lr_max_text: str = ''
        self.plot_lr_min_text = str(init_lr_min)
        self.plot_lr_max_text = str(init_lr_max)
        self.plot_wd_text = str(init_wd)
        self.rerun_wd = init_wd

        self.rerun_lr_min_box = mwidgets.TextBox(self.plot_lr_min_ax, label='LR min', initial=str(init_lr_min))
        self.rerun_lr_max_box = mwidgets.TextBox(self.plot_lr_max_ax, label='LR max', initial=str(init_lr_max))
        self.rerun_wd_box = mwidgets.TextBox(self.plot_wd_ax, label='Weight decay', initial=str(init_wd))
        self.rerun_btn = mwidgets.Button(self.plot_btn_ax, label='PLOT')

        # add event handling
        self.rerun_lr_min_box.on_text_change(self._on_plot_lr_min_change)
        self.rerun_lr_max_box.on_text_change(self._on_plot_lr_max_change)
        self.rerun_wd_box.on_text_change(self._on_plot_wd_change)
        self.rerun_btn.on_clicked(self._on_rerun_click)

        # add the value input controls
        self.lr_min_text = ''
        self.lr_max_text = ''
        self.wd_text = str(init_wd)

        self.lr_min_box = mwidgets.TextBox(self.lr_min_ax, label='LR min')
        self.lr_max_box = mwidgets.TextBox(self.lr_max_ax, label='LR max')
        self.wd_box = mwidgets.TextBox(self.wd_ax, label='Weight decay', initial=str(init_wd))
        self.save_btn = mwidgets.Button(self.btn_ax, label='SAVE')

        # add event handling
        self.rerun_lr_min_box.on_text_change(self._on_lr_min_change)
        self.rerun_lr_max_box.on_text_change(self._on_lr_max_change)
        self.rerun_wd_box.on_text_change(self._on_wd_change)
        self.save_btn.on_clicked(self._on_save_click)

    def run(self):
        self.create_axes()
        self.create_plot()
        self.add_widgets()
        plt.show()


class BaseLRRangeTest(object):
    def build_optimizer_trainers_loaders(self) -> OptimizerEngineLoaderTupleType:
        """Method that builds the optimizer, the engines and data loaders.
        Should be overriden by any subclass."""
        raise NotImplementedError

    def run(self, lr_min: float = 1e-7, lr_max: float = 1e1, num_steps: int = 50,
            smooth_f: float = .05, diverge_th: float = 5.,
            initial_wd_values: Optional[List[float]] = None) -> Dict['str', float]:
        raise NotImplementedError


class ModelOrEngineLRRangeTest(BaseLRRangeTest):
    def __init__(self, optimizer: OptimizerType, train_loader: DataLoaderType,
                 model: Optional[torch.nn.Module] = None,
                 train_engine: Optional[ignite.engine.Engine] = None,
                 test_engine: Optional[ignite.engine.Engine] = None, test_loader: Optional[DataLoaderType] = None,
                 loss_fn: Optional[LossFnType] = None, eval_metric: Optional[ignite.metrics.Metric] = None,
                 descending: bool = True, device: str = 'cuda') -> None:
        """
        An LR range test base class that simplifies initialization of the engines Provides several
        parameters setups in which it can be run. For some of these, the class automatically
        creates trainers and evaluators, and for others, the users can provide their own.

        Allowed parameter combinations:
        * (model, optimizer, loss_fn, train_loader) - The data is taken from `train_loader` and `model` is trained
            with `optimizer`. The loss value is taken at the end of each iteration from the output of the trainer.
            The output should have a key called 'loss'
        * (model, optimizer, loss_fn, train_loader, test_loader) - The same a the last one, but the loss is computed
            with data from the `test_loader` after each iteration of `train_loader`.
        * (model, optimizer, loss_fn, test_engine, train_loader, test_loader) - A default trainer will be created, but
            `test_engine` will be used as an evaluator.
        * (train_engine, optimizer, train_loader) - The `optimizer` should belong to the `train_engine`. The loss
            is taken from the output of the train_engine.
        * (train_engine, model, optimizer, train_loader, test_loader) - A default evaluator is build using
            `model` and `loss_fn` and the loss is computed on the test set.
        * (train_engine, test_engine, train_loader, test_loader) - Computes the loss using `test_engine` on data
            from `test_loader`.

        Either `model` or `train_engine` has to be specified. If `model` is specified, so should be `loss_fn`.
        `loss_fn` abd `model` also have to be specified if using a `test_loader` but relying on the default
        evaluator instead of a custom `test_engine` and so does.

        The train engine should have key called 'loss' in the output. The test engine should have a metric
        called `loss` if used.

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

        super().__init__()
        self.descending = descending
        self.optimizer: OptimizerType = optimizer
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

        # if the test loader is present, then we need an engine for training
        if test_loader is not None:
            # test engine is needed only if we have a test loader
            if test_engine is None:
                # create a default test engine with the loss fn as output
                # use the given eval_metric if provided, but fallback
                # to using the loss averaged over the entire epoch
                if eval_metric is None:
                    if loss_fn is None:
                        # error if no metric or loss_fn
                        raise TypeError('loss_fn has to be provided if using the default evaluator and not '
                                        'providing a metric')
                    eval_metric = Loss(loss_fn)

                if model is None:
                    raise TypeError('model must be provided if using the default evaluator')

                # create the test engine
                self.test_engine = ignite.engine.create_supervised_evaluator(model,
                                                                             metrics={'loss': eval_metric},
                                                                             device=device, non_blocking=True)
            else:
                self.test_engine = test_engine  # use the specified engine
        else:
            self.test_engine = None  # no need for a test engine if no test loader specified

    def build_optimizer_trainers_loaders(self) -> OptimizerEngineLoaderTupleType:
        return self.optimizer, self.train_engine, self.train_loader, self.test_engine, self.test_loader


class InteractiveLRRangeTest(ModelOrEngineLRRangeTest):
    def run(self, lr_min: float = 1e-7, lr_max: float = 1e1, num_steps: int = 50,
            smooth_f: float = .05, diverge_th: float = 5.,
            initial_wd_values: Optional[List[float]] = None, pbar: bool = False) -> Dict['str', float]:
        """Perform an interactive  LR range test. The method constructs the loss plots for the given
        weight decays in the interval specified delimited by `lr_min` and `lr_max` for `num_steps` of
        incrementation. Ifn o weight decay values are specified, it uses none.

        The plots are smoothed with and exponential moving average with a coefficient of `smooth_f`. The
        training will stop prematurely if the smoothed loss becomes `diverge_th` times larger
        than the best loss encountered.

        The best interval is selected from the plot by dragging. On exit, the last interval selected is returned
        alongside the last entered weight decay value. The plot can be rerun with different values of `lr_min, `lr_max`,
        `rerun_wd` and `num_steps`.
        """

        # initialize the rerun_wd values
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
                                num_steps=num_steps, smooth_f=smooth_f, diverge_th=diverge_th,
                                descending=self.descending,
                                pbar=pbar)
            value_list = run
            all_values.append((list(value_list), wd))

        # rerun if use click 'PLOT'
        ok = False
        wd = initial_wd_values[-1]
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
                results = self.build_optimizer_trainers_loaders()
                optimizer, train_engine, train_loader, test_engine, test_loader = results

                # update the weight_decay
                for param_group in optimizer.param_groups:
                    param_group['weight_decay'] = wd

                # do a lr finding run
                value_list = lr_finder.run(optimizer=optimizer, train_engine=train_engine, train_loader=train_loader,
                                           test_engine=test_engine, test_loader=test_loader, lr_min=lr_min,
                                           lr_max=lr_max,
                                           num_steps=num_steps, smooth_f=smooth_f, diverge_th=diverge_th)
                all_values.append((list(value_list), wd))

        return {'lr_min': plot.lr_min, 'lr_max': plot.lr_max, 'weight_decay': wd}


class AutomaticLRRangeTest(ModelOrEngineLRRangeTest):
    """Range test class that automatically selects the best values for the minimum and maximum LR
    and weight decay values, based on the approximate gradient of the loss with respect to the learning rate."""

    def run(self, lr_min: float = 1e-7, lr_max: float = 1e1, num_steps: int = 50,
            smooth_f: float = .05, diverge_th: float = 5.,
            wd_values: Optional[List[float]] = None, mult_f: float = 15.,
            pbar: bool = False) -> Dict['str', float]:
        """Similar to the interactive test, but the values for lr are selected automatically.
        The maximum lr is selected as the steepest descent value of the smoothed value plot.
        The best weight decay is selected as being the one whose steepest descent point is the greatest.

        :param pbar: whether to print a progress bar during training
        :param mult_f: what fraction of the best lr should the lowest lr be
        :param wd_values: the weight decay values to test for
        :param diverge_th:  the coefficient by which the current metric must differ from the best recorded value
            to consider that the metric has diverged
        :param num_steps: the number of steps to run annealing on
        :param lr_max: the lr to end on
        :param lr_min: the lr to start from
        :param smooth_f: the alpha coefficient for exponentially weighted average
        :param descending: whether the metric is expected to descend or not (ie. for accuracy should be false)
        """

        # initialize the rerun_wd values
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
        lr_min = lr_max / mult_f  # divide the max_lr by a factor of mult_f

        return {'lr_min': lr_min, 'lr_max': lr_max, 'weight_decay': wd}
