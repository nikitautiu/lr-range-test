import copy
from typing import Optional, Sequence

import ignite.engine
import numpy as np
import torch
import tqdm
from ignite.engine import Events

from lr_range_test.type_aliases import DataLoaderType, HistoryType


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

            # stop if loss diverges. this means that
            # either it grows more than a factor of diverge_th than the best loss if it's supposed to descend
            # or it drops diverge_th times
            if self.descending:
                diverging = loss > self.diverge_th * self.best_loss or np.isnan(loss)
            else:
                diverging = loss < self.best_loss / self.diverge_th or np.isnan(loss)
            if diverging:
                self.train_engine.terminate()
                if self.pbar is not None:
                    tqdm.tqdm.write('Stopping early, the loss has diverged')
                else:
                    print('Stopping early, the loss has diverged')
            else:
                # append it to the history
                self.history.append((self.lr_values[self.current_ind], loss))

            # update the learning rate
            if self.current_ind < len(self.lr_values) - 1:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr_values[self.current_ind]

            self.current_ind += 1  # update the iteration counter
            if self.pbar is not None:
                self.pbar.update()   # add a step to the progressbar
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
        self.pbar = None
        if pbar:
            self.pbar = tqdm.tqdm(total=num_steps)

        # set the initial learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr_values[0]

        # add the event to evaluate the learning rate
        train_engine.on(Events.ITERATION_COMPLETED)(self._train_step)
        train_engine.run(data=train_loader, max_epochs=(num_steps // len(self.train_loader)) + 1)
        train_engine.remove_event_handler(self._train_step, Events.ITERATION_COMPLETED)
        if pbar:
            self.pbar.close()  # close the progressbar
            self.pbar = None

        return self.history
