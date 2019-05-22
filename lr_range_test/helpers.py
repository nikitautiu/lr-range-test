from typing import Optional, List, Dict

import ignite
import torch

from lr_range_test.lr_range import AutomaticLRRangeTest, InteractiveLRRangeTest
from lr_range_test.type_aliases import OptimizerType, DataLoaderType, LossFnType


def lr_range_test(optimizer: OptimizerType, model: torch.nn.Module,
                  train_loader: DataLoaderType, loss_fn: LossFnType,
                  eval_metric: Optional[ignite.metrics.Metric] = None,
                  test_loader: Optional[DataLoaderType] = None,
                  lr_min: float = 1e-7, lr_max: float = 1e1, num_steps: int = 50,
                  smooth_f: float = .05, diverge_th: float = 5.,
                  wd_values: Optional[List[float]] = None, pbar: bool = False,
                  automatic: bool = False, descending: bool = True,
                  device: str = 'cuda') -> Dict['str', float]:
    """
    Run the a lr range test in either an interactive or an automatic fashion as proposed by
    `Cyclical Learning Rates for Training Neural Networks <https://arxiv.org/pdf/1506.01186.pdf>`_.


    The function expects a ``model`` and ``optimizer`` for which to perform the test. This model
    is optimized with wrt. a loss function ``loss_fn``. The data is loaded from a given iterable
    (or a standard pytorch ``DataLoader``) called ``train_loader``. The loss will be calculated as
    the batch loss after each step if a ``test_loader`` is not specified. If ``test_loader`` is specified
    the loss is computed and averaged on the entirety of the test data.

    The learning rate of the model is varied from ``lr_min`` to ``lr_max`` exponentially over the course
    of ``num_steps`` iterations and smoothed with an exponential moving average with an alpha coefficient
    of ``smooth_f``. The training is stopped early if the loss diverges by a factor of more than ``diverge_th``
    from the best recorded loss.

    A custom evaluation metric such as accuracy can be specified with an
    `ignite metric <https://pytorch.org/ignite/metrics.html>`_. If the metric is expected to increase during training,
    (eg. accuracy) the ``descending`` parameter should be set to ``False``.

    Finally the test


    :param automatic: whether to perform an automatic lr range test or an interactive one
    :param model: A torch module receiving inputs and outputting predictions
    :param eval_metric: An ignite metric to use when evaluating the test_loader.
    :param optimizer: The optimizer to use for the LR range test.
    :param train_loader: An iterable to load data from and feed to the trainer.
    :param test_loader: An iterable to load data from and feed to the evaluator,
    :param loss_fn: An objective function taking outputs and predictions and returning a metric.
    :param device: the device to do the training/evaluation on (default: cuda)
    :param descending: whether the metric/loss chosen should descend or not (ie. accuracy should not)
    :param pbar: whether to print a progress bar during training
    :param wd_values: the weight decay values to test for
    :param diverge_th:  the coefficient by which the current metric must differ from the best recorded value
        to consider that the metric has diverged
    :param num_steps: the number of steps to increase LR over
    :param lr_max: the lr to end on
    :param lr_min: the lr to start from
    :param smooth_f: the alpha coefficient for the exponential moving average
    :return:
    """

    if automatic:
        tester = AutomaticLRRangeTest(optimizer=optimizer, model=model, loss_fn=loss_fn,
                                      train_loader=train_loader, test_loader=test_loader,
                                      eval_metric=eval_metric, descending=descending, device=device)
    else:
        tester = InteractiveLRRangeTest(optimizer=optimizer, model=model, loss_fn=loss_fn,
                                        train_loader=train_loader, test_loader=test_loader,
                                        eval_metric=eval_metric, descending=descending, device=device)
    return tester.run(lr_min=lr_min, lr_max=lr_max, num_steps=num_steps, smooth_f=smooth_f, diverge_th=diverge_th,
                      wd_values=wd_values, pbar=pbar)
