from typing import Optional

import matplotlib
import numpy as np
from matplotlib import pyplot as plt, widgets as mwidgets
from matplotlib.gridspec import GridSpec

from type_aliases import PlotDataType


class InteractiveLrPlot(object):
    """Interactive plot class which plots metric-over-LR curves for different weight decay values. Allows the user
    to input new values to use for the plot and to input desired values for the LR range."""

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
