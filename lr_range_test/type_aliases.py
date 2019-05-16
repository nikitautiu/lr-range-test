from typing import Tuple, Optional, List, Iterable, Sized, Union, Callable

import ignite
import torch


# torch type alisases
class DataLoaderType(Sized, Iterable):
    pass


OptimizerType = torch.optim.Optimizer
EngineType = ignite.engine.Engine
OptimizerEngineLoaderTupleType = Tuple[
    OptimizerType, EngineType, DataLoaderType, Optional[EngineType], Optional[DataLoaderType]]
EngineOrModelType = Union[OptimizerType, torch.nn.Module]

# types for plotting
HistorySampleType = Tuple[float, float]
HistoryType = List[HistorySampleType]
PlotDataType = List[Tuple[HistoryType, float]]
LossFnType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
