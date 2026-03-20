"""
前端UI模块
"""
from .data_models import SimulationData, OutputCapture
from .input_panel import (GridInputPanel, WellsInputPanel, FracturesInputPanel, 
                          ResultsPanel, AlgorithmSelector, groupbox_style)

__all__ = [
    'SimulationData', 'OutputCapture',
    'GridInputPanel', 'WellsInputPanel', 'FracturesInputPanel', 'ResultsPanel', 'AlgorithmSelector',
    'groupbox_style', 'MainWindow'
]


def __getattr__(name):
    if name == 'MainWindow':
        from .main_window import MainWindow
        return MainWindow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
