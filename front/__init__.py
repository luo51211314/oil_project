"""
前端UI模块
"""
from .data_models import SimulationData, OutputCapture
from .input_panel import (GridInputPanel, WellsInputPanel, FracturesInputPanel, 
                          ResultsPanel, AlgorithmSelector, groupbox_style)
from .main_window import MainWindow

__all__ = [
    'SimulationData', 'OutputCapture',
    'GridInputPanel', 'WellsInputPanel', 'FracturesInputPanel', 'ResultsPanel', 'AlgorithmSelector',
    'groupbox_style', 'MainWindow'
]
