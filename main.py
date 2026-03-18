#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EDFM Simulator - Main Entry Point
ResInsight-Style Reservoir Simulation with Python + PyQt + VTK

Architecture:
    - front/: UI components and input handling
    - visual/: VTK visualization rendering
    - src/: C++ algorithm implementation
    - build/: Compiled C++ module
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from front.main_window import main

if __name__ == '__main__':
    main()
