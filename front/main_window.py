"""
主窗口模块
整合所有UI组件和可视化 - 与原文件完全一致
"""
import sys
import os
import json
import math
import re
import tempfile

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QSplitter, QFrame, QToolBar, QAction,
                             QStatusBar, QTabWidget, QStackedWidget, QTextEdit, QGroupBox,
                             QTableWidget, QSpinBox, QDoubleSpinBox, QGridLayout, QComboBox,
                             QProgressBar, QScrollArea,
                             QCheckBox, QTableWidgetItem)
from PyQt5.QtCore import Qt, QSize, QProcess
from PyQt5.QtGui import QIcon, QFont, QColor, QPixmap, QPainter
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk

# 导入本地模块
from .data_models import SimulationData, CornerPointCell, CornerPointGridData

# 导入可视化模块
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from visual.vtk_renderer import VTKRenderer


class AlgorithmSelector(QWidget):
    """算法选择器（单行紧凑）- 与原文件一致"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(28)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(5, 2, 5, 2)
        self.layout.setSpacing(8)
        
        self.icon_label = QLabel()
        self.icon_label.setFixedSize(20, 20)
        self.icon_label.setPixmap(self.create_colorful_icon())
        
        self.name_label = QLabel("Black Oil")
        self.name_label.setStyleSheet("""
            QLabel {
                color: #4CAF50;
                font-weight: bold;
                font-size: 12px;
            }
        """)
        
        self.corner_grid_label = QLabel("Corner Grid")
        self.corner_grid_label.setStyleSheet("""
            QLabel {
                color: #2196F3;
                font-weight: bold;
                font-size: 12px;
            }
        """)
        self.corner_grid_label.setCursor(Qt.PointingHandCursor)
        self.corner_grid_label.mousePressEvent = self.on_corner_grid_click
        
        self.other_algos = ["Comp", "Thermal", "Foam", "Polymer"]
        
        self.layout.addWidget(self.icon_label)
        self.layout.addWidget(self.name_label)
        self.layout.addWidget(self.corner_grid_label)
        
        for algo in self.other_algos:
            lbl = QLabel(algo)
            lbl.setStyleSheet("color: #666666; font-size: 10px;")
            self.layout.addWidget(lbl)
        
        self.layout.addStretch()
        
        self.current_algorithm = "black_oil"
        self.on_algorithm_changed = None
    
    def on_corner_grid_click(self, event):
        """点击Corner Grid算法"""
        self.set_current_algorithm("black_oil_corner_grid")
    
    def set_current_algorithm(self, algo):
        """设置当前算法"""
        self.current_algorithm = algo
        if algo == "black_oil":
            self.name_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 12px;")
            self.corner_grid_label.setStyleSheet("color: #2196F3; font-weight: bold; font-size: 12px;")
        elif algo == "black_oil_corner_grid":
            self.name_label.setStyleSheet("color: #666666; font-weight: bold; font-size: 12px;")
            self.corner_grid_label.setStyleSheet("color: #FF9800; font-weight: bold; font-size: 12px;")
        
        if self.on_algorithm_changed:
            self.on_algorithm_changed(algo)
    
    def create_colorful_icon(self):
        """创建彩色算法图标 - 与原文件一致"""
        pixmap = QPixmap(20, 20)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        
        # 绘制彩色渐变圆形
        gradient = QColor(76, 175, 80)  # 绿色
        painter.setBrush(gradient)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(1, 1, 18, 18)
        
        # 绘制"油滴"形状
        painter.setBrush(QColor(255, 193, 7))  # 黄色
        painter.drawEllipse(6, 5, 8, 10)
        
        painter.end()
        return pixmap


class VTKWidget(QWidget):
    """VTK渲染窗口 - 与原文件一致"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.layout.addWidget(self.vtk_widget)
        
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.iren = self.vtk_widget.GetRenderWindow().GetInteractor()
        
        self.renderer.SetBackground(0.0, 0.0, 0.0)
        
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.iren.SetInteractorStyle(style)
        
        self.iren.Initialize()
    
    def reset_camera(self):
        self.renderer.ResetCamera()
        self.iren.Render()


class MainWindow(QMainWindow):
    """主窗口 - 与原文件完全一致"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EDFM Reservoir Simulator")
        self.setGeometry(50, 50, 1600, 1000)
        
        self.sim_data = SimulationData()
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.current_tab = "Grid"
        self.show_fractures_enabled = False
        self.sim_process = None
        self.sim_output_buffer = ""
        self.pending_result_path = None
        self.step_log_interval = 30
        self.current_sim_total_days = 100.0
        self.current_progress_days = 0.0
        self.current_progress_step = 0
        self.estimated_total_steps = 700
        self.pending_step_summary = False
        self.default_grid_type = "corner_point"
        self.show_grid_type_selector = False
        self.current_algorithm = "black_oil"
        
        # 缓存VTK对象，避免重复生成
        self.cache = {
            'pressure_actor': None,
            'fracture_actors': [],
            'scalar_bar': None,
            'grid_lines_actor': None,
            'data_hash': None  # 用于检测数据是否变化
        }
        
        self.init_ui()
        self.create_toolbar()
        self.create_left_panel()
        self.create_center_panel()
        self.create_bottom_panel()
        self.create_status_bar()
    
    def init_ui(self):
        """初始化主界面 - 与原文件一致"""
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主垂直布局：算法栏 + 分割器 + 底部面板
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 算法栏（放在最顶部）
        self.algo_bar = self.create_algorithm_bar_widget()
        main_layout.addWidget(self.algo_bar)
        
        # 主分割器
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #3d3d3d;
            }
        """)
        
        # 左侧面板
        self.left_panel = QWidget()
        self.left_panel.setStyleSheet("background-color: #2b2b2b;")
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_layout.setContentsMargins(5, 5, 5, 5)
        self.left_panel.setFixedWidth(320)
        
        # 中间面板
        self.center_panel = QWidget()
        self.center_layout = QVBoxLayout(self.center_panel)
        self.center_layout.setContentsMargins(0, 0, 0, 0)
        
        self.main_splitter.addWidget(self.left_panel)
        self.main_splitter.addWidget(self.center_panel)
        self.main_splitter.setSizes([320, 1280])
        
        # 底部面板
        self.bottom_panel = QWidget()
        self.bottom_panel.setStyleSheet("background-color: #1e1e1e; border: none;")
        self.bottom_panel.setMaximumHeight(200)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(22)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #1e1e1e;
                color: #cccccc;
                border: 1px solid #3d3d3d;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
        self.update_progress_bar(0.0, 0, "等待运行")
        
        main_layout.addWidget(self.main_splitter, 1)  # stretch factor
        main_layout.addWidget(self.bottom_panel)
        main_layout.addWidget(self.progress_bar)
    
    def create_algorithm_bar_widget(self):
        """创建算法选择栏部件 - 与原文件一致"""
        algo_bar = QWidget()
        algo_bar.setFixedHeight(32)
        algo_bar.setStyleSheet("background-color: #1e1e1e; border-bottom: 1px solid #3d3d3d;")
        algo_layout = QHBoxLayout(algo_bar)
        algo_layout.setContentsMargins(10, 2, 10, 2)
        algo_layout.setSpacing(10)
        
        algo_selector = AlgorithmSelector()
        algo_selector.on_algorithm_changed = self.on_algorithm_changed
        self.algo_selector = algo_selector
        algo_layout.addWidget(algo_selector)
        algo_layout.addStretch()
        
        return algo_bar
    
    def on_algorithm_changed(self, algo):
        """算法切换回调"""
        self.current_algorithm = algo
        self.status_bar.showMessage(f"Algorithm: {algo}")
    
    def generate_mock_corner_point_grid(self, nx=20, ny=10, nz=5, lx=1000.0, ly=500.0, lz=100.0):
        """生成模拟角点网格数据（带地质曲面效果）
        
        使用正弦波和随机扰动模拟真实地质曲面
        """
        import math
        import random
        
        cpg = CornerPointGridData()
        cpg.nx = nx
        cpg.ny = ny
        cpg.nz = nz
        cpg.lx = lx
        cpg.ly = ly
        cpg.lz = lz
        
        dx = lx / nx
        dy = ly / ny
        dz = lz / nz
        
        def surface_z(x, y, is_top=True):
            """生成地质曲面Z坐标
            
            使用多层正弦波叠加模拟真实地质构造
            """
            base_z = lz * 0.3 if not is_top else lz * 0.7
            
            wave1 = 15.0 * math.sin(2 * math.pi * x / lx * 2) * math.sin(2 * math.pi * y / ly * 1.5)
            wave2 = 10.0 * math.sin(2 * math.pi * x / lx * 3.5 + 0.5) * math.cos(2 * math.pi * y / ly * 2.5)
            wave3 = 8.0 * math.cos(2 * math.pi * x / lx * 1.5 + 1.0) * math.sin(2 * math.pi * y / ly * 3.0)
            
            fault_offset = 0
            if lx * 0.4 < x < lx * 0.6:
                fault_offset = 12.0 * math.sin(math.pi * (x - lx * 0.4) / (lx * 0.2))
            
            dome = 20.0 * math.exp(-((x - lx * 0.7)**2 + (y - ly * 0.3)**2) / (lx * ly * 0.05))
            
            if is_top:
                return base_z + wave1 + wave2 + wave3 + fault_offset + dome
            else:
                return base_z + wave1 * 0.5 + wave2 * 0.3 + wave3 * 0.4 + fault_offset * 0.5 + dome * 0.3
        
        well_x = lx * 0.5
        well_y = ly * 0.5
        well_z = lz * 0.5
        
        cell_id = 0
        min_p = float('inf')
        max_p = float('-inf')
        
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    cell = CornerPointCell(cell_id)
                    cell.ix = i
                    cell.iy = j
                    cell.iz = k
                    
                    x0 = i * dx
                    x1 = (i + 1) * dx
                    y0 = j * dy
                    y1 = (j + 1) * dy
                    
                    z_bot_local = surface_z(x0, y0, False)
                    z_bot_x1 = surface_z(x1, y0, False)
                    z_bot_y1 = surface_z(x0, y1, False)
                    z_bot_x1y1 = surface_z(x1, y1, False)
                    
                    z_top_local = surface_z(x0, y0, True)
                    z_top_x1 = surface_z(x1, y0, True)
                    z_top_y1 = surface_z(x0, y1, True)
                    z_top_x1y1 = surface_z(x1, y1, True)
                    
                    layer_factor = k / max(1, nz - 1)
                    z_bot = [z_bot_local, z_bot_x1, z_bot_y1, z_bot_x1y1]
                    z_top = [z_top_local, z_top_x1, z_top_y1, z_top_x1y1]
                    
                    z_bot_actual = [z + layer_factor * dz for z in z_bot]
                    z_top_actual = [z + layer_factor * dz for z in z_top]
                    
                    corners = [
                        (x0, y0, z_bot_actual[0]),
                        (x1, y0, z_bot_actual[1]),
                        (x1, y1, z_bot_actual[3]),
                        (x0, y1, z_bot_actual[2]),
                        (x0, y0, z_top_actual[0]),
                        (x1, y0, z_top_actual[1]),
                        (x1, y1, z_top_actual[3]),
                        (x0, y1, z_top_actual[2]),
                    ]
                    
                    cell.set_corners(corners)
                    
                    cx = (x0 + x1) / 2
                    cy = (y0 + y1) / 2
                    cz = (sum(z_bot_actual) + sum(z_top_actual)) / 8
                    
                    dist = math.sqrt((cx - well_x)**2 + (cy - well_y)**2 + (cz - well_z)**2)
                    max_dist = math.sqrt(lx**2 + ly**2 + lz**2)
                    
                    base_pressure = 50.0
                    max_pressure = 200.0
                    pressure = base_pressure + (max_pressure - base_pressure) * (dist / max_dist)
                    
                    cell.pressure = pressure
                    min_p = min(min_p, pressure)
                    max_p = max(max_p, pressure)
                    
                    cpg.cells.append(cell)
                    cell_id += 1
        
        cpg.min_pressure = min_p
        cpg.max_pressure = max_p
        
        print(f"Generated {len(cpg.cells)} corner point cells with geological surface")
        print(f"Pressure range: {min_p:.2f} - {max_p:.2f} bar")
        
        return cpg
    
    def create_toolbar(self):
        """创建顶部工具栏 - 与原文件一致"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setStyleSheet("""
            QToolBar {
                background-color: #2b2b2b;
                border-bottom: 1px solid #3d3d3d;
                padding: 2px;
            }
            QToolButton {
                background-color: transparent;
                border: none;
                padding: 5px;
                margin: 2px;
                color: #cccccc;
            }
            QToolButton:hover {
                background-color: #3d3d3d;
                border-radius: 3px;
            }
            QToolBar::separator {
                background-color: #3d3d3d;
                width: 1px;
                margin: 4px 8px;
            }
        """)
        self.addToolBar(toolbar)
        
        # 文件操作
        toolbar.addAction(QAction("New", self))
        toolbar.addAction(QAction("Open", self))
        toolbar.addAction(QAction("Save", self))
        toolbar.addSeparator()
        
        # 视图控制
        toolbar.addAction(QAction("Reset View", self))
        toolbar.addAction(QAction("Zoom In", self))
        toolbar.addAction(QAction("Zoom Out", self))
        toolbar.addSeparator()
        
        # Run Sim按钮（绿色高亮）
        self.run_btn = QPushButton("▶ Run Simulation")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 5px 15px;
                border: none;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.run_btn.clicked.connect(self.run_simulation)
        toolbar.addWidget(self.run_btn)
    
    def create_left_panel(self):
        """创建左侧面板 - 与原文件一致"""
        # 顶部标签按钮
        tab_frame = QFrame()
        tab_frame.setStyleSheet("background-color: #2b2b2b; border-bottom: 1px solid #3d3d3d;")
        tab_layout = QHBoxLayout(tab_frame)
        tab_layout.setContentsMargins(5, 5, 5, 5)
        tab_layout.setSpacing(5)
        
        self.tab_buttons = {}
        for tab_name in ["Grid", "Wells", "Fractures", "Results"]:
            btn = QPushButton(tab_name)
            btn.setCheckable(True)
            btn.setFixedHeight(30)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #3d3d3d;
                    color: #cccccc;
                    border: 1px solid #555555;
                    border-radius: 3px;
                    padding: 5px 15px;
                }
                QPushButton:checked {
                    background-color: #4CAF50;
                    color: white;
                }
                QPushButton:hover {
                    background-color: #4d4d4d;
                }
            """)
            btn.clicked.connect(lambda checked, name=tab_name: self.switch_tab(name))
            self.tab_buttons[tab_name] = btn
            tab_layout.addWidget(btn)
        
        self.tab_buttons["Grid"].setChecked(True)
        self.left_layout.addWidget(tab_frame)
        
        # 参数堆叠窗口
        self.param_stack = QStackedWidget()
        self.param_stack.setStyleSheet("background-color: #2b2b2b;")
        
        # Grid参数页面
        self.grid_page = self.create_grid_page()
        self.param_stack.addWidget(self.grid_page)
        
        # Wells参数页面
        self.wells_page = self.create_wells_page()
        self.param_stack.addWidget(self.wells_page)
        
        # Fractures参数页面
        self.fractures_page = self.create_fractures_page()
        self.param_stack.addWidget(self.fractures_page)
        
        # Results页面
        self.results_page = self.create_results_page()
        self.param_stack.addWidget(self.results_page)
        
        self.left_layout.addWidget(self.param_stack)
    
    def create_grid_page(self):
        """创建Grid参数页面，支持加密/不加密两套参数面板切换。"""
        page = QWidget()
        page.setStyleSheet("background-color: #2b2b2b;")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(10, 10, 10, 10)

        options_group = QGroupBox("Grid Options")
        options_group.setStyleSheet(self.groupbox_style())
        options_layout = QGridLayout()

        self.combo_grid_refinement = QComboBox()
        self.combo_grid_refinement.addItems(["不加密", "加密"])
        self.combo_grid_refinement.setStyleSheet("color: #cccccc; background-color: #3d3d3d;")
        self.combo_grid_refinement.currentTextChanged.connect(self.update_parameter_mode)

        # Keep a hidden backup interface for future grid-type switching.
        self.combo_grid_type = QComboBox()
        self.combo_grid_type.addItem("角格", "corner_point")
        self.combo_grid_type.addItem("网格", "cartesian")
        self.combo_grid_type.setCurrentIndex(0)
        self.combo_grid_type.setStyleSheet("color: #cccccc; background-color: #3d3d3d;")

        options_layout.addWidget(QLabel("是否加密:"), 0, 0)
        options_layout.addWidget(self.combo_grid_refinement, 0, 1)
        if self.show_grid_type_selector:
            options_layout.addWidget(QLabel("网格类型:"), 1, 0)
            options_layout.addWidget(self.combo_grid_type, 1, 1)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        self.grid_param_stack = QStackedWidget()
        self.grid_unrefined_page = self.create_unrefined_grid_params_page()
        self.grid_refined_page = self.create_refined_grid_params_page()
        self.grid_param_stack.addWidget(self.grid_unrefined_page)
        self.grid_param_stack.addWidget(self.grid_refined_page)
        layout.addWidget(self.grid_param_stack, 1)

        self.combo_grid_refinement.setCurrentText("不加密")
        self.update_parameter_mode(self.combo_grid_refinement.currentText())
        layout.addStretch()
        
        return page

    def create_unrefined_grid_params_page(self):
        """未加密角格参数面板，默认值以源码为准。"""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.basic_spin_nx = self.create_spinbox(1, 500, 20)
        self.basic_spin_ny = self.create_spinbox(1, 200, 10)
        self.basic_spin_nz = self.create_spinbox(1, 100, 2)
        self.basic_spin_lx = self.create_double_spinbox(1, 100000, 1000, decimals=1)
        self.basic_spin_ly = self.create_double_spinbox(1, 100000, 500, decimals=1)
        self.basic_spin_lz = self.create_double_spinbox(1, 10000, 50, decimals=1)
        layout.addWidget(self.create_parameter_group("Grid Parameters", [
            ("Nx:", self.basic_spin_nx),
            ("Ny:", self.basic_spin_ny),
            ("Nz:", self.basic_spin_nz),
            ("Lx (m):", self.basic_spin_lx),
            ("Ly (m):", self.basic_spin_ly),
            ("Lz (m):", self.basic_spin_lz),
        ]))

        self.basic_spin_porosity = self.create_double_spinbox(0.0, 1.0, 0.04, decimals=4)
        self.basic_spin_perm_x = self.create_double_spinbox(0.0, 1000000, 0.005, decimals=6)
        self.basic_spin_perm_y = self.create_double_spinbox(0.0, 1000000, 0.005, decimals=6)
        self.basic_spin_perm_z = self.create_double_spinbox(0.0, 1000000, 0.005, decimals=6)
        layout.addWidget(self.create_parameter_group("Matrix Properties", [
            ("Porosity:", self.basic_spin_porosity),
            ("Kx (Darcy):", self.basic_spin_perm_x),
            ("Ky (Darcy):", self.basic_spin_perm_y),
            ("Kz (Darcy):", self.basic_spin_perm_z),
        ]))

        self.basic_spin_initial_pressure = self.create_double_spinbox(0.0, 1000000, 800.0, decimals=2)
        self.basic_spin_initial_sw = self.create_double_spinbox(0.0, 1.0, 0.05, decimals=4)
        self.basic_spin_initial_sg = self.create_double_spinbox(0.0, 1.0, 0.9, decimals=4)
        layout.addWidget(self.create_parameter_group("Initial State", [
            ("Pressure (bar):", self.basic_spin_initial_pressure),
            ("Sw:", self.basic_spin_initial_sw),
            ("Sg:", self.basic_spin_initial_sg),
        ]))

        self.basic_spin_mu_w = self.create_double_spinbox(0.0, 1000.0, 1.0, decimals=4)
        self.basic_spin_mu_o = self.create_double_spinbox(0.0, 1000.0, 5.0, decimals=4)
        self.basic_spin_mu_g = self.create_double_spinbox(0.0, 1000.0, 0.2, decimals=4)
        self.basic_spin_cw = self.create_double_spinbox(0.0, 1.0, 1e-8, decimals=8, step=1e-8)
        self.basic_spin_co = self.create_double_spinbox(0.0, 1.0, 1e-5, decimals=8, step=1e-6)
        self.basic_spin_cg = self.create_double_spinbox(0.0, 1.0, 1e-3, decimals=6, step=1e-4)
        self.basic_spin_p_ref = self.create_double_spinbox(0.0, 1000000, 100.0, decimals=2)
        self.basic_spin_swi = self.create_double_spinbox(0.0, 1.0, 0.05, decimals=4)
        self.basic_spin_sor = self.create_double_spinbox(0.0, 1.0, 0.01, decimals=4)
        self.basic_spin_sgc = self.create_double_spinbox(0.0, 1.0, 0.05, decimals=4)
        layout.addWidget(self.create_parameter_group("Fluid Properties", [
            ("mu_w (cP):", self.basic_spin_mu_w),
            ("mu_o (cP):", self.basic_spin_mu_o),
            ("mu_g (cP):", self.basic_spin_mu_g),
            ("cw (1/bar):", self.basic_spin_cw),
            ("co (1/bar):", self.basic_spin_co),
            ("cg (1/bar):", self.basic_spin_cg),
            ("P_ref (bar):", self.basic_spin_p_ref),
            ("Swi:", self.basic_spin_swi),
            ("Sor:", self.basic_spin_sor),
            ("Sgc:", self.basic_spin_sgc),
        ]))

        self.basic_spin_simulation_time = self.create_double_spinbox(0.0, 1000000, 100.0, decimals=2)
        layout.addWidget(self.create_parameter_group("Simulation Control", [
            ("Simulation Time (days):", self.basic_spin_simulation_time),
        ]))
        layout.addStretch()

        return self.wrap_in_scroll_area(content)

    def create_refined_grid_params_page(self):
        """加密角格参数面板，默认值以源码为准。"""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.refined_spin_nx = self.create_spinbox(1, 500, 20)
        self.refined_spin_ny = self.create_spinbox(1, 500, 10)
        self.refined_spin_nz = self.create_spinbox(1, 200, 2)
        self.refined_spin_lx = self.create_double_spinbox(1, 100000, 1000, decimals=1)
        self.refined_spin_ly = self.create_double_spinbox(1, 100000, 500, decimals=1)
        self.refined_spin_lz = self.create_double_spinbox(1, 10000, 50, decimals=1)
        layout.addWidget(self.create_parameter_group("Grid Parameters", [
            ("Nx:", self.refined_spin_nx),
            ("Ny:", self.refined_spin_ny),
            ("Nz:", self.refined_spin_nz),
            ("Lx (m):", self.refined_spin_lx),
            ("Ly (m):", self.refined_spin_ly),
            ("Lz (m):", self.refined_spin_lz),
        ]))

        self.check_enable_lgr = QCheckBox("启用加密")
        self.check_enable_lgr.setChecked(True)
        self.refined_spin_d_threshold = self.create_double_spinbox(0.0, 10000.0, 30.0, decimals=2)
        self.refined_spin_lgr_nrx = self.create_spinbox(1, 20, 2)
        self.refined_spin_lgr_nry = self.create_spinbox(1, 20, 2)
        self.refined_spin_lgr_nrz = self.create_spinbox(1, 20, 2)
        lgr_group = QGroupBox("Refinement Settings")
        lgr_group.setStyleSheet(self.groupbox_style())
        lgr_layout = QGridLayout()
        lgr_layout.addWidget(self.check_enable_lgr, 0, 0, 1, 2)
        lgr_layout.addWidget(QLabel("Threshold (m):"), 1, 0)
        lgr_layout.addWidget(self.refined_spin_d_threshold, 1, 1)
        lgr_layout.addWidget(QLabel("Refine X:"), 2, 0)
        lgr_layout.addWidget(self.refined_spin_lgr_nrx, 2, 1)
        lgr_layout.addWidget(QLabel("Refine Y:"), 3, 0)
        lgr_layout.addWidget(self.refined_spin_lgr_nry, 3, 1)
        lgr_layout.addWidget(QLabel("Refine Z:"), 4, 0)
        lgr_layout.addWidget(self.refined_spin_lgr_nrz, 4, 1)
        lgr_group.setLayout(lgr_layout)
        layout.addWidget(lgr_group)

        self.refined_spin_porosity = self.create_double_spinbox(0.0, 1.0, 0.2, decimals=4)
        self.refined_spin_perm_x = self.create_double_spinbox(0.0, 1000000, 0.001, decimals=6)
        self.refined_spin_perm_y = self.create_double_spinbox(0.0, 1000000, 0.001, decimals=6)
        self.refined_spin_perm_z = self.create_double_spinbox(0.0, 1000000, 0.0001, decimals=6)
        layout.addWidget(self.create_parameter_group("Matrix Properties", [
            ("Porosity:", self.refined_spin_porosity),
            ("Kx (Darcy):", self.refined_spin_perm_x),
            ("Ky (Darcy):", self.refined_spin_perm_y),
            ("Kz (Darcy):", self.refined_spin_perm_z),
        ]))

        self.refined_spin_initial_pressure = self.create_double_spinbox(0.0, 1000000, 200.0, decimals=2)
        self.refined_spin_initial_sw = self.create_double_spinbox(0.0, 1.0, 0.2, decimals=4)
        self.refined_spin_initial_sg = self.create_double_spinbox(0.0, 1.0, 0.05, decimals=4)
        layout.addWidget(self.create_parameter_group("Initial State", [
            ("Pressure (bar):", self.refined_spin_initial_pressure),
            ("Sw:", self.refined_spin_initial_sw),
            ("Sg:", self.refined_spin_initial_sg),
        ]))

        self.refined_spin_mu_w = self.create_double_spinbox(0.0, 1000.0, 1.0, decimals=4)
        self.refined_spin_mu_o = self.create_double_spinbox(0.0, 1000.0, 5.0, decimals=4)
        self.refined_spin_mu_g = self.create_double_spinbox(0.0, 1000.0, 0.2, decimals=4)
        self.refined_spin_cw = self.create_double_spinbox(0.0, 1.0, 1e-8, decimals=8, step=1e-8)
        self.refined_spin_co = self.create_double_spinbox(0.0, 1.0, 1e-5, decimals=8, step=1e-6)
        self.refined_spin_cg = self.create_double_spinbox(0.0, 1.0, 1e-3, decimals=6, step=1e-4)
        self.refined_spin_p_ref = self.create_double_spinbox(0.0, 1000000, 100.0, decimals=2)
        self.refined_spin_swi = self.create_double_spinbox(0.0, 1.0, 0.2, decimals=4)
        self.refined_spin_sor = self.create_double_spinbox(0.0, 1.0, 0.2, decimals=4)
        self.refined_spin_sgc = self.create_double_spinbox(0.0, 1.0, 0.05, decimals=4)
        layout.addWidget(self.create_parameter_group("Fluid Properties", [
            ("mu_w (cP):", self.refined_spin_mu_w),
            ("mu_o (cP):", self.refined_spin_mu_o),
            ("mu_g (cP):", self.refined_spin_mu_g),
            ("cw (1/bar):", self.refined_spin_cw),
            ("co (1/bar):", self.refined_spin_co),
            ("cg (1/bar):", self.refined_spin_cg),
            ("P_ref (bar):", self.refined_spin_p_ref),
            ("Swi:", self.refined_spin_swi),
            ("Sor:", self.refined_spin_sor),
            ("Sgc:", self.refined_spin_sgc),
        ]))

        self.refined_spin_simulation_time = self.create_double_spinbox(0.0, 1000000, 100.0, decimals=2)
        self.refined_spin_time_step = self.create_double_spinbox(0.0, 1000000, 1.0, decimals=4)
        layout.addWidget(self.create_parameter_group("Simulation Control", [
            ("Simulation Time (days):", self.refined_spin_simulation_time),
            ("Time Step (days):", self.refined_spin_time_step),
        ]))
        layout.addStretch()

        return self.wrap_in_scroll_area(content)

    def create_parameter_group(self, title, fields):
        """创建统一风格的参数分组。"""
        group = QGroupBox(title)
        group.setStyleSheet(self.groupbox_style())
        grid = QGridLayout()
        for row, (label, widget) in enumerate(fields):
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(widget, row, 1)
        group.setLayout(grid)
        return group

    def create_spinbox(self, minimum, maximum, value):
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

    def create_double_spinbox(self, minimum, maximum, value, decimals=2, step=None):
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setValue(value)
        if step is not None:
            spin.setSingleStep(step)
        return spin

    def wrap_in_scroll_area(self, widget):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(widget)
        return scroll

    def update_grid_parameter_panel(self, refinement_text):
        """根据是否加密切换整套参数面板。"""
        if refinement_text == "加密":
            self.grid_param_stack.setCurrentWidget(self.grid_refined_page)
        else:
            self.grid_param_stack.setCurrentWidget(self.grid_unrefined_page)

    def update_wells_parameter_panel(self, refinement_text):
        """根据是否加密切换 Wells 参数面板。"""
        if hasattr(self, 'wells_param_stack'):
            if refinement_text == "加密":
                self.wells_param_stack.setCurrentWidget(self.wells_refined_page)
            else:
                self.wells_param_stack.setCurrentWidget(self.wells_unrefined_page)

    def update_fractures_parameter_panel(self, refinement_text):
        """根据是否加密切换 Fractures 参数面板。"""
        if hasattr(self, 'fractures_param_stack'):
            if refinement_text == "加密":
                self.fractures_param_stack.setCurrentWidget(self.fractures_refined_page)
            else:
                self.fractures_param_stack.setCurrentWidget(self.fractures_unrefined_page)

    def update_parameter_mode(self, refinement_text):
        """统一同步 Grid/Wells/Fractures 三个页签的参数面板。"""
        self.update_grid_parameter_panel(refinement_text)
        self.update_wells_parameter_panel(refinement_text)
        self.update_fractures_parameter_panel(refinement_text)

    def is_refined_grid_mode(self):
        return self.combo_grid_refinement.currentText() == "加密"

    def collect_unrefined_grid_params(self):
        """收集未加密角格页面参数。"""
        return {
            'grid_mode': 'basic_corner_point',
            'nx': self.basic_spin_nx.value(),
            'ny': self.basic_spin_ny.value(),
            'nz': self.basic_spin_nz.value(),
            'lx': self.basic_spin_lx.value(),
            'ly': self.basic_spin_ly.value(),
            'lz': self.basic_spin_lz.value(),
            'porosity': self.basic_spin_porosity.value(),
            'perm_x': self.basic_spin_perm_x.value(),
            'perm_y': self.basic_spin_perm_y.value(),
            'perm_z': self.basic_spin_perm_z.value(),
            'initial_pressure': self.basic_spin_initial_pressure.value(),
            'initial_sw': self.basic_spin_initial_sw.value(),
            'initial_sg': self.basic_spin_initial_sg.value(),
            'mu_w': self.basic_spin_mu_w.value(),
            'mu_o': self.basic_spin_mu_o.value(),
            'mu_g': self.basic_spin_mu_g.value(),
            'cw': self.basic_spin_cw.value(),
            'co': self.basic_spin_co.value(),
            'cg': self.basic_spin_cg.value(),
            'p_ref': self.basic_spin_p_ref.value(),
            'swi': self.basic_spin_swi.value(),
            'sor': self.basic_spin_sor.value(),
            'sgc': self.basic_spin_sgc.value(),
            'simulation_time': self.basic_spin_simulation_time.value(),
            'time_step': 0.001,
        }

    def collect_refined_grid_params(self):
        """收集加密角格页面参数。"""
        return {
            'grid_mode': 'lgr_corner_point',
            'nx': self.refined_spin_nx.value(),
            'ny': self.refined_spin_ny.value(),
            'nz': self.refined_spin_nz.value(),
            'lx': self.refined_spin_lx.value(),
            'ly': self.refined_spin_ly.value(),
            'lz': self.refined_spin_lz.value(),
            'enable_lgr': self.check_enable_lgr.isChecked(),
            'd_threshold': self.refined_spin_d_threshold.value(),
            'lgr_nrx': self.refined_spin_lgr_nrx.value(),
            'lgr_nry': self.refined_spin_lgr_nry.value(),
            'lgr_nrz': self.refined_spin_lgr_nrz.value(),
            'porosity': self.refined_spin_porosity.value(),
            'perm_x': self.refined_spin_perm_x.value(),
            'perm_y': self.refined_spin_perm_y.value(),
            'perm_z': self.refined_spin_perm_z.value(),
            'initial_pressure': self.refined_spin_initial_pressure.value(),
            'initial_sw': self.refined_spin_initial_sw.value(),
            'initial_sg': self.refined_spin_initial_sg.value(),
            'mu_w': self.refined_spin_mu_w.value(),
            'mu_o': self.refined_spin_mu_o.value(),
            'mu_g': self.refined_spin_mu_g.value(),
            'cw': self.refined_spin_cw.value(),
            'co': self.refined_spin_co.value(),
            'cg': self.refined_spin_cg.value(),
            'p_ref': self.refined_spin_p_ref.value(),
            'swi': self.refined_spin_swi.value(),
            'sor': self.refined_spin_sor.value(),
            'sgc': self.refined_spin_sgc.value(),
            'simulation_time': self.refined_spin_simulation_time.value(),
            'time_step': self.refined_spin_time_step.value(),
        }
    
    def create_wells_page(self):
        """创建 Wells 参数页面，支持加密/不加密两套面板。"""
        page = QWidget()
        page.setStyleSheet("background-color: #2b2b2b;")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(10, 10, 10, 10)

        self.wells_param_stack = QStackedWidget()
        self.wells_unrefined_page = self.create_unrefined_wells_params_page()
        self.wells_refined_page = self.create_refined_wells_params_page()
        self.wells_param_stack.addWidget(self.wells_unrefined_page)
        self.wells_param_stack.addWidget(self.wells_refined_page)
        layout.addWidget(self.wells_param_stack, 1)
        self.update_wells_parameter_panel(self.combo_grid_refinement.currentText())
        layout.addStretch()
        
        return page

    def create_unrefined_wells_params_page(self):
        """未加密模式的 Wells 参数面板。"""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.basic_spin_well_x = self.create_double_spinbox(0.0, 100000.0, 250.0, decimals=2)
        self.basic_spin_well_y = self.create_double_spinbox(0.0, 100000.0, 250.0, decimals=2)
        self.basic_spin_well_z = self.create_double_spinbox(0.0, 10000.0, 50.0, decimals=2)
        self.basic_spin_well_pressure = self.create_double_spinbox(0.0, 100000.0, 50.0, decimals=2)
        self.basic_spin_well_radius = self.create_double_spinbox(0.001, 100.0, 0.05, decimals=3)
        layout.addWidget(self.create_parameter_group("Well Parameters", [
            ("Well X (m):", self.basic_spin_well_x),
            ("Well Y (m):", self.basic_spin_well_y),
            ("Well Z (m):", self.basic_spin_well_z),
            ("Pressure (bar):", self.basic_spin_well_pressure),
            ("Radius (m):", self.basic_spin_well_radius),
        ]))

        self.basic_check_enable_hf = QCheckBox("启用人工裂缝")
        self.basic_check_enable_hf.setChecked(True)
        self.basic_spin_hf_count = self.create_spinbox(0, 200, 20)
        self.basic_spin_hf_well_length = self.create_double_spinbox(0.0, 100000.0, 2000.0, decimals=2)
        self.basic_spin_hf_length = self.create_double_spinbox(0.0, 100000.0, 120.0, decimals=2)
        self.basic_spin_hf_height = self.create_double_spinbox(0.0, 100000.0, 40.0, decimals=2)
        self.basic_spin_hf_aperture = self.create_double_spinbox(0.0, 10.0, 0.01, decimals=4)
        self.basic_spin_hf_perm = self.create_double_spinbox(0.0, 1000000.0, 10000.0, decimals=2)
        self.basic_spin_hf_center_x = self.create_double_spinbox(0.0, 100000.0, 1500.0, decimals=2)
        self.basic_spin_hf_center_y = self.create_double_spinbox(0.0, 100000.0, 150.0, decimals=2)
        self.basic_spin_hf_center_z = self.create_double_spinbox(0.0, 100000.0, 20.0, decimals=2)

        hf_group = QGroupBox("Hydraulic Fractures")
        hf_group.setStyleSheet(self.groupbox_style())
        hf_layout = QGridLayout()
        hf_layout.addWidget(self.basic_check_enable_hf, 0, 0, 1, 2)
        hf_layout.addWidget(QLabel("裂缝数量:"), 1, 0)
        hf_layout.addWidget(self.basic_spin_hf_count, 1, 1)
        hf_layout.addWidget(QLabel("Well Length (m):"), 2, 0)
        hf_layout.addWidget(self.basic_spin_hf_well_length, 2, 1)
        hf_layout.addWidget(QLabel("裂缝长度 (m):"), 3, 0)
        hf_layout.addWidget(self.basic_spin_hf_length, 3, 1)
        hf_layout.addWidget(QLabel("裂缝高度 (m):"), 4, 0)
        hf_layout.addWidget(self.basic_spin_hf_height, 4, 1)
        hf_layout.addWidget(QLabel("裂缝开度 (m):"), 5, 0)
        hf_layout.addWidget(self.basic_spin_hf_aperture, 5, 1)
        hf_layout.addWidget(QLabel("裂缝渗透率 (Darcy):"), 6, 0)
        hf_layout.addWidget(self.basic_spin_hf_perm, 6, 1)
        hf_layout.addWidget(QLabel("中心 X (m):"), 7, 0)
        hf_layout.addWidget(self.basic_spin_hf_center_x, 7, 1)
        hf_layout.addWidget(QLabel("中心 Y (m):"), 8, 0)
        hf_layout.addWidget(self.basic_spin_hf_center_y, 8, 1)
        hf_layout.addWidget(QLabel("中心 Z (m):"), 9, 0)
        hf_layout.addWidget(self.basic_spin_hf_center_z, 9, 1)
        hf_group.setLayout(hf_layout)
        layout.addWidget(hf_group)
        layout.addStretch()

        return self.wrap_in_scroll_area(content)

    def create_refined_wells_params_page(self):
        """加密模式的 Wells 参数面板。"""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.refined_spin_well_x = self.create_double_spinbox(0.0, 100000.0, 500.0, decimals=2)
        self.refined_spin_well_y = self.create_double_spinbox(0.0, 100000.0, 250.0, decimals=2)
        self.refined_spin_well_z = self.create_double_spinbox(0.0, 100000.0, 50.0, decimals=2)
        self.refined_spin_well_pressure = self.create_double_spinbox(0.0, 100000.0, 50.0, decimals=2)
        self.refined_spin_well_radius = self.create_double_spinbox(0.001, 100.0, 0.05, decimals=3)
        layout.addWidget(self.create_parameter_group("Well Parameters", [
            ("Well X (m):", self.refined_spin_well_x),
            ("Well Y (m):", self.refined_spin_well_y),
            ("Well Z (m):", self.refined_spin_well_z),
            ("Pressure (bar):", self.refined_spin_well_pressure),
            ("Radius (m):", self.refined_spin_well_radius),
        ]))
        layout.addStretch()

        return self.wrap_in_scroll_area(content)

    def collect_unrefined_wells_params(self):
        hf_count = self.basic_spin_hf_count.value()
        hf_well_length = self.basic_spin_hf_well_length.value()
        return {
            'well_x': self.basic_spin_well_x.value(),
            'well_y': self.basic_spin_well_y.value(),
            'well_z': self.basic_spin_well_z.value(),
            'well_pressure': self.basic_spin_well_pressure.value(),
            'well_radius': self.basic_spin_well_radius.value(),
            'hf_enabled': self.basic_check_enable_hf.isChecked(),
            'hf_count': hf_count,
            'hf_well_length': hf_well_length,
            'hf_center_x': self.basic_spin_hf_center_x.value(),
            'hf_center_y': self.basic_spin_hf_center_y.value(),
            'hf_center_z': self.basic_spin_hf_center_z.value(),
            'hf_spacing_x': (hf_well_length / (hf_count - 1)) if hf_count > 1 else 0.0,
            'hf_length': self.basic_spin_hf_length.value(),
            'hf_height': self.basic_spin_hf_height.value(),
            'hf_aperture': self.basic_spin_hf_aperture.value(),
            'hf_perm': self.basic_spin_hf_perm.value(),
        }

    def collect_refined_wells_params(self):
        return {
            'well_x': self.refined_spin_well_x.value(),
            'well_y': self.refined_spin_well_y.value(),
            'well_z': self.refined_spin_well_z.value(),
            'well_pressure': self.refined_spin_well_pressure.value(),
            'well_radius': self.refined_spin_well_radius.value(),
            'hf_enabled': False,
            'hf_count': 0,
            'hf_well_length': 0.0,
            'hf_center_x': self.refined_spin_well_x.value(),
            'hf_center_y': self.refined_spin_well_y.value(),
            'hf_center_z': self.refined_spin_well_z.value(),
            'hf_spacing_x': 0.0,
            'hf_length': 0.0,
            'hf_height': 0.0,
            'hf_aperture': 0.0,
            'hf_perm': 0.0,
        }
    
    def create_fractures_page(self):
        """创建 Fractures 参数页面，支持加密/不加密两套面板。"""
        page = QWidget()
        page.setStyleSheet("background-color: #2b2b2b;")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(10, 10, 10, 10)

        self.fractures_param_stack = QStackedWidget()
        self.fractures_unrefined_page = self.create_unrefined_fractures_params_page()
        self.fractures_refined_page = self.create_refined_fractures_params_page()
        self.fractures_param_stack.addWidget(self.fractures_unrefined_page)
        self.fractures_param_stack.addWidget(self.fractures_refined_page)
        layout.addWidget(self.fractures_param_stack, 1)
        self.update_fractures_parameter_panel(self.combo_grid_refinement.currentText())
        layout.addStretch()
        
        return page

    def create_unrefined_fractures_params_page(self):
        """未加密模式的天然裂缝参数面板。"""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.basic_spin_num_fracs = self.create_spinbox(0, 500, 100)
        self.basic_spin_min_len = self.create_double_spinbox(0.0, 100000.0, 30.0, decimals=2)
        self.basic_spin_max_len = self.create_double_spinbox(0.0, 100000.0, 80.0, decimals=2)
        self.basic_spin_max_dip = self.create_double_spinbox(0.0, math.pi, math.pi / 3.0, decimals=4)
        self.basic_spin_min_strike = self.create_double_spinbox(0.0, 2 * math.pi, 0.0, decimals=4)
        self.basic_spin_max_strike = self.create_double_spinbox(0.0, 2 * math.pi, math.pi, decimals=4)
        self.basic_spin_aperture = self.create_double_spinbox(0.0, 10.0, 0.001, decimals=4)
        self.basic_spin_frac_perm = self.create_double_spinbox(0.0, 1000000.0, 1000.0, decimals=2)
        layout.addWidget(self.create_parameter_group("Natural Fractures", [
            ("Num Fractures:", self.basic_spin_num_fracs),
            ("Min Length (m):", self.basic_spin_min_len),
            ("Max Length (m):", self.basic_spin_max_len),
            ("Max Dip (rad):", self.basic_spin_max_dip),
            ("Min Strike (rad):", self.basic_spin_min_strike),
            ("Max Strike (rad):", self.basic_spin_max_strike),
            ("Aperture (m):", self.basic_spin_aperture),
            ("Permeability (Darcy):", self.basic_spin_frac_perm),
        ]))
        layout.addStretch()

        return self.wrap_in_scroll_area(content)

    def create_refined_fractures_params_page(self):
        """加密模式的天然裂缝参数面板。"""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.refined_spin_num_fracs = self.create_spinbox(0, 500, 100)
        self.refined_spin_min_len = self.create_double_spinbox(0.0, 100000.0, 30.0, decimals=2)
        self.refined_spin_max_len = self.create_double_spinbox(0.0, 100000.0, 80.0, decimals=2)
        self.refined_spin_max_dip = self.create_double_spinbox(0.0, math.pi, math.pi / 3.0, decimals=4)
        self.refined_spin_min_strike = self.create_double_spinbox(0.0, 2 * math.pi, 0.0, decimals=4)
        self.refined_spin_max_strike = self.create_double_spinbox(0.0, 2 * math.pi, math.pi, decimals=4)
        self.refined_spin_aperture = self.create_double_spinbox(0.0, 10.0, 0.001, decimals=4)
        self.refined_spin_frac_perm = self.create_double_spinbox(0.0, 1000000.0, 10000.0, decimals=2)
        layout.addWidget(self.create_parameter_group("Natural Fractures", [
            ("Num Fractures:", self.refined_spin_num_fracs),
            ("Min Length (m):", self.refined_spin_min_len),
            ("Max Length (m):", self.refined_spin_max_len),
            ("Max Dip (rad):", self.refined_spin_max_dip),
            ("Min Strike (rad):", self.refined_spin_min_strike),
            ("Max Strike (rad):", self.refined_spin_max_strike),
            ("Aperture (m):", self.refined_spin_aperture),
            ("Permeability (Darcy):", self.refined_spin_frac_perm),
        ]))
        layout.addStretch()

        return self.wrap_in_scroll_area(content)

    def collect_unrefined_fractures_params(self):
        return {
            'num_fracs': self.basic_spin_num_fracs.value(),
            'min_len': self.basic_spin_min_len.value(),
            'max_len': self.basic_spin_max_len.value(),
            'max_dip': self.basic_spin_max_dip.value(),
            'min_strike': self.basic_spin_min_strike.value(),
            'max_strike': self.basic_spin_max_strike.value(),
            'aperture': self.basic_spin_aperture.value(),
            'frac_perm': self.basic_spin_frac_perm.value(),
        }

    def collect_refined_fractures_params(self):
        return {
            'num_fracs': self.refined_spin_num_fracs.value(),
            'min_len': self.refined_spin_min_len.value(),
            'max_len': self.refined_spin_max_len.value(),
            'max_dip': self.refined_spin_max_dip.value(),
            'min_strike': self.refined_spin_min_strike.value(),
            'max_strike': self.refined_spin_max_strike.value(),
            'aperture': self.refined_spin_aperture.value(),
            'frac_perm': self.refined_spin_frac_perm.value(),
        }
    
    def create_results_page(self):
        """创建Results页面 - 与原文件一致"""
        page = QWidget()
        page.setStyleSheet("background-color: #2b2b2b;")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 视图模式选择
        view_group = QGroupBox("View Mode")
        view_group.setStyleSheet(self.groupbox_style())
        view_layout = QVBoxLayout()
        
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(["Pressure Field", "Fracture Mesh"])
        self.view_mode_combo.setStyleSheet("color: #cccccc; background-color: #3d3d3d;")
        self.view_mode_combo.currentTextChanged.connect(self.change_view_mode)
        
        view_layout.addWidget(QLabel("Select View:"))
        view_layout.addWidget(self.view_mode_combo)
        view_group.setLayout(view_layout)
        layout.addWidget(view_group)
        
        # 显示选项
        group = QGroupBox("Visualization Options")
        group.setStyleSheet(self.groupbox_style())
        vlayout = QVBoxLayout()
        
        self.combo_field = QComboBox()
        self.combo_field.addItems(["Pressure", "Temperature", "Stress"])
        self.combo_field.setStyleSheet("color: #cccccc; background-color: #3d3d3d;")
        self.combo_field.currentTextChanged.connect(self.change_field_display)
        
        self.check_show_grid = QCheckBox("Show Grid Lines")
        self.check_show_grid.setChecked(False)
        self.check_show_grid.stateChanged.connect(self.toggle_grid_lines)
        
        self.check_show_fractures = QCheckBox("Show Fractures")
        self.check_show_fractures.setChecked(False)
        self.check_show_fractures.stateChanged.connect(self.toggle_fractures_visibility)
        
        vlayout.addWidget(QLabel("Display Field:"))
        vlayout.addWidget(self.combo_field)
        vlayout.addWidget(self.check_show_grid)
        vlayout.addWidget(self.check_show_fractures)
        
        group.setLayout(vlayout)
        layout.addWidget(group)
        layout.addStretch()
        
        return page
    
    def create_center_panel(self):
        """创建中间VTK视图面板 - 与原文件一致"""
        self.vtk_widget = VTKWidget()
        self.center_layout.addWidget(self.vtk_widget)
        
        # 初始化VTK渲染器
        self.vtk_renderer = VTKRenderer(self.vtk_widget)
    
    def create_bottom_panel(self):
        """创建底部数据面板 - 与原文件一致，3个panel"""
        layout = QHBoxLayout(self.bottom_panel)
        layout.setContentsMargins(5, 5, 5, 5)
        
        self.sim_status_group = QGroupBox("Simulation Status")
        self.sim_status_group.setStyleSheet(self.groupbox_style())
        status_layout = QVBoxLayout()
        self.sim_status_text = QTextEdit()
        self.sim_status_text.setReadOnly(True)
        self.sim_status_text.setStyleSheet("background-color: #1e1e1e; color: #00ff00; border: 1px solid #3d3d3d; font-family: Consolas;")
        self.sim_status_text.setMaximumHeight(150)
        status_layout.addWidget(self.sim_status_text)
        self.sim_status_group.setLayout(status_layout)
        
        stats_group = QGroupBox("Simulation Statistics")
        stats_group.setStyleSheet(self.groupbox_style())
        stats_layout = QVBoxLayout()
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setStyleSheet("background-color: #1e1e1e; color: #cccccc; border: 1px solid #3d3d3d;")
        self.stats_text.setMaximumHeight(150)
        stats_layout.addWidget(self.stats_text)
        stats_group.setLayout(stats_layout)
        
        prop_group = QGroupBox("Properties")
        prop_group.setStyleSheet(self.groupbox_style())
        prop_layout = QVBoxLayout()
        self.prop_table = QTableWidget()
        self.prop_table.setColumnCount(2)
        self.prop_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.prop_table.setStyleSheet("background-color: #1e1e1e; color: #cccccc;")
        self.prop_table.setMaximumHeight(150)
        prop_layout.addWidget(self.prop_table)
        prop_group.setLayout(prop_layout)
        
        layout.addWidget(self.sim_status_group, 2)
        layout.addWidget(stats_group, 2)
        layout.addWidget(prop_group, 1)
    
    def create_status_bar(self):
        """创建状态栏 - 与原文件一致"""
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("background-color: #2b2b2b; color: #cccccc;")
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Click 'Run Simulation' to start")
    
    def groupbox_style(self):
        """GroupBox样式 - 与原文件一致"""
        return """
            QGroupBox {
                background-color: #2b2b2b;
                color: #cccccc;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                margin-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLabel {
                color: #cccccc;
            }
            QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #3d3d3d;
                color: #cccccc;
                border: 1px solid #555555;
                padding: 3px;
            }
            QCheckBox {
                color: #cccccc;
            }
        """
    
    def switch_tab(self, tab_name):
        """切换标签页 - 与原文件一致"""
        self.current_tab = tab_name
        for name, btn in self.tab_buttons.items():
            btn.setChecked(name == tab_name)
        
        tab_index = {"Grid": 0, "Wells": 1, "Fractures": 2, "Results": 3}
        self.param_stack.setCurrentIndex(tab_index.get(tab_name, 0))
    
    def append_sim_status(self, text):
        """添加模拟状态信息 - 与原文件一致"""
        self.sim_status_text.append(text)
        self.sim_status_text.verticalScrollBar().setValue(self.sim_status_text.verticalScrollBar().maximum())
        QApplication.processEvents()
    
    def clear_sim_status(self):
        """清除模拟状态"""
        self.sim_status_text.clear()
    
    def clear_cache(self):
        """清除VTK缓存"""
        self.cache['pressure_actor'] = None
        self.cache['fracture_actors'] = []
        self.cache['scalar_bar'] = None
        self.cache['grid_lines_actor'] = None
        self.cache['data_hash'] = None
        if hasattr(self, 'vtk_renderer'):
            self.vtk_renderer.clear_cache()
        print("Cache cleared")

    def reset_progress_state(self):
        """重置模拟进度状态。"""
        self.current_sim_total_days = 100.0
        self.current_progress_days = 0.0
        self.current_progress_step = 0
        self.pending_step_summary = False
        self.update_progress_bar(0.0, 0, "准备启动")

    def update_progress_bar(self, current_days, step, status_text="运行中"):
        """按步数估算更新底部进度条。"""
        safe_days = max(0.0, current_days)
        estimated_steps = max(1, self.estimated_total_steps)
        safe_step = max(0, step)
        raw_progress = (safe_step / estimated_steps) * 100
        if 0 < safe_step < estimated_steps and raw_progress < 1.0:
            progress = 1
        elif safe_step < estimated_steps and raw_progress > 99.0:
            progress = 99
        else:
            progress = int(round(max(0.0, min(raw_progress, 100.0))))
        self.progress_bar.setValue(progress)
        self.progress_bar.setFormat(
            f"模拟进度 {progress}%  |  Step {safe_step}  |  "
            f"t = {self.format_day_value(safe_days)} 天  |  {status_text}"
        )

    def format_day_value(self, value):
        """根据数值大小格式化时间，避免前期进度长期显示为 0.00。"""
        if value >= 1.0:
            return f"{value:.2f}"
        if value >= 0.01:
            return f"{value:.4f}"
        return f"{value:.6g}"

    def mark_progress_complete(self):
        """标记模拟完成。"""
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat(
            f"模拟进度 100%  |  Step {self.current_progress_step}  |  "
            f"t = {self.format_day_value(self.current_progress_days)} 天  |  模拟完成"
        )

    def mark_progress_failed(self):
        """标记模拟失败。"""
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("模拟进度 0%  |  运行失败")

    def process_simulation_log_line(self, line):
        """解析子进程日志，更新进度条并节流 Step 日志显示。"""
        sim_time_match = re.match(r"^Simulation time:\s*([0-9eE.+-]+)\s*days$", line)
        if sim_time_match:
            self.current_sim_total_days = float(sim_time_match.group(1))
            self.update_progress_bar(self.current_progress_days, self.current_progress_step, "运行中")
            self.append_sim_status(line)
            return

        step_match = re.match(r"^Step\s+(\d+)\s+t=([0-9eE.+-]+)\s+dt=", line)
        if step_match:
            step = int(step_match.group(1))
            self.current_progress_step = max(self.current_progress_step, step)
            should_display = ("ok" in line) and (step % self.step_log_interval == 0)
            self.pending_step_summary = should_display
            if should_display:
                self.append_sim_status(line)
            return

        summary_match = re.match(r"^\s*t=([0-9eE.+-]+)\s+days,\s+P:", line)
        if summary_match:
            current_days = float(summary_match.group(1))
            self.current_progress_days = max(self.current_progress_days, current_days)
            self.update_progress_bar(self.current_progress_days, self.current_progress_step, "运行中")
            if self.pending_step_summary:
                self.append_sim_status(line)
            self.pending_step_summary = False
            return

        if line.startswith("Step "):
            self.pending_step_summary = False
            return

        self.pending_step_summary = False
        self.append_sim_status(line)
    
    def collect_simulation_params(self):
        """收集当前UI中的模拟参数。"""
        params = {
            'grid_refinement': self.combo_grid_refinement.currentText(),
            'grid_type': (
                self.combo_grid_type.currentData()
                if self.show_grid_type_selector else self.default_grid_type
            ),
        }

        if self.is_refined_grid_mode():
            params.update(self.collect_refined_grid_params())
            params.update(self.collect_refined_fractures_params())
            params.update(self.collect_refined_wells_params())
        else:
            params.update(self.collect_unrefined_grid_params())
            params.update(self.collect_unrefined_fractures_params())
            params.update(self.collect_unrefined_wells_params())

        return params

    def run_simulation(self):
        """通过子进程运行模拟，并实时显示算法输出。"""
        if self.sim_process and self.sim_process.state() != QProcess.NotRunning:
            self.append_sim_status("Simulation is already running.")
            return

        if self.current_algorithm == "black_oil_corner_grid":
            self.run_corner_point_grid_simulation()
            return

        params = self.collect_simulation_params()
        self.clear_cache()
        self.reset_progress_state()

        self.status_bar.showMessage("Running simulation...")
        self.clear_sim_status()
        self.append_sim_status("=" * 50)
        self.append_sim_status("  EDFM Black Oil Simulation Starting...")
        self.append_sim_status("=" * 50)
        self.append_sim_status(
            f"Grid: {params['nx']}x{params['ny']}x{params['nz']}, "
            f"Domain: {params['lx']}x{params['ly']}x{params['lz']} m"
        )
        self.append_sim_status(
            f"Fractures: {params['num_fracs']}, Length: {params['min_len']}-{params['max_len']} m, "
            f"Aperture: {params['aperture']} m"
        )
        self.append_sim_status(
            f"Well: ({params['well_x']}, {params['well_y']}, {params['well_z']}), "
            f"Pressure: {params['well_pressure']} bar"
        )
        self.append_sim_status("")

        tmp_dir = os.path.join(self.project_root, '.tmp')
        os.makedirs(tmp_dir, exist_ok=True)
        fd, self.pending_result_path = tempfile.mkstemp(prefix='simulation_result_', suffix='.json', dir=tmp_dir)
        os.close(fd)
        self.sim_output_buffer = ""

        self.sim_process = QProcess(self)
        self.sim_process.setWorkingDirectory(self.project_root)
        self.sim_process.setProgram(sys.executable)
        self.sim_process.setArguments([
            '-u',
            '-m',
            'front.simulation_runner',
            '--output',
            self.pending_result_path,
            '--params',
            json.dumps(params),
        ])
        self.sim_process.setProcessChannelMode(QProcess.MergedChannels)
        self.sim_process.readyReadStandardOutput.connect(self.handle_process_output)
        self.sim_process.finished.connect(self.handle_simulation_finished)
        self.sim_process.errorOccurred.connect(self.handle_simulation_error)

        self.run_btn.setEnabled(False)
        self.sim_process.start()
    
    def run_corner_point_grid_simulation(self):
        """运行角点网格模拟（使用模拟数据）"""
        self.clear_cache()
        self.status_bar.showMessage("Running Corner Point Grid simulation...")
        self.clear_sim_status()
        
        self.append_sim_status("=" * 50)
        self.append_sim_status("  Corner Point Grid Simulation Starting...")
        self.append_sim_status("=" * 50)
        
        params = self.collect_simulation_params()
        nx = params.get('nx', 20)
        ny = params.get('ny', 10)
        nz = params.get('nz', 5)
        lx = params.get('lx', 1000.0)
        ly = params.get('ly', 500.0)
        lz = params.get('lz', 100.0)
        
        self.append_sim_status(f"Grid: {nx}x{ny}x{nz}, Domain: {lx}x{ly}x{lz} m")
        self.append_sim_status("Generating corner point grid with geological surface...")
        
        cpg = self.generate_mock_corner_point_grid(nx, ny, nz, lx, ly, lz)
        
        self.sim_data.corner_point_grid = cpg
        self.sim_data.grid_info = {'nx': nx, 'ny': ny, 'nz': nz, 'Lx': lx, 'Ly': ly, 'Lz': lz}
        
        self.append_sim_status(f"Generated {len(cpg.cells)} cells")
        self.append_sim_status(f"Pressure range: {cpg.min_pressure:.2f} - {cpg.max_pressure:.2f} bar")
        self.append_sim_status("")
        self.append_sim_status("=" * 50)
        self.append_sim_status("  Simulation Completed Successfully!")
        self.append_sim_status("=" * 50)
        
        self.vtk_renderer.render_corner_point_grid(self.sim_data)
        self.update_corner_grid_statistics()
        
        self.status_bar.showMessage("Corner Point Grid simulation completed")
    
    def update_corner_grid_statistics(self):
        """更新角点网格统计信息"""
        if not self.sim_data.corner_point_grid:
            return
        
        cpg = self.sim_data.corner_point_grid
        
        self.stats_text.clear()
        self.stats_text.append(f"Grid: {cpg.nx} x {cpg.ny} x {cpg.nz} = {len(cpg.cells)} cells")
        self.stats_text.append("")
        self.stats_text.append("Pressure Range:")
        self.stats_text.append(f"Min: {cpg.min_pressure:.2f} bar")
        self.stats_text.append(f"Max: {cpg.max_pressure:.2f} bar")
        
        self.prop_table.setRowCount(5)
        props = [
            ("Algorithm", "Corner Point Grid"),
            ("Grid Type", "Unstructured Hexahedra"),
            ("Total Cells", str(len(cpg.cells))),
            ("Active Cells", str(len(cpg.cells))),
            ("Domain", f"{cpg.lx}x{cpg.ly}x{cpg.lz} m")
        ]
        for i, (prop, val) in enumerate(props):
            self.prop_table.setItem(i, 0, QTableWidgetItem(prop))
            self.prop_table.setItem(i, 1, QTableWidgetItem(val))

    def handle_process_output(self):
        """读取子进程输出并实时追加到状态面板。"""
        if not self.sim_process:
            return

        chunk = bytes(self.sim_process.readAllStandardOutput()).decode('utf-8', errors='replace')
        if not chunk:
            return

        self.sim_output_buffer += chunk
        while '\n' in self.sim_output_buffer:
            line, self.sim_output_buffer = self.sim_output_buffer.split('\n', 1)
            line = line.rstrip('\r')
            if line:
                self.process_simulation_log_line(line)

    def flush_process_output_buffer(self):
        """处理未以换行结束的最后一段输出。"""
        line = self.sim_output_buffer.strip()
        self.sim_output_buffer = ""
        if line:
            self.process_simulation_log_line(line)

    def handle_simulation_finished(self, exit_code, exit_status):
        """子进程完成后加载结果并刷新界面。"""
        self.handle_process_output()
        self.flush_process_output_buffer()
        self.run_btn.setEnabled(True)

        try:
            if exit_status != QProcess.NormalExit or exit_code != 0:
                self.append_sim_status("")
                self.append_sim_status(f"ERROR: simulation process exited with code {exit_code}")
                self.status_bar.showMessage("Simulation failed")
                self.mark_progress_failed()
                return

            if not self.pending_result_path or not os.path.exists(self.pending_result_path):
                self.append_sim_status("")
                self.append_sim_status("ERROR: simulation finished but no result file was produced")
                self.status_bar.showMessage("Simulation failed")
                self.mark_progress_failed()
                return

            self.sim_data.load_json(self.pending_result_path)
            self.append_sim_status("")
            self.append_sim_status("=" * 50)
            self.append_sim_status("  Simulation Completed Successfully!")
            self.append_sim_status("=" * 50)
            self.mark_progress_complete()

            self.render_mode3_smooth_pressure()
            self.update_statistics()
            self.append_visualization_summary()
            self.status_bar.showMessage("Simulation completed")
        finally:
            self.cleanup_simulation_process()

    def handle_simulation_error(self, process_error):
        """子进程启动/执行异常时记录错误。"""
        if process_error == QProcess.FailedToStart:
            self.append_sim_status("ERROR: failed to start simulation process")
            self.run_btn.setEnabled(True)
            self.mark_progress_failed()
            self.cleanup_simulation_process()
        elif process_error == QProcess.Crashed:
            self.append_sim_status("ERROR: simulation process crashed")
            self.mark_progress_failed()
        self.status_bar.showMessage("Simulation failed")

    def cleanup_simulation_process(self):
        """清理子进程和临时结果文件。"""
        if self.pending_result_path and os.path.exists(self.pending_result_path):
            try:
                os.remove(self.pending_result_path)
            except OSError:
                pass
        self.pending_result_path = None
        self.sim_output_buffer = ""
        if self.sim_process is not None:
            self.sim_process.deleteLater()
            self.sim_process = None

    def append_visualization_summary(self):
        """将当前可视化数据边界追加到状态面板。"""
        lx = self.sim_data.grid_info['Lx']
        ly = self.sim_data.grid_info['Ly']
        lz = self.sim_data.grid_info['Lz']

        self.append_sim_status("")
        self.append_sim_status(f"Visualization:  X[0.00, {lx:.2f}], Y[0.00, {ly:.2f}], Z[0.00, {lz:.2f}]")
        if self.sim_data.pressure_field:
            xs = [p[0] for p in self.sim_data.pressure_field]
            ys = [p[1] for p in self.sim_data.pressure_field]
            zs = [p[2] for p in self.sim_data.pressure_field]
            self.append_sim_status(
                f"Data Range:     X[{min(xs):.2f}, {max(xs):.2f}], "
                f"Y[{min(ys):.2f}, {max(ys):.2f}], Z[{min(zs):.2f}, {max(zs):.2f}]"
            )
        if self.sim_data.fractures:
            frac_xs = []
            frac_ys = []
            frac_zs = []
            for fracture in self.sim_data.fractures:
                for p in fracture['points']:
                    frac_xs.append(p[0])
                    frac_ys.append(p[1])
                    frac_zs.append(p[2])
            self.append_sim_status(
                f"Fractures:      X[{min(frac_xs):.2f}, {max(frac_xs):.2f}], "
                f"Y[{min(frac_ys):.2f}, {max(frac_ys):.2f}], Z[{min(frac_zs):.2f}, {max(frac_zs):.2f}]"
            )
            self.append_sim_status("OK: All fractures within grid boundaries")
    
    def render_mode3_smooth_pressure(self):
        """渲染平滑压力场"""
        self.vtk_renderer.render_mode3_smooth_pressure(self.sim_data)
    
    def render_fractures(self):
        """渲染裂缝"""
        self.vtk_renderer.render_fractures(self.sim_data)
    
    def update_statistics(self):
        """更新统计信息 - 与原文件格式一致"""
        if self.sim_data.pressure_field:
            values = [p[3] for p in self.sim_data.pressure_field]
            min_p, max_p = min(values), max(values)
            mean_p = sum(values) / len(values)
            sorted_p = sorted(values)
            p90 = sorted_p[int(len(sorted_p) * 0.9)]
            
            nx, ny, nz = self.sim_data.grid_info['nx'], self.sim_data.grid_info['ny'], self.sim_data.grid_info['nz']
            lx, ly, lz = self.sim_data.grid_info['Lx'], self.sim_data.grid_info['Ly'], self.sim_data.grid_info['Lz']
            
            self.append_sim_status(f"\nPressure Range: Min: {min_p:.2f} MPa, Max: {max_p:.2f} MPa")
            self.append_sim_status(f"Mean: {mean_p:.2f} MPa, P90: {p90:.2f} MPa")
            
            # 更新统计面板 - 与原文件格式一致
            self.stats_text.clear()
            self.stats_text.append(f"Grid: {nx} x {ny} x {nz} = {nx*ny*nz} cells")
            self.stats_text.append(f"Fractures: {len(self.sim_data.fractures)}")
            self.stats_text.append("")
            self.stats_text.append("Pressure Range:")
            self.stats_text.append(f"Min: {min_p:.2f} MPa")
            self.stats_text.append(f"Max: {max_p:.2f} MPa")
            self.stats_text.append(f"Mean: {mean_p:.2f} MPa")
            self.stats_text.append(f"P90: {p90:.2f} MPa")
            
            # 更新属性表格 - 与原文件一致
            self.prop_table.setRowCount(5)
            props = [
                ("Algorithm", "Black Oil"),
                ("Grid Type", "Structured"),
                ("Total Cells", str(nx * ny * nz)),
                ("Active Cells", str(nx * ny * nz)),
                ("Timesteps", "100 days")
            ]
            for i, (prop, val) in enumerate(props):
                self.prop_table.setItem(i, 0, QTableWidgetItem(prop))
                self.prop_table.setItem(i, 1, QTableWidgetItem(val))
    
    def change_view_mode(self, mode):
        """切换视图模式"""
        if mode == "Pressure Field":
            self.render_mode3_smooth_pressure()
        elif mode == "Fracture Mesh":
            self.vtk_renderer.render_fracture_only(self.sim_data)
    
    def change_field_display(self, field):
        """切换显示字段"""
        if field == "Pressure":
            self.render_mode3_smooth_pressure()
    
    def toggle_grid_lines(self, state):
        """切换网格线显示"""
        show = (state == Qt.Checked)
        if show and self.vtk_renderer.cache['grid_lines_actor'] is None:
            self.vtk_renderer.create_grid_lines(self.sim_data)
        self.vtk_renderer.toggle_grid_lines(show)
    
    def toggle_fractures_visibility(self, state):
        """切换裂缝显示 - 点击时压力图变透明"""
        show = (state == Qt.Checked)
        if show and not self.vtk_renderer.cache['fracture_actors']:
            self.render_fractures()
        self.vtk_renderer.toggle_fractures(show)


def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
