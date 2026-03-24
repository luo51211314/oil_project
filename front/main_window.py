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
                             QProgressBar, QScrollArea, QFileDialog, QDialog, QDialogButtonBox,
                             QCheckBox, QTableWidgetItem)
from PyQt5.QtCore import Qt, QSize, QProcess, QTimer
from PyQt5.QtGui import QIcon, QFont, QColor, QPixmap, QPainter
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk

# 导入本地模块
from .data_models import SimulationData, CornerPointCell, CornerPointGridData

# 导入可视化模块
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from visual.vtk_renderer import VTKRenderer


class AlgorithmSelector(QWidget):
    """算法选择器 - 统一注册方式"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(28)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(5, 2, 5, 2)
        self.layout.setSpacing(8)
        
        self.icon_label = QLabel()
        self.icon_label.setFixedSize(20, 20)
        self.icon_label.setPixmap(self.create_colorful_icon())
        
        # 注册可用算法
        self.algorithms = {
            "black_oil": {"name": "Black Oil", "color": "#4CAF50", "active_color": "#4CAF50"},
            "black_oil_corner_grid": {"name": "Corner Grid", "color": "#2196F3", "active_color": "#FF9800"},
        }
        self.disabled_algos = ["Comp", "Thermal", "Foam", "Polymer"]
        
        self.algo_labels = {}
        for algo_id, config in self.algorithms.items():
            lbl = QLabel(config["name"])
            lbl.setStyleSheet(f"color: {config['color']}; font-weight: bold; font-size: 12px;")
            lbl.setCursor(Qt.PointingHandCursor)
            lbl.mousePressEvent = lambda event, a=algo_id: self.on_algo_click(a)
            self.algo_labels[algo_id] = lbl
            self.layout.addWidget(lbl)
        
        for algo in self.disabled_algos:
            lbl = QLabel(algo)
            lbl.setStyleSheet("color: #666666; font-size: 10px;")
            self.layout.addWidget(lbl)
        
        self.layout.addStretch()
        
        self.current_algorithm = "black_oil"
        self.on_algorithm_changed = None
    
    def on_algo_click(self, algo_id):
        """统一算法点击处理"""
        self.set_current_algorithm(algo_id)
    
    def set_current_algorithm(self, algo_id):
        """设置当前算法并更新样式"""
        self.current_algorithm = algo_id
        for aid, lbl in self.algo_labels.items():
            config = self.algorithms[aid]
            if aid == algo_id:
                lbl.setStyleSheet(f"color: {config['active_color']}; font-weight: bold; font-size: 12px;")
            else:
                lbl.setStyleSheet(f"color: #666666; font-weight: bold; font-size: 12px;")
        
        if self.on_algorithm_changed:
            self.on_algorithm_changed(algo_id)
    
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
        self.corner_selection_mode_active = False
        self.corner_selection_dragging = False
        self.corner_selection_start_xy = None
        self.corner_selection_style = None
        self.corner_selection_style_observer_ids = []
        self.corner_selection_previous_style = None
        self.corner_selection_saved_camera = None
        self.corner_selection_cube_source = None
        self.corner_selection_actor = None
        self.corner_selection_outline_actor = None
        self.corner_selection_handle_actor = None
        self.selection_tool_controls = {}
        self.selection_params_by_algorithm = {}
        
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
        if self.corner_selection_mode_active:
            self.deactivate_corner_rectangle_selection_mode()
        self.current_algorithm = algo
        self.update_algorithm_parameter_pages()
        self.switch_tab(self.current_tab)
        if hasattr(self, 'status_bar'):
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
        
        # 按算法切换的参数区
        self.algorithm_param_stack = QStackedWidget()
        self.algorithm_param_stack.setStyleSheet("background-color: #2b2b2b;")

        # Black Oil 参数堆叠窗口
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

        # Corner Grid 参数堆叠窗口
        self.corner_param_stack = QStackedWidget()
        self.corner_param_stack.setStyleSheet("background-color: #2b2b2b;")

        self.corner_grid_page = self.create_corner_grid_page()
        self.corner_param_stack.addWidget(self.corner_grid_page)

        self.corner_wells_page = self.create_corner_wells_page()
        self.corner_param_stack.addWidget(self.corner_wells_page)

        self.corner_fractures_page = self.create_corner_fractures_page()
        self.corner_param_stack.addWidget(self.corner_fractures_page)

        self.corner_results_page = self.create_corner_results_page()
        self.corner_param_stack.addWidget(self.corner_results_page)

        self.algorithm_param_stack.addWidget(self.param_stack)
        self.algorithm_param_stack.addWidget(self.corner_param_stack)
        self.left_layout.addWidget(self.algorithm_param_stack)
        self.update_algorithm_parameter_pages()
    
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
        layout.addStretch()

        return self.wrap_in_scroll_area(content)

    def create_basic_hydraulic_fractures_group(self):
        """未加密模式的人工裂缝参数分组。"""
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

        group = QGroupBox("Hydraulic Fractures")
        group.setStyleSheet(self.groupbox_style())
        layout = QGridLayout()
        layout.addWidget(self.basic_check_enable_hf, 0, 0, 1, 2)
        layout.addWidget(QLabel("裂缝数量:"), 1, 0)
        layout.addWidget(self.basic_spin_hf_count, 1, 1)
        layout.addWidget(QLabel("Well Length (m):"), 2, 0)
        layout.addWidget(self.basic_spin_hf_well_length, 2, 1)
        layout.addWidget(QLabel("裂缝长度 (m):"), 3, 0)
        layout.addWidget(self.basic_spin_hf_length, 3, 1)
        layout.addWidget(QLabel("裂缝高度 (m):"), 4, 0)
        layout.addWidget(self.basic_spin_hf_height, 4, 1)
        layout.addWidget(QLabel("裂缝开度 (m):"), 5, 0)
        layout.addWidget(self.basic_spin_hf_aperture, 5, 1)
        layout.addWidget(QLabel("裂缝渗透率 (Darcy):"), 6, 0)
        layout.addWidget(self.basic_spin_hf_perm, 6, 1)
        layout.addWidget(QLabel("中心 X (m):"), 7, 0)
        layout.addWidget(self.basic_spin_hf_center_x, 7, 1)
        layout.addWidget(QLabel("中心 Y (m):"), 8, 0)
        layout.addWidget(self.basic_spin_hf_center_y, 8, 1)
        layout.addWidget(QLabel("中心 Z (m):"), 9, 0)
        layout.addWidget(self.basic_spin_hf_center_z, 9, 1)
        group.setLayout(layout)
        return group

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
        layout.addWidget(self.create_basic_hydraulic_fractures_group())
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
    
    def create_results_page(self, algorithm_key="black_oil"):
        """创建 Results 页面。"""
        page = QWidget()
        page.setStyleSheet("background-color: #2b2b2b;")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(10, 10, 10, 10)

        view_group = QGroupBox("View Mode")
        view_group.setStyleSheet(self.groupbox_style())
        view_layout = QVBoxLayout()

        view_mode_combo = QComboBox()
        view_mode_combo.addItems(["Pressure Field", "Fracture Mesh"])
        view_mode_combo.setStyleSheet("color: #cccccc; background-color: #3d3d3d;")
        view_mode_combo.currentTextChanged.connect(self.change_view_mode)

        view_layout.addWidget(QLabel("Select View:"))
        view_layout.addWidget(view_mode_combo)
        view_group.setLayout(view_layout)
        layout.addWidget(view_group)

        group = QGroupBox("Visualization Options")
        group.setStyleSheet(self.groupbox_style())
        vlayout = QVBoxLayout()

        combo_field = QComboBox()
        combo_field.addItems(["Pressure", "Temperature", "Stress"])
        combo_field.setStyleSheet("color: #cccccc; background-color: #3d3d3d;")
        combo_field.currentTextChanged.connect(self.change_field_display)

        check_show_grid = QCheckBox("Show Grid Lines")
        check_show_grid.setChecked(False)
        check_show_grid.stateChanged.connect(self.toggle_grid_lines)

        check_show_fractures = QCheckBox("Show Fractures")
        check_show_fractures.setChecked(False)
        check_show_fractures.stateChanged.connect(self.toggle_fractures_visibility)

        vlayout.addWidget(QLabel("Display Field:"))
        vlayout.addWidget(combo_field)
        vlayout.addWidget(check_show_grid)
        vlayout.addWidget(check_show_fractures)

        group.setLayout(vlayout)
        layout.addWidget(group)

        if algorithm_key in {"black_oil", "black_oil_corner_grid"}:
            layout.addWidget(self.create_selection_tools_group(algorithm_key))

        layout.addStretch()

        self.register_results_controls(
            algorithm_key,
            view_mode_combo,
            combo_field,
            check_show_grid,
            check_show_fractures,
        )
        return page

    def create_selection_tools_group(self, algorithm_key):
        """创建结果页的框选工具分组。"""
        group = QGroupBox("Selection Tools")
        group.setStyleSheet(self.groupbox_style())
        layout = QVBoxLayout()

        toggle_btn = QPushButton("开始矩形框选")
        toggle_btn.setCheckable(True)
        toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #3d3d3d;
                color: #cccccc;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 6px 10px;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
            }
            QPushButton:checked {
                background-color: #1565C0;
                color: white;
            }
        """)
        toggle_btn.toggled.connect(self.toggle_corner_rectangle_selection_mode)

        status_label = QLabel("未选择区域")
        status_label.setWordWrap(True)
        status_label.setStyleSheet("color: #8f8f8f; padding: 4px 0;")

        self.selection_tool_controls[algorithm_key] = {
            'toggle_btn': toggle_btn,
            'status_label': status_label,
        }

        layout.addWidget(toggle_btn)
        layout.addWidget(status_label)
        group.setLayout(layout)
        return group

    def create_corner_grid_page(self):
        """创建 Corner Grid 的 Grid 页面。"""
        page = QWidget()
        page.setStyleSheet("background-color: #2b2b2b;")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(10, 10, 10, 10)

        options_group = QGroupBox("Grid Options")
        options_group.setStyleSheet(self.groupbox_style())
        options_layout = QGridLayout()

        self.corner_combo_grid_refinement = QComboBox()
        self.corner_combo_grid_refinement.addItems(["不加密", "加密"])
        self.corner_combo_grid_refinement.setStyleSheet("color: #cccccc; background-color: #3d3d3d;")

        options_layout.addWidget(QLabel("是否加密:"), 0, 0)
        options_layout.addWidget(self.corner_combo_grid_refinement, 0, 1)
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        file_group = QGroupBox("File Import")
        file_group.setStyleSheet(self.groupbox_style())
        file_layout = QVBoxLayout()

        self.corner_grid_file_path = ""
        self.corner_import_csv_btn = QPushButton("导入 CSV 文件")
        self.corner_import_csv_btn.setStyleSheet("""
            QPushButton {
                background-color: #3d3d3d;
                color: #cccccc;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 6px 10px;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
            }
        """)
        self.corner_import_csv_btn.clicked.connect(self.select_corner_grid_csv_file)

        self.corner_grid_file_name_label = QLabel("未选择文件")
        self.corner_grid_file_name_label.setWordWrap(True)
        self.corner_grid_file_name_label.setStyleSheet("color: #8f8f8f; padding: 4px 0;")

        file_layout.addWidget(self.corner_import_csv_btn)
        file_layout.addWidget(self.corner_grid_file_name_label)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        layout.addStretch()
        return page

    def create_corner_wells_page(self):
        """创建 Corner Grid 的 Wells 占位页面。"""
        return self.create_placeholder_param_page([
            ("Well Parameters", "Corner Grid 的井参数将放在这里。"),
        ])

    def create_corner_fractures_page(self):
        """创建 Corner Grid 的 Fractures 占位页面。"""
        return self.create_placeholder_param_page([
            ("Natural Fractures", "Corner Grid 的天然裂缝参数将放在这里。"),
            ("Hydraulic Fractures", "Corner Grid 的人工裂缝参数将放在这里。"),
        ])

    def create_corner_results_page(self):
        """创建 Corner Grid 的 Results 页面。"""
        return self.create_results_page("black_oil_corner_grid")

    def create_placeholder_param_page(self, groups):
        """创建仅包含分组框和说明文字的占位参数页。"""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        for title, message in groups:
            layout.addWidget(self.create_placeholder_group(title, message))

        layout.addStretch()
        return self.wrap_in_scroll_area(content)

    def create_placeholder_group(self, title, message):
        """创建占位分组框。"""
        group = QGroupBox(title)
        group.setStyleSheet(self.groupbox_style())
        layout = QVBoxLayout()

        label = QLabel(message)
        label.setWordWrap(True)
        label.setStyleSheet("color: #8f8f8f; padding: 6px 0;")
        layout.addWidget(label)

        group.setLayout(layout)
        return group

    def select_corner_grid_csv_file(self):
        """为 Corner Grid 选择 CSV 输入文件，仅更新界面显示。"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 Corner Grid CSV 文件",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not file_path:
            return

        self.corner_grid_file_path = file_path
        self.corner_grid_file_name_label.setText(os.path.basename(file_path))

    def register_results_controls(self, algorithm_key, view_mode_combo, combo_field,
                                  check_show_grid, check_show_fractures):
        """登记各算法 Results 页对应的独立控件引用。"""
        if not hasattr(self, 'results_controls'):
            self.results_controls = {}

        self.results_controls[algorithm_key] = {
            'view_mode_combo': view_mode_combo,
            'combo_field': combo_field,
            'check_show_grid': check_show_grid,
            'check_show_fractures': check_show_fractures,
        }

        if algorithm_key == "black_oil":
            self.view_mode_combo = view_mode_combo
            self.combo_field = combo_field
            self.check_show_grid = check_show_grid
            self.check_show_fractures = check_show_fractures

    def update_algorithm_parameter_pages(self):
        """根据当前算法切换左侧参数页集合。"""
        if not hasattr(self, 'algorithm_param_stack'):
            return

        if self.current_algorithm == "black_oil_corner_grid":
            self.algorithm_param_stack.setCurrentWidget(self.corner_param_stack)
        else:
            self.algorithm_param_stack.setCurrentWidget(self.param_stack)

    def get_active_param_stack(self):
        """获取当前算法对应的页签堆叠窗口。"""
        if self.current_algorithm == "black_oil_corner_grid" and hasattr(self, 'corner_param_stack'):
            return self.corner_param_stack
        return self.param_stack
    
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
        should_keep_selection = (tab_name == "Results" and self.current_algorithm in {"black_oil", "black_oil_corner_grid"})
        if self.corner_selection_mode_active and not should_keep_selection:
            self.deactivate_corner_rectangle_selection_mode()

        self.current_tab = tab_name
        for name, btn in self.tab_buttons.items():
            btn.setChecked(name == tab_name)
        
        tab_index = {"Grid": 0, "Wells": 1, "Fractures": 2, "Results": 3}
        self.get_active_param_stack().setCurrentIndex(tab_index.get(tab_name, 0))
        self.sync_selection_tool_status()
    
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
        self.deactivate_corner_rectangle_selection_mode(restore_camera=False, clear_actor=True)
        self.clear_corner_selection_overlay(clear_params=False)
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

        if self.current_algorithm == "black_oil":
            params.update(self.collect_black_oil_region_fracture_params(params))

        return params

    def collect_black_oil_region_fracture_params(self, params):
        """将 Black Oil 的框选区域参数整理成算法输入。"""
        selection = self.selection_params_by_algorithm.get("black_oil")
        if not selection:
            return {
                'region_num_fracs': 0,
                'region_x_min': 0.0,
                'region_x_max': 0.0,
                'region_y_min': 0.0,
                'region_y_max': 0.0,
                'region_z_min': 0.0,
                'region_z_max': 0.0,
            }

        return {
            'region_num_fracs': int(selection['N']),
            'region_x_min': float(selection['x1']),
            'region_x_max': float(selection['x2']),
            'region_y_min': float(selection['y1']),
            'region_y_max': float(selection['y2']),
            'region_z_min': 0.0,
            'region_z_max': float(params['lz']),
        }

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
        if params.get('region_num_fracs', 0) > 0:
            self.append_sim_status(
                "Region Fractures: "
                f"N={params['region_num_fracs']}, "
                f"X[{params['region_x_min']:.2f}, {params['region_x_max']:.2f}], "
                f"Y[{params['region_y_min']:.2f}, {params['region_y_max']:.2f}], "
                f"Z[{params['region_z_min']:.2f}, {params['region_z_max']:.2f}]"
            )
        else:
            self.append_sim_status("Region Fractures: disabled")
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

    def toggle_corner_rectangle_selection_mode(self, checked):
        """切换 Corner Grid 结果页的矩形框选模式。"""
        if checked:
            self.activate_corner_rectangle_selection_mode()
        else:
            self.deactivate_corner_rectangle_selection_mode()

    def activate_corner_rectangle_selection_mode(self):
        """进入 Corner Grid 的矩形框选模式。"""
        if self.current_algorithm not in {"black_oil", "black_oil_corner_grid"}:
            self.set_corner_selection_toggle_button(False)
            return

        if self.current_tab != "Results":
            self.set_corner_selection_toggle_button(False)
            return

        if not self.has_corner_selection_data():
            self.update_corner_selection_status("请先运行当前算法模拟后再进行框选。")
            self.status_bar.showMessage("Selection unavailable")
            self.set_corner_selection_toggle_button(False)
            return

        controls = getattr(self, 'results_controls', {}).get(self.current_algorithm, {})
        view_mode_combo = controls.get('view_mode_combo')
        combo_field = controls.get('combo_field')
        if view_mode_combo and view_mode_combo.currentText() != "Pressure Field":
            view_mode_combo.setCurrentText("Pressure Field")
        if combo_field and combo_field.currentText() != "Pressure":
            combo_field.setCurrentText("Pressure")

        self.clear_corner_selection_overlay(clear_params=True)
        self.corner_selection_saved_camera = self.capture_camera_state()
        self.configure_corner_selection_camera()

        iren = self.vtk_widget.iren
        self.corner_selection_previous_style = iren.GetInteractorStyle()
        self.corner_selection_style = vtk.vtkInteractorStyleUser()
        self.corner_selection_style_observer_ids = [
            self.corner_selection_style.AddObserver("LeftButtonPressEvent", self.handle_corner_selection_press),
            self.corner_selection_style.AddObserver("MouseMoveEvent", self.handle_corner_selection_move),
            self.corner_selection_style.AddObserver("LeftButtonReleaseEvent", self.handle_corner_selection_release),
        ]
        iren.SetInteractorStyle(self.corner_selection_style)

        self.corner_selection_mode_active = True
        self.corner_selection_dragging = False
        self.corner_selection_start_xy = None
        self.set_corner_selection_toggle_button(True)
        self.vtk_widget.vtk_widget.setCursor(Qt.CrossCursor)
        self.update_corner_selection_status("拖拽鼠标框选 XY 区域。")
        self.status_bar.showMessage("Corner Grid rectangle selection enabled")
        self.vtk_widget.iren.Render()

    def deactivate_corner_rectangle_selection_mode(self, restore_camera=True, clear_actor=False):
        """退出 Corner Grid 的矩形框选模式。"""
        iren = getattr(self, 'vtk_widget', None)
        if iren is not None:
            iren = self.vtk_widget.iren

        if iren and self.corner_selection_previous_style is not None:
            iren.SetInteractorStyle(self.corner_selection_previous_style)

        if self.corner_selection_style is not None:
            for observer_id in self.corner_selection_style_observer_ids:
                self.corner_selection_style.RemoveObserver(observer_id)

        self.corner_selection_style = None
        self.corner_selection_style_observer_ids = []
        self.corner_selection_previous_style = None
        self.corner_selection_mode_active = False
        self.corner_selection_dragging = False
        self.corner_selection_start_xy = None

        if restore_camera and self.corner_selection_saved_camera:
            self.restore_camera_state(self.corner_selection_saved_camera)
        self.corner_selection_saved_camera = None

        if clear_actor or self.get_current_selection_params() is None:
            self.clear_corner_selection_overlay(clear_params=False)

        if hasattr(self, 'vtk_widget'):
            self.vtk_widget.vtk_widget.unsetCursor()
            self.vtk_widget.iren.Render()

        self.set_corner_selection_toggle_button(False)
        if hasattr(self, 'status_bar'):
            self.status_bar.showMessage("Ready - Click 'Run Simulation' to start")

    def handle_corner_selection_press(self, caller, event):
        """开始 Corner Grid 矩形框选。"""
        if not self.corner_selection_mode_active:
            return

        x, y = self.vtk_widget.iren.GetEventPosition()
        self.corner_selection_start_xy = self.display_to_corner_world_xy(x, y)
        self.corner_selection_dragging = True
        self.update_corner_selection_preview(
            self.corner_selection_start_xy,
            self.corner_selection_start_xy,
            finalized=False,
        )

    def handle_corner_selection_move(self, caller, event):
        """更新 Corner Grid 矩形框选预览。"""
        if not (self.corner_selection_mode_active and self.corner_selection_dragging):
            return

        x, y = self.vtk_widget.iren.GetEventPosition()
        current_xy = self.display_to_corner_world_xy(x, y)
        self.update_corner_selection_preview(
            self.corner_selection_start_xy,
            current_xy,
            finalized=False,
        )

    def handle_corner_selection_release(self, caller, event):
        """完成 Corner Grid 矩形框选并弹出参数输入窗。"""
        if not (self.corner_selection_mode_active and self.corner_selection_dragging):
            return

        self.corner_selection_dragging = False
        x, y = self.vtk_widget.iren.GetEventPosition()
        end_xy = self.display_to_corner_world_xy(x, y)
        bounds = self.normalize_corner_xy_bounds(self.corner_selection_start_xy, end_xy)

        if not self.corner_selection_bounds_valid(bounds):
            self.clear_corner_selection_overlay(clear_params=False)
            self.update_corner_selection_status("框选区域过小，请重新选择。")
            return

        self.update_corner_selection_preview(bounds[:2], bounds[2:], finalized=True)
        QTimer.singleShot(0, lambda b=bounds: self.finish_corner_selection_release(b))

    def finish_corner_selection_release(self, bounds):
        """在 Qt 主事件循环中完成选区参数输入，避免 VTK 回调中直接弹窗闪退。"""
        params = self.open_corner_selection_parameter_dialog(bounds)
        if params is None:
            self.clear_corner_selection_overlay(clear_params=False)
            self.update_corner_selection_status("已取消当前选区。")
            self.deactivate_corner_rectangle_selection_mode()
            return

        self.selection_params_by_algorithm[self.current_algorithm] = params
        self.sync_selection_tool_status()
        self.deactivate_corner_rectangle_selection_mode()

    def display_to_corner_world_xy(self, display_x, display_y):
        """将屏幕坐标映射到俯视平行投影视图下的 XY 坐标。"""
        renderer = self.vtk_widget.renderer
        render_window = self.vtk_widget.vtk_widget.GetRenderWindow()
        width, height = render_window.GetSize()
        if width <= 0 or height <= 0:
            return 0.0, 0.0

        viewport = renderer.GetViewport()
        view_x0 = viewport[0] * width
        view_y0 = viewport[1] * height
        view_width = max(1.0, (viewport[2] - viewport[0]) * width)
        view_height = max(1.0, (viewport[3] - viewport[1]) * height)

        u = max(0.0, min(1.0, (display_x - view_x0) / view_width))
        v = max(0.0, min(1.0, (display_y - view_y0) / view_height))

        camera = renderer.GetActiveCamera()
        focal_x, focal_y, _ = camera.GetFocalPoint()
        world_height = 2.0 * camera.GetParallelScale()
        world_width = world_height * (view_width / view_height)

        x = focal_x + (u - 0.5) * world_width
        y = focal_y + (v - 0.5) * world_height

        min_x, max_x, min_y, max_y, _, _ = self.get_corner_selection_world_bounds()
        x = max(min_x, min(max_x, x))
        y = max(min_y, min(max_y, y))
        return x, y

    def normalize_corner_xy_bounds(self, start_xy, end_xy):
        """归一化矩形框选的 XY 边界。"""
        x1, y1 = start_xy
        x2, y2 = end_xy
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    def corner_selection_bounds_valid(self, bounds):
        """检查当前矩形框选是否形成有效区域。"""
        x1, y1, x2, y2 = bounds
        min_x, max_x, min_y, max_y, _, _ = self.get_corner_selection_world_bounds()
        eps_x = max(1e-6, (max_x - min_x) * 0.005)
        eps_y = max(1e-6, (max_y - min_y) * 0.005)
        return (x2 - x1) >= eps_x and (y2 - y1) >= eps_y

    def get_corner_selection_world_bounds(self):
        """获取 Corner Grid 当前可框选的世界坐标边界。"""
        if self.current_algorithm == "black_oil_corner_grid":
            cpg = self.sim_data.corner_point_grid
            if cpg and cpg.cells:
                min_x = min_y = min_z = float('inf')
                max_x = max_y = max_z = float('-inf')
                for cell in cpg.cells:
                    for corner in cell.corners:
                        min_x = min(min_x, corner[0])
                        max_x = max(max_x, corner[0])
                        min_y = min(min_y, corner[1])
                        max_y = max(max_y, corner[1])
                        min_z = min(min_z, corner[2])
                        max_z = max(max_z, corner[2])
                return min_x, max_x, min_y, max_y, min_z, max_z

        info = self.sim_data.grid_info
        return 0.0, float(info['Lx']), 0.0, float(info['Ly']), 0.0, float(info['Lz'])

    def has_corner_selection_data(self):
        """判断 Corner Grid 是否已有可供框选的结果数据。"""
        if self.current_algorithm == "black_oil_corner_grid":
            return bool(self.sim_data.corner_point_grid and self.sim_data.corner_point_grid.cells)
        if self.current_algorithm == "black_oil":
            return bool(self.sim_data.pressure_field)
        return False

    def capture_camera_state(self):
        """保存当前相机状态，便于退出框选模式后恢复。"""
        camera = self.vtk_widget.renderer.GetActiveCamera()
        return {
            'position': camera.GetPosition(),
            'focal_point': camera.GetFocalPoint(),
            'view_up': camera.GetViewUp(),
            'parallel_projection': camera.GetParallelProjection(),
            'parallel_scale': camera.GetParallelScale(),
        }

    def restore_camera_state(self, state):
        """恢复进入框选模式前的相机状态。"""
        camera = self.vtk_widget.renderer.GetActiveCamera()
        camera.SetPosition(*state['position'])
        camera.SetFocalPoint(*state['focal_point'])
        camera.SetViewUp(*state['view_up'])
        if state['parallel_projection']:
            camera.ParallelProjectionOn()
        else:
            camera.ParallelProjectionOff()
        camera.SetParallelScale(state['parallel_scale'])
        self.vtk_widget.renderer.ResetCameraClippingRange()

    def configure_corner_selection_camera(self):
        """将相机切到俯视平行投影视图，便于 XY 矩形框选。"""
        min_x, max_x, min_y, max_y, min_z, max_z = self.get_corner_selection_world_bounds()
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        cz = (min_z + max_z) / 2.0
        dx = max_x - min_x
        dy = max_y - min_y
        dz = max_z - min_z

        render_window = self.vtk_widget.vtk_widget.GetRenderWindow()
        width, height = render_window.GetSize()
        aspect = (width / height) if height else 1.0

        camera = self.vtk_widget.renderer.GetActiveCamera()
        camera.SetFocalPoint(cx, cy, cz)
        camera.SetPosition(cx, cy, max_z + max(dx, dy, dz, 1.0) * 3.0)
        camera.SetViewUp(0.0, 1.0, 0.0)
        camera.ParallelProjectionOn()
        camera.SetParallelScale(max(dy / 2.0, dx / max(2.0 * aspect, 1e-6), 1.0) * 1.05)
        self.vtk_widget.renderer.ResetCameraClippingRange()

    def update_corner_selection_preview(self, start_xy, end_xy, finalized=False):
        """更新 Corner Grid 框选区域的三维可视化预览。"""
        min_x, min_y, max_x, max_y = self.normalize_corner_xy_bounds(start_xy, end_xy)
        _, _, _, _, min_z, max_z = self.get_corner_selection_world_bounds()

        if self.corner_selection_cube_source is None:
            self.corner_selection_cube_source = vtk.vtkCubeSource()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(self.corner_selection_cube_source.GetOutputPort())
            self.corner_selection_actor = vtk.vtkActor()
            self.corner_selection_actor.SetMapper(mapper)
            self.vtk_widget.renderer.AddActor(self.corner_selection_actor)

        self.corner_selection_cube_source.SetBounds(min_x, max_x, min_y, max_y, min_z, max_z)
        self.corner_selection_cube_source.Update()

        prop = self.corner_selection_actor.GetProperty()
        prop.SetEdgeVisibility(1)
        prop.SetLineWidth(5.5)
        prop.SetAmbient(0.55)
        prop.SetDiffuse(0.8)
        prop.SetSpecular(0.35)
        prop.SetSpecularPower(18.0)
        prop.SetInterpolationToFlat()
        if hasattr(prop, "SetRenderLinesAsTubes"):
            prop.SetRenderLinesAsTubes(1)
        if finalized:
            prop.SetColor(1.0, 0.08, 0.0)
            prop.SetOpacity(0.58)
            prop.SetEdgeColor(0.2, 1.0, 0.2)
        else:
            prop.SetColor(0.0, 1.0, 1.0)
            prop.SetOpacity(0.42)
            prop.SetEdgeColor(1.0, 1.0, 1.0)

        self.update_corner_selection_outline(min_x, max_x, min_y, max_y, min_z, max_z, finalized)
        self.vtk_widget.iren.Render()

    def update_corner_selection_outline(self, min_x, max_x, min_y, max_y, min_z, max_z, finalized):
        """用亮色管状轮廓和角点标记增强选区可见性。"""
        if self.corner_selection_outline_actor is not None:
            self.vtk_widget.renderer.RemoveActor(self.corner_selection_outline_actor)
            self.corner_selection_outline_actor = None
        if self.corner_selection_handle_actor is not None:
            self.vtk_widget.renderer.RemoveActor(self.corner_selection_handle_actor)
            self.corner_selection_handle_actor = None

        points = vtk.vtkPoints()
        corners = [
            (min_x, min_y, min_z),
            (max_x, min_y, min_z),
            (max_x, max_y, min_z),
            (min_x, max_y, min_z),
            (min_x, min_y, max_z),
            (max_x, min_y, max_z),
            (max_x, max_y, max_z),
            (min_x, max_y, max_z),
        ]
        for corner in corners:
            points.InsertNextPoint(*corner)

        lines = vtk.vtkCellArray()
        edge_pairs = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for p0, p1 in edge_pairs:
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, p0)
            line.GetPointIds().SetId(1, p1)
            lines.InsertNextCell(line)

        edge_poly = vtk.vtkPolyData()
        edge_poly.SetPoints(points)
        edge_poly.SetLines(lines)

        max_dim = max(max_x - min_x, max_y - min_y, max_z - min_z, 1.0)
        tube = vtk.vtkTubeFilter()
        tube.SetInputData(edge_poly)
        tube.SetRadius(max_dim * (0.010 if finalized else 0.008))
        tube.SetNumberOfSides(16)
        tube.CappingOn()

        outline_mapper = vtk.vtkPolyDataMapper()
        outline_mapper.SetInputConnection(tube.GetOutputPort())
        self.corner_selection_outline_actor = vtk.vtkActor()
        self.corner_selection_outline_actor.SetMapper(outline_mapper)
        outline_prop = self.corner_selection_outline_actor.GetProperty()
        outline_prop.SetLighting(False)
        if finalized:
            outline_prop.SetColor(0.2, 1.0, 0.2)
        else:
            outline_prop.SetColor(1.0, 1.0, 1.0)
        self.vtk_widget.renderer.AddActor(self.corner_selection_outline_actor)

        append_poly = vtk.vtkAppendPolyData()
        handle_radius = max_dim * (0.018 if finalized else 0.014)
        top_corners = corners[4:8]
        for x, y, z in top_corners:
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(x, y, z)
            sphere.SetRadius(handle_radius)
            sphere.SetThetaResolution(18)
            sphere.SetPhiResolution(18)
            append_poly.AddInputConnection(sphere.GetOutputPort())
        append_poly.Update()

        handle_mapper = vtk.vtkPolyDataMapper()
        handle_mapper.SetInputConnection(append_poly.GetOutputPort())
        self.corner_selection_handle_actor = vtk.vtkActor()
        self.corner_selection_handle_actor.SetMapper(handle_mapper)
        handle_prop = self.corner_selection_handle_actor.GetProperty()
        handle_prop.SetLighting(False)
        if finalized:
            handle_prop.SetColor(0.2, 1.0, 0.2)
        else:
            handle_prop.SetColor(1.0, 1.0, 1.0)
        self.vtk_widget.renderer.AddActor(self.corner_selection_handle_actor)

    def reapply_corner_selection_overlay(self):
        """在重新渲染后恢复已确认的 Corner Grid 选区高亮。"""
        params = self.get_current_selection_params()
        if not params:
            return

        self.update_corner_selection_preview(
            (params['x1'], params['y1']),
            (params['x2'], params['y2']),
            finalized=True,
        )

    def clear_corner_selection_overlay(self, clear_params=False):
        """清除 Corner Grid 框选区域的可视化高亮。"""
        if self.corner_selection_actor is not None:
            self.vtk_widget.renderer.RemoveActor(self.corner_selection_actor)
        if self.corner_selection_outline_actor is not None:
            self.vtk_widget.renderer.RemoveActor(self.corner_selection_outline_actor)
        if self.corner_selection_handle_actor is not None:
            self.vtk_widget.renderer.RemoveActor(self.corner_selection_handle_actor)
        self.corner_selection_actor = None
        self.corner_selection_outline_actor = None
        self.corner_selection_handle_actor = None
        self.corner_selection_cube_source = None
        if clear_params:
            self.selection_params_by_algorithm.pop(self.current_algorithm, None)
            self.update_corner_selection_status("未选择区域")
        if hasattr(self, 'vtk_widget'):
            self.vtk_widget.iren.Render()

    def update_corner_selection_status(self, text):
        """更新 Corner Grid 框选工具的状态文字。"""
        controls = self.get_selection_tool_controls()
        if controls and controls.get('status_label') is not None:
            controls['status_label'].setText(text)

    def set_corner_selection_toggle_button(self, checked):
        """同步 Corner Grid 框选按钮状态，避免递归触发。"""
        controls = self.get_selection_tool_controls()
        toggle_btn = controls.get('toggle_btn') if controls else None
        if toggle_btn is None:
            return
        toggle_btn.blockSignals(True)
        toggle_btn.setChecked(checked)
        toggle_btn.setText("退出矩形框选" if checked else "开始矩形框选")
        toggle_btn.blockSignals(False)

    def get_selection_tool_controls(self, algorithm_key=None):
        """获取指定算法的框选工具控件。"""
        return self.selection_tool_controls.get(algorithm_key or self.current_algorithm)

    def get_current_selection_params(self):
        """获取当前算法的框选参数。"""
        return self.selection_params_by_algorithm.get(self.current_algorithm)

    def sync_selection_tool_status(self):
        """根据当前算法已保存的选区，刷新工具区状态文本。"""
        params = self.get_current_selection_params()
        if not params:
            self.update_corner_selection_status("未选择区域")
            return
        status = (
            f"区域: ({params['x1']:.2f}, {params['y1']:.2f}) -> "
            f"({params['x2']:.2f}, {params['y2']:.2f}), N = {params['N']}"
        )
        if self.current_algorithm == "black_oil":
            status += "\n将于下次运行时应用。"
        self.update_corner_selection_status(status)

    def open_corner_selection_parameter_dialog(self, bounds):
        """弹出矩形框选参数输入窗，返回确认后的参数。"""
        x1, y1, x2, y2 = bounds
        dialog = QDialog(self)
        dialog.setWindowTitle("天然裂缝区域参数")
        dialog.setModal(True)
        dialog.resize(460, 300)
        dialog.setMinimumSize(420, 280)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
                color: #cccccc;
            }
            QLabel {
                color: #cccccc;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #3d3d3d;
                color: #cccccc;
                border: 1px solid #555555;
                padding: 5px;
                min-height: 28px;
            }
            QPushButton {
                background-color: #3d3d3d;
                color: #cccccc;
                border: 1px solid #555555;
                padding: 5px 12px;
            }
        """)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(12)

        x1_spin = self.create_double_spinbox(-1e9, 1e9, x1, decimals=3)
        y1_spin = self.create_double_spinbox(-1e9, 1e9, y1, decimals=3)
        x2_spin = self.create_double_spinbox(-1e9, 1e9, x2, decimals=3)
        y2_spin = self.create_double_spinbox(-1e9, 1e9, y2, decimals=3)
        n_spin = self.create_spinbox(1, 100000, 1)

        for row, (label_text, widget) in enumerate([
            ("x1", x1_spin),
            ("y1", y1_spin),
            ("x2", x2_spin),
            ("y2", y2_spin),
            ("N", n_spin),
        ]):
            grid.addWidget(QLabel(f"{label_text}:"), row, 0)
            grid.addWidget(widget, row, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        layout.addLayout(grid)
        layout.addWidget(buttons)

        if dialog.exec_() != QDialog.Accepted:
            return None

        x1_val = x1_spin.value()
        y1_val = y1_spin.value()
        x2_val = x2_spin.value()
        y2_val = y2_spin.value()
        return {
            'x1': min(x1_val, x2_val),
            'y1': min(y1_val, y2_val),
            'x2': max(x1_val, x2_val),
            'y2': max(y1_val, y2_val),
            'N': n_spin.value(),
        }
    
    def render_mode3_smooth_pressure(self):
        """渲染平滑压力场"""
        if self.current_algorithm == "black_oil_corner_grid" and self.sim_data.corner_point_grid:
            self.vtk_renderer.render_corner_point_grid(self.sim_data)
            self.reapply_corner_selection_overlay()
            return
        self.vtk_renderer.render_mode3_smooth_pressure(self.sim_data)
        self.reapply_corner_selection_overlay()
    
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
        if self.current_algorithm == "black_oil_corner_grid":
            self.vtk_renderer.render_corner_point_grid(self.sim_data)
            self.reapply_corner_selection_overlay()
            return

        if mode == "Pressure Field":
            self.render_mode3_smooth_pressure()
        elif mode == "Fracture Mesh":
            self.vtk_renderer.render_fracture_only(self.sim_data)
    
    def change_field_display(self, field):
        """切换显示字段"""
        if self.current_algorithm == "black_oil_corner_grid":
            self.vtk_renderer.render_corner_point_grid(self.sim_data)
            self.reapply_corner_selection_overlay()
            return
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
