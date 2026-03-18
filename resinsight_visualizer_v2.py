import sys
import os
import numpy as np
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QComboBox, QGroupBox, 
                             QCheckBox, QSplitter, QFrame, QToolBar, QAction, QStatusBar,
                             QTabWidget, QStackedWidget, QTableWidget,
                             QTableWidgetItem, QTextEdit, QGridLayout, QSpinBox, QDoubleSpinBox,
                             QFileDialog, QMessageBox, QButtonGroup, QRadioButton)
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QFont, QColor, QPixmap, QPainter
import io
import threading

# 导入C++模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'build', 'Release'))
try:
    import edfm_core
    HAS_CPP_MODULE = True
    print("Successfully imported edfm_core module")
except ImportError as e:
    HAS_CPP_MODULE = False
    print(f"C++ module not found: {e}, using mock data")


class OutputCapture:
    """捕获C++输出并重定向到Qt界面"""
    def __init__(self, callback):
        self.callback = callback
        self.original_stdout = sys.stdout
        self.buffer = io.StringIO()
        
    def write(self, text):
        self.original_stdout.write(text)
        self.buffer.write(text)
        if self.callback and text.strip():
            self.callback(text.strip())
            
    def flush(self):
        self.original_stdout.flush()
        
    def get_output(self):
        return self.buffer.getvalue()
        
    def restore(self):
        sys.stdout = self.original_stdout


class SimulationData:
    """模拟数据结构"""
    def __init__(self):
        self.grid_info = {'nx': 20, 'ny': 20, 'nz': 5, 'Lx': 500.0, 'Ly': 500.0, 'Lz': 100.0}
        self.pressure_field = []
        self.temperature_field = []
        self.stress_field = []
        self.fractures = []
        self.wells = []
        
    def generate_from_cpp(self, sim_result, nx, ny, nz, lx, ly, lz):
        self.grid_info = {'nx': nx, 'ny': ny, 'nz': nz, 'Lx': lx, 'Ly': ly, 'Lz': lz}
        self.pressure_field = list(sim_result.pressure_field)
        self.temperature_field = list(sim_result.temperature_field) if sim_result.temperature_field else []
        self.stress_field = list(sim_result.stress_field) if sim_result.stress_field else []
        
        self.fractures = []
        vertices = list(sim_result.fracture_vertices)
        
        for i in range(0, len(vertices), 4):
            if i + 3 < len(vertices):
                frac = {
                    'id': i // 4,
                    'points': [
                        (vertices[i][0], vertices[i][1], vertices[i][2]),
                        (vertices[i+1][0], vertices[i+1][1], vertices[i+1][2]),
                        (vertices[i+2][0], vertices[i+2][1], vertices[i+2][2]),
                        (vertices[i+3][0], vertices[i+3][1], vertices[i+3][2])
                    ]
                }
                self.fractures.append(frac)
        
        print(f"Loaded {len(self.pressure_field)} pressure points, {len(self.fractures)} fractures")


class AlgorithmSelector(QWidget):
    """算法选择器（单行紧凑）"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(28)  # 缩窄高度
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(5, 2, 5, 2)
        self.layout.setSpacing(8)
        
        # 创建彩色图标（小尺寸）
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
        
        # 其他算法（灰色显示，无法点击）
        self.other_algos = ["Comp", "Thermal", "Foam", "Polymer"]
        
        self.layout.addWidget(self.icon_label)
        self.layout.addWidget(self.name_label)
        
        for algo in self.other_algos:
            lbl = QLabel(algo)
            lbl.setStyleSheet("color: #666666; font-size: 10px;")
            self.layout.addWidget(lbl)
        
        self.layout.addStretch()
        
    def create_colorful_icon(self):
        """创建彩色算法图标"""
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
    """VTK渲染窗口"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.layout.addWidget(self.vtk_widget)
        
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.iren = self.vtk_widget.GetRenderWindow().GetInteractor()
        
        self.renderer.SetBackground(0.0, 0.0, 0.0)  # 黑色背景
        self.iren.Initialize()
        
    def reset_camera(self):
        self.renderer.ResetCamera()
        self.iren.Render()


class ResInsightVisualizerV2(QMainWindow):
    """ResInsight风格可视化器V2"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EDFM Reservoir Simulator")
        self.setGeometry(50, 50, 1600, 1000)
        
        self.sim_data = SimulationData()
        self.current_tab = "Grid"
        self.show_fractures_enabled = False
        
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
        """初始化主界面"""
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
        
        # 左侧面板
        self.left_panel = QWidget()
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
        self.bottom_panel.setMaximumHeight(200)
        
        main_layout.addWidget(self.main_splitter, 1)  # stretch factor
        main_layout.addWidget(self.bottom_panel)
        
    def create_algorithm_bar_widget(self):
        """创建算法选择栏部件"""
        algo_bar = QWidget()
        algo_bar.setFixedHeight(32)  # 固定高度32像素
        algo_bar.setStyleSheet("background-color: #1e1e1e; border-bottom: 1px solid #3d3d3d;")
        algo_layout = QHBoxLayout(algo_bar)
        algo_layout.setContentsMargins(10, 2, 10, 2)
        algo_layout.setSpacing(10)
        
        # 添加彩色算法选择器
        algo_selector = AlgorithmSelector()
        algo_layout.addWidget(algo_selector)
        algo_layout.addStretch()
        
        return algo_bar
        
    def create_toolbar(self):
        """创建顶部工具栏"""
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
        run_btn = QPushButton("▶ Run Simulation")
        run_btn.setStyleSheet("""
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
        run_btn.clicked.connect(self.run_simulation)
        toolbar.addWidget(run_btn)
        
    def create_left_panel(self):
        """创建左侧面板 - 可切换的参数输入"""
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
        """创建Grid参数页面"""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(10, 10, 10, 10)
        
        group = QGroupBox("Grid Parameters")
        group.setStyleSheet(self.groupbox_style())
        grid = QGridLayout()
        
        # 与原始算法CSV一致 (20x10x2, 1000x500x50)
        self.spin_nx = QSpinBox()
        self.spin_nx.setRange(1, 200)
        self.spin_nx.setValue(20)
        self.spin_ny = QSpinBox()
        self.spin_ny.setRange(1, 100)
        self.spin_ny.setValue(10)
        self.spin_nz = QSpinBox()
        self.spin_nz.setRange(1, 50)
        self.spin_nz.setValue(2)
        
        # 域大小与原始算法CSV一致
        self.spin_lx = QDoubleSpinBox()
        self.spin_lx.setRange(1, 10000)
        self.spin_lx.setValue(1000)
        self.spin_ly = QDoubleSpinBox()
        self.spin_ly.setRange(1, 10000)
        self.spin_ly.setValue(500)
        self.spin_lz = QDoubleSpinBox()
        self.spin_lz.setRange(1, 1000)
        self.spin_lz.setValue(50)
        
        grid.addWidget(QLabel("Nx:"), 0, 0)
        grid.addWidget(self.spin_nx, 0, 1)
        grid.addWidget(QLabel("Ny:"), 1, 0)
        grid.addWidget(self.spin_ny, 1, 1)
        grid.addWidget(QLabel("Nz:"), 2, 0)
        grid.addWidget(self.spin_nz, 2, 1)
        grid.addWidget(QLabel("Lx (m):"), 3, 0)
        grid.addWidget(self.spin_lx, 3, 1)
        grid.addWidget(QLabel("Ly (m):"), 4, 0)
        grid.addWidget(self.spin_ly, 4, 1)
        grid.addWidget(QLabel("Lz (m):"), 5, 0)
        grid.addWidget(self.spin_lz, 5, 1)
        
        group.setLayout(grid)
        layout.addWidget(group)
        layout.addStretch()
        
        return page
        
    def create_wells_page(self):
        """创建Wells参数页面"""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(10, 10, 10, 10)
        
        group = QGroupBox("Well Parameters")
        group.setStyleSheet(self.groupbox_style())
        grid = QGridLayout()
        
        # 井位置在域中心 (250, 250)
        self.spin_well_x = QDoubleSpinBox()
        self.spin_well_x.setRange(0, 10000)
        self.spin_well_x.setValue(250)
        self.spin_well_y = QDoubleSpinBox()
        self.spin_well_y.setRange(0, 10000)
        self.spin_well_y.setValue(250)
        self.spin_well_z = QDoubleSpinBox()
        self.spin_well_z.setRange(0, 1000)
        self.spin_well_z.setValue(50)
        self.spin_well_pressure = QDoubleSpinBox()
        self.spin_well_pressure.setRange(0, 1000)
        self.spin_well_pressure.setValue(50)
        self.spin_well_radius = QDoubleSpinBox()
        self.spin_well_radius.setDecimals(3)
        self.spin_well_radius.setRange(0.01, 10)
        self.spin_well_radius.setValue(0.1)
        
        grid.addWidget(QLabel("Well X (m):"), 0, 0)
        grid.addWidget(self.spin_well_x, 0, 1)
        grid.addWidget(QLabel("Well Y (m):"), 1, 0)
        grid.addWidget(self.spin_well_y, 1, 1)
        grid.addWidget(QLabel("Well Z (m):"), 2, 0)
        grid.addWidget(self.spin_well_z, 2, 1)
        grid.addWidget(QLabel("Pressure (MPa):"), 3, 0)
        grid.addWidget(self.spin_well_pressure, 3, 1)
        grid.addWidget(QLabel("Radius (m):"), 4, 0)
        grid.addWidget(self.spin_well_radius, 4, 1)
        
        group.setLayout(grid)
        layout.addWidget(group)
        layout.addStretch()
        
        return page
        
    def create_fractures_page(self):
        """创建Fractures参数页面"""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(10, 10, 10, 10)
        
        group = QGroupBox("Fracture Parameters")
        group.setStyleSheet(self.groupbox_style())
        grid = QGridLayout()
        
        # 修正参数: 与原始算法一致 - 100个裂缝, 长度30-80m, 开度0.001m
        self.spin_num_fracs = QSpinBox()
        self.spin_num_fracs.setRange(0, 200)
        self.spin_num_fracs.setValue(100)
        self.spin_min_len = QDoubleSpinBox()
        self.spin_min_len.setRange(1, 500)
        self.spin_min_len.setValue(30)
        self.spin_max_len = QDoubleSpinBox()
        self.spin_max_len.setRange(1, 500)
        self.spin_max_len.setValue(80)
        self.spin_aperture = QDoubleSpinBox()
        self.spin_aperture.setDecimals(4)
        self.spin_aperture.setRange(0.0, 1.0)
        self.spin_aperture.setValue(0.001)
        
        grid.addWidget(QLabel("Num Fractures:"), 0, 0)
        grid.addWidget(self.spin_num_fracs, 0, 1)
        grid.addWidget(QLabel("Min Length (m):"), 1, 0)
        grid.addWidget(self.spin_min_len, 1, 1)
        grid.addWidget(QLabel("Max Length (m):"), 2, 0)
        grid.addWidget(self.spin_max_len, 2, 1)
        grid.addWidget(QLabel("Aperture (m):"), 3, 0)
        grid.addWidget(self.spin_aperture, 3, 1)
        
        group.setLayout(grid)
        layout.addWidget(group)
        layout.addStretch()
        
        return page
        
    def create_results_page(self):
        """创建Results页面"""
        page = QWidget()
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
        
    def switch_tab(self, tab_name):
        """切换标签页"""
        self.current_tab = tab_name
        for name, btn in self.tab_buttons.items():
            btn.setChecked(name == tab_name)
        
        tab_index = {"Grid": 0, "Wells": 1, "Fractures": 2, "Results": 3}
        self.param_stack.setCurrentIndex(tab_index.get(tab_name, 0))
        
    def create_center_panel(self):
        """创建中间VTK视图面板"""
        self.vtk_widget = VTKWidget()
        self.center_layout.addWidget(self.vtk_widget)
        
    def create_bottom_panel(self):
        """创建底部数据面板"""
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
        
    def append_sim_status(self, text):
        """添加模拟状态信息"""
        self.sim_status_text.append(text)
        self.sim_status_text.verticalScrollBar().setValue(self.sim_status_text.verticalScrollBar().maximum())
        QApplication.processEvents()
        
    def clear_sim_status(self):
        """清除模拟状态"""
        self.sim_status_text.clear()
        
    def create_status_bar(self):
        """创建状态栏"""
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("background-color: #2b2b2b; color: #cccccc;")
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Click 'Run Simulation' to start")
        
    def groupbox_style(self):
        """GroupBox样式"""
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
        
    def clear_cache(self):
        """清除VTK缓存"""
        self.cache['pressure_actor'] = None
        self.cache['fracture_actors'] = []
        self.cache['scalar_bar'] = None
        self.cache['grid_lines_actor'] = None
        self.cache['data_hash'] = None
        print("Cache cleared")
        
    def run_simulation(self):
        """运行模拟"""
        # 运行新模拟前清除缓存
        self.clear_cache()
        
        self.status_bar.showMessage("Running simulation...")
        self.clear_sim_status()
        self.append_sim_status("=" * 50)
        self.append_sim_status("  EDFM Black Oil Simulation Starting...")
        self.append_sim_status("=" * 50)
        
        output_capture = OutputCapture(self.append_sim_status)
        sys.stdout = output_capture
        
        try:
            nx = self.spin_nx.value()
            ny = self.spin_ny.value()
            nz = self.spin_nz.value()
            lx = self.spin_lx.value()
            ly = self.spin_ly.value()
            lz = self.spin_lz.value()
            
            num_fracs = self.spin_num_fracs.value()
            min_len = self.spin_min_len.value()
            max_len = self.spin_max_len.value()
            aperture = self.spin_aperture.value()
            print(f"DEBUG Python: aperture={aperture}, type={type(aperture)}")
            
            well_x = self.spin_well_x.value()
            well_y = self.spin_well_y.value()
            well_z = self.spin_well_z.value()
            well_pressure = self.spin_well_pressure.value()
            
            self.append_sim_status(f"\nGrid: {nx}x{ny}x{nz}, Domain: {lx}x{ly}x{lz} m")
            self.append_sim_status(f"Fractures: {num_fracs}, Length: {min_len}-{max_len} m, Aperture: {aperture} m")
            self.append_sim_status(f"Well: ({well_x}, {well_y}, {well_z}), Pressure: {well_pressure} bar")
            self.append_sim_status("")
            
            if HAS_CPP_MODULE:
                sim = edfm_core.EDFMSimulator()
                sim.setGridParameters(nx, ny, nz, lx, ly, lz)
                # 修正参数: 使用UI中的裂缝开度值
                import math
                sim.setFractureParameters(num_fracs, min_len, max_len, math.pi/3.0, 0.0, math.pi, aperture, 10000.0)
                sim.setSimulationParameters(100.0, 1.0, 0.2, 0.001, 0.001, 0.0001)
                # 修正参数: 井半径0.05 (不是0.1)
                sim.setWellParameters(well_x, well_y, well_z, 0.05, well_pressure)
                
                result = sim.runSimulation()
                self.sim_data.generate_from_cpp(result, nx, ny, nz, lx, ly, lz)
            else:
                self.generate_mock_data(nx, ny, nz, lx, ly, lz, num_fracs, min_len, max_len,
                                       well_x, well_y, well_z, well_pressure)
            
            self.append_sim_status("")
            self.append_sim_status("=" * 50)
            self.append_sim_status("  Simulation Completed Successfully!")
            self.append_sim_status("=" * 50)
            
            self.render_mode3_smooth_pressure()
            self.update_statistics()
            
            self.status_bar.showMessage("Simulation completed successfully")
            
        except Exception as e:
            self.append_sim_status(f"\nERROR: {str(e)}")
            self.status_bar.showMessage(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            output_capture.restore()
            
    def generate_mock_data(self, nx, ny, nz, lx, ly, lz, num_fracs, min_len, max_len,
                          well_x, well_y, well_z, well_pressure):
        """生成模拟数据"""
        self.sim_data.grid_info = {'nx': nx, 'ny': ny, 'nz': nz, 'Lx': lx, 'Ly': ly, 'Lz': lz}
        
        dx, dy, dz = lx / nx, ly / ny, lz / nz
        
        self.sim_data.pressure_field = []
        self.sim_data.temperature_field = []
        self.sim_data.stress_field = []
        
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    x = (i + 0.5) * dx
                    y = (j + 0.5) * dy
                    z = (k + 0.5) * dz
                    
                    dist = np.sqrt((x - well_x)**2 + (y - well_y)**2 + (z - well_z)**2)
                    pressure = well_pressure * np.exp(-dist / 100.0)
                    temperature = 300.0 + pressure * 0.5
                    stress = pressure * 1.5
                    
                    self.sim_data.pressure_field.append((x, y, z, pressure))
                    self.sim_data.temperature_field.append((x, y, z, temperature))
                    self.sim_data.stress_field.append((x, y, z, stress))
        
        # 生成裂缝 - 限制在网格范围内
        self.sim_data.fractures = []
        for i in range(num_fracs):
            cx, cy, cz = lx * 0.5, ly * 0.5, lz * 0.5
            length = min_len + (max_len - min_len) * i / max(1, num_fracs - 1)
            angle = 6.28 * i / max(1, num_fracs)
            dx_f = length * np.cos(angle) * 0.5
            dy_f = length * np.sin(angle) * 0.5
            
            # 计算裂缝四个角点
            p1 = (cx - dx_f, cy - dy_f, cz)
            p2 = (cx + dx_f, cy - dy_f, cz)
            p3 = (cx + dx_f, cy + dy_f, cz)
            p4 = (cx - dx_f, cy + dy_f, cz)
            
            # 限制在网格边界内
            def clamp_point(p, lx, ly, lz):
                return (max(0, min(lx, p[0])), max(0, min(ly, p[1])), max(0, min(lz, p[2])))
            
            p1 = clamp_point(p1, lx, ly, lz)
            p2 = clamp_point(p2, lx, ly, lz)
            p3 = clamp_point(p3, lx, ly, lz)
            p4 = clamp_point(p4, lx, ly, lz)
            
            frac = {
                'id': i,
                'points': [p1, p2, p3, p4]
            }
            self.sim_data.fractures.append(frac)
            
    def render_mode3_smooth_pressure(self):
        """渲染Mode3: 平滑压力实体块（带缓存）"""
        renderer = self.vtk_widget.renderer
        
        # 获取压力数据
        field_data = self.sim_data.pressure_field
        if not field_data:
            return
        
        # 计算数据哈希，检测是否需要重新生成
        data_hash = hash(str(len(field_data)) + str(field_data[0]) + str(field_data[-1]))
        
        # 如果缓存有效且数据未变化，直接使用缓存
        if self.cache['data_hash'] == data_hash and self.cache['pressure_actor'] is not None:
            print("Using cached pressure visualization")
            renderer.AddActor(self.cache['pressure_actor'])
            if self.cache['scalar_bar']:
                renderer.AddViewProp(self.cache['scalar_bar'])
            self.setup_camera(renderer)
            self.vtk_widget.iren.Render()
            return
        
        # 清除旧视图
        renderer.RemoveAllViewProps()
        
        values = [p[3] for p in field_data]
        min_p, max_p = min(values), max(values)
        
        print(f"Pressure range: min={min_p}, max={max_p}")
        print(f"Number of leaf cells: {len(field_data)}")
        
        # 获取网格边界
        lx = self.sim_data.grid_info['Lx']
        ly = self.sim_data.grid_info['Ly']
        lz = self.sim_data.grid_info['Lz']
        
        # 使用完整网格域作为可视化边界（让压力场覆盖整个网格）
        min_x, max_x = 0.0, lx
        min_y, max_y = 0.0, ly
        min_z, max_z = 0.0, lz
        
        # 从叶网格数据中提取实际数据范围（用于信息输出）
        xs = [p[0] for p in field_data]
        ys = [p[1] for p in field_data]
        zs = [p[2] for p in field_data]
        data_min_x, data_max_x = min(xs), max(xs)
        data_min_y, data_max_y = min(ys), max(ys)
        data_min_z, data_max_z = min(zs), max(zs)
        
        # 输出边界对比
        print("=" * 60)
        print("BOUNDARY COMPARISON:")
        print(f"Visualization:  X[{min_x:.2f}, {max_x:.2f}], Y[{min_y:.2f}, {max_y:.2f}], Z[{min_z:.2f}, {max_z:.2f}]")
        print(f"Data Range:     X[{data_min_x:.2f}, {data_max_x:.2f}], Y[{data_min_y:.2f}, {data_max_y:.2f}], Z[{data_min_z:.2f}, {data_max_z:.2f}]")
        
        # 同时输出裂缝边界
        if self.sim_data.fractures:
            frac_xs = []
            frac_ys = []
            frac_zs = []
            for fracture in self.sim_data.fractures:
                for p in fracture['points']:
                    frac_xs.append(p[0])
                    frac_ys.append(p[1])
                    frac_zs.append(p[2])
            
            print(f"Fractures:      X[{min(frac_xs):.2f}, {max(frac_xs):.2f}], Y[{min(frac_ys):.2f}, {max(frac_ys):.2f}], Z[{min(frac_zs):.2f}, {max(frac_zs):.2f}]")
            
            # 检查是否超出网格边界
            x_exceed = min(frac_xs) < 0 or max(frac_xs) > lx
            y_exceed = min(frac_ys) < 0 or max(frac_ys) > ly
            z_exceed = min(frac_zs) < 0 or max(frac_zs) > lz
            
            if x_exceed or y_exceed or z_exceed:
                print("WARNING: Fractures EXCEED grid boundaries!")
                if x_exceed: print(f"  X: fractures [{min(frac_xs):.2f}, {max(frac_xs):.2f}] vs grid [0, {lx:.2f}]")
                if y_exceed: print(f"  Y: fractures [{min(frac_ys):.2f}, {max(frac_ys):.2f}] vs grid [0, {ly:.2f}]")
                if z_exceed: print(f"  Z: fractures [{min(frac_zs):.2f}, {max(frac_zs):.2f}] vs grid [0, {lz:.2f}]")
            else:
                print("OK: All fractures within grid boundaries")
        print("=" * 60)
        
        # 降低分辨率以提高性能
        nx, ny, nz = 50, 25, 10
        
        print(f"Creating structured grid: {nx}x{ny}x{nz}")
        
        # 创建结构化网格
        grid = vtk.vtkStructuredGrid()
        grid.SetDimensions(nx, ny, nz)
        
        points = vtk.vtkPoints()
        pressure_arr = vtk.vtkDoubleArray()
        pressure_arr.SetName("Pressure")
        
        # 预计算：将叶网格数据按x,y,z排序以便快速查找
        sorted_field = sorted(field_data, key=lambda p: (p[0], p[1], p[2]))
        
        # 创建规则网格点并进行简单插值（最近邻）
        for k in range(nz):
            z = min_z + (max_z - min_z) * k / (nz - 1) if nz > 1 else min_z
            for j in range(ny):
                y = min_y + (max_y - min_y) * j / (ny - 1) if ny > 1 else min_y
                for i in range(nx):
                    x = min_x + (max_x - min_x) * i / (nx - 1) if nx > 1 else min_x
                    points.InsertNextPoint(x, y, z)
                    
                    # 简单最近邻插值（性能更好）
                    min_dist = float('inf')
                    nearest_p = min_p
                    for fx, fy, fz, fp in sorted_field:
                        dist = abs(x-fx) + abs(y-fy) + abs(z-fz)  # 曼哈顿距离更快
                        if dist < min_dist:
                            min_dist = dist
                            nearest_p = fp
                            if dist < 1e-6:  # 足够近就停止
                                break
                    
                    pressure_arr.InsertNextValue(nearest_p)
        
        grid.SetPoints(points)
        grid.GetPointData().SetScalars(pressure_arr)
        
        print("Grid created, extracting surface...")
        
        # 使用GeometryFilter提取表面
        geom_filter = vtk.vtkGeometryFilter()
        geom_filter.SetInputData(grid)
        geom_filter.Update()
        
        # 创建颜色映射表（彩虹色：蓝->绿->黄->红）
        lut = vtk.vtkLookupTable()
        lut.SetTableRange(min_p, max_p)
        lut.SetHueRange(0.667, 0.0)  # 蓝到红
        lut.SetSaturationRange(1.0, 1.0)
        lut.SetValueRange(1.0, 1.0)
        lut.SetNumberOfTableValues(256)
        lut.Build()
        
        # 创建颜色条
        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetLookupTable(lut)
        scalar_bar.SetTitle("Pressure (MPa)")
        scalar_bar.SetNumberOfLabels(6)
        scalar_bar.SetWidth(0.08)
        scalar_bar.SetHeight(0.4)
        scalar_bar.GetLabelTextProperty().SetColor(1, 1, 1)
        scalar_bar.GetTitleTextProperty().SetColor(1, 1, 1)
        scalar_bar.GetLabelTextProperty().SetFontSize(12)
        scalar_bar.SetPosition(0.88, 0.25)
        renderer.AddViewProp(scalar_bar)
        
        # 创建mapper和actor（参考C++代码）
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputConnection(geom_filter.GetOutputPort())
        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(min_p, max_p)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(1.0)
        actor.GetProperty().SetEdgeVisibility(0)
        
        renderer.AddActor(actor)
        
        # 保存到缓存
        self.cache['data_hash'] = data_hash
        self.cache['pressure_actor'] = actor
        self.cache['scalar_bar'] = scalar_bar
        
        print("Cached pressure visualization")
        
        # 设置相机
        self.setup_camera(renderer)
        
        self.vtk_widget.iren.Render()
        
    def render_fractures(self, renderer):
        """渲染裂缝（带缓存）"""
        # 检查缓存
        if self.cache['fracture_actors']:
            print("Using cached fracture visualization")
            for actor in self.cache['fracture_actors']:
                renderer.AddActor(actor)
            return
        
        print("Generating fracture visualization...")
        
        # 输出裂缝边界和网格边界对比
        if self.sim_data.fractures:
            frac_xs = []
            frac_ys = []
            frac_zs = []
            for fracture in self.sim_data.fractures:
                for p in fracture['points']:
                    frac_xs.append(p[0])
                    frac_ys.append(p[1])
                    frac_zs.append(p[2])
            
            lx = self.sim_data.grid_info['Lx']
            ly = self.sim_data.grid_info['Ly']
            lz = self.sim_data.grid_info['Lz']
            
            print("=" * 60)
            print("BOUNDARY CHECK:")
            print(f"Grid Domain: X[0, {lx}], Y[0, {ly}], Z[0, {lz}]")
            print(f"Fracture Range: X[{min(frac_xs):.2f}, {max(frac_xs):.2f}], Y[{min(frac_ys):.2f}, {max(frac_ys):.2f}], Z[{min(frac_zs):.2f}, {max(frac_zs):.2f}]")
            
            x_out = min(frac_xs) < 0 or max(frac_xs) > lx
            y_out = min(frac_ys) < 0 or max(frac_ys) > ly
            z_out = min(frac_zs) < 0 or max(frac_zs) > lz
            
            if x_out or y_out or z_out:
                print("WARNING: Fractures EXCEED grid boundaries!")
                if x_out: print(f"  X out of range: min={min(frac_xs):.2f}, max={max(frac_xs):.2f}, should be [0, {lx}]")
                if y_out: print(f"  Y out of range: min={min(frac_ys):.2f}, max={max(frac_ys):.2f}, should be [0, {ly}]")
                if z_out: print(f"  Z out of range: min={min(frac_zs):.2f}, max={max(frac_zs):.2f}, should be [0, {lz}]")
            else:
                print("OK: All fractures within grid boundaries")
            print("=" * 60)
        
        # 创建裂缝面（橙色半透明）
        points = vtk.vtkPoints()
        polys = vtk.vtkCellArray()
        
        for fracture in self.sim_data.fractures:
            frac_points = fracture['points']
            n_points = len(frac_points)
            
            polygon = vtk.vtkPolygon()
            polygon.GetPointIds().SetNumberOfIds(n_points)
            
            base_idx = points.GetNumberOfPoints()
            for i, (x, y, z) in enumerate(frac_points):
                points.InsertNextPoint(x, y, z)
                polygon.GetPointIds().SetId(i, base_idx + i)
            
            polys.InsertNextCell(polygon)
        
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetPolys(polys)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 0.5, 0.0)  # 橙色
        actor.GetProperty().SetOpacity(0.6)
        actor.GetProperty().SetEdgeVisibility(1)
        actor.GetProperty().SetEdgeColor(0.8, 0.4, 0.0)
        actor.GetProperty().SetLineWidth(1.0)
        
        renderer.AddActor(actor)
        self.cache['fracture_actors'].append(actor)
        
        # 添加裂缝边界线（黑色）
        line_points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        
        for fracture in self.sim_data.fractures:
            frac_points = fracture['points']
            n_points = len(frac_points)
            
            for i in range(n_points):
                x0, y0, z0 = frac_points[i]
                x1, y1, z1 = frac_points[(i + 1) % n_points]
                
                p1 = line_points.InsertNextPoint(x0, y0, z0)
                p2 = line_points.InsertNextPoint(x1, y1, z1)
                
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, p1)
                line.GetPointIds().SetId(1, p2)
                lines.InsertNextCell(line)
        
        line_data = vtk.vtkPolyData()
        line_data.SetPoints(line_points)
        line_data.SetLines(lines)
        
        line_mapper = vtk.vtkPolyDataMapper()
        line_mapper.SetInputData(line_data)
        
        line_actor = vtk.vtkActor()
        line_actor.SetMapper(line_mapper)
        line_actor.GetProperty().SetColor(0.0, 0.0, 0.0)
        line_actor.GetProperty().SetLineWidth(1.5)
        
        renderer.AddActor(line_actor)
        self.cache['fracture_actors'].append(line_actor)
        
        print("Cached fracture visualization")
        
    def setup_camera(self, renderer):
        """设置相机"""
        camera = renderer.GetActiveCamera()
        
        lx = self.sim_data.grid_info['Lx']
        ly = self.sim_data.grid_info['Ly']
        lz = self.sim_data.grid_info['Lz']
        
        cx, cy, cz = lx / 2.0, ly / 2.0, lz / 2.0
        max_dim = max(lx, ly, lz)
        
        dist = max_dim * 2.5
        
        camera.SetPosition(cx + dist, cy + dist, cz + dist)
        camera.SetFocalPoint(cx, cy, cz)
        camera.SetViewUp(0, 0, 1)
        
        renderer.ResetCamera()
        camera.Zoom(1.1)
        
    def change_view_mode(self, mode):
        """切换视图模式"""
        if mode == "Pressure Field":
            self.render_mode3_smooth_pressure()
        elif mode == "Fracture Mesh":
            self.render_fracture_only()
            
    def change_field_display(self, field):
        """切换显示的场"""
        if self.view_mode_combo.currentText() == "Pressure Field":
            self.render_mode3_smooth_pressure()
            
    def toggle_grid_lines(self, state):
        """切换网格线显示"""
        renderer = self.vtk_widget.renderer
        show_grid = (state == Qt.Checked)
        
        if show_grid:
            # 显示网格线
            if self.cache['grid_lines_actor'] is None:
                self.create_grid_lines()
            if self.cache['grid_lines_actor']:
                renderer.AddActor(self.cache['grid_lines_actor'])
        else:
            # 隐藏网格线
            if self.cache['grid_lines_actor']:
                renderer.RemoveActor(self.cache['grid_lines_actor'])
        
        self.vtk_widget.iren.Render()
    
    def create_grid_lines(self):
        """创建网格线"""
        lx = self.sim_data.grid_info['Lx']
        ly = self.sim_data.grid_info['Ly']
        lz = self.sim_data.grid_info['Lz']
        nx = self.sim_data.grid_info['nx']
        ny = self.sim_data.grid_info['ny']
        nz = self.sim_data.grid_info['nz']
        
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        
        # X方向线 (沿Y轴)
        for i in range(nx + 1):
            x = i * lx / nx
            for k in range(nz + 1):
                z = k * lz / nz
                p1 = points.InsertNextPoint(x, 0, z)
                p2 = points.InsertNextPoint(x, ly, z)
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, p1)
                line.GetPointIds().SetId(1, p2)
                lines.InsertNextCell(line)
        
        # Y方向线 (沿X轴)
        for j in range(ny + 1):
            y = j * ly / ny
            for k in range(nz + 1):
                z = k * lz / nz
                p1 = points.InsertNextPoint(0, y, z)
                p2 = points.InsertNextPoint(lx, y, z)
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, p1)
                line.GetPointIds().SetId(1, p2)
                lines.InsertNextCell(line)
        
        # Z方向线 (垂直)
        for i in range(nx + 1):
            x = i * lx / nx
            for j in range(ny + 1):
                y = j * ly / ny
                p1 = points.InsertNextPoint(x, y, 0)
                p2 = points.InsertNextPoint(x, y, lz)
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, p1)
                line.GetPointIds().SetId(1, p2)
                lines.InsertNextCell(line)
        
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetLines(lines)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.8, 0.8, 0.8)  # 浅灰色，更明显
        actor.GetProperty().SetLineWidth(1.0)
        actor.GetProperty().SetOpacity(0.8)
        
        self.cache['grid_lines_actor'] = actor
            
    def toggle_fractures_visibility(self, state):
        """切换裂缝显示 - 点击时压力图变透明"""
        self.show_fractures_enabled = (state == Qt.Checked)
        
        renderer = self.vtk_widget.renderer
        
        if self.show_fractures_enabled:
            # 显示裂缝 - 压力图变半透明
            if self.cache['pressure_actor']:
                self.cache['pressure_actor'].GetProperty().SetOpacity(0.3)
            # 添加裂缝到当前视图
            self.render_fractures(renderer)
        else:
            # 隐藏裂缝 - 压力图恢复不透明
            if self.cache['pressure_actor']:
                self.cache['pressure_actor'].GetProperty().SetOpacity(1.0)
            # 移除裂缝actors
            for actor in self.cache['fracture_actors']:
                renderer.RemoveActor(actor)
        
        self.vtk_widget.iren.Render()
            
    def render_fracture_only(self):
        """仅渲染裂缝"""
        renderer = self.vtk_widget.renderer
        renderer.RemoveAllViewProps()
        
        if self.sim_data.fractures:
            self.render_fractures(renderer)
        
        self.setup_camera(renderer)
        self.vtk_widget.iren.Render()
            
    def update_statistics(self):
        """更新统计信息"""
        if not self.sim_data.pressure_field:
            return
        
        nx = self.sim_data.grid_info['nx']
        ny = self.sim_data.grid_info['ny']
        nz = self.sim_data.grid_info['nz']
        
        pressures = [p[3] for p in self.sim_data.pressure_field]
        
        stats_text = f"""Grid: {nx} x {ny} x {nz} = {nx*ny*nz} cells
Fractures: {len(self.sim_data.fractures)}

Pressure Range:
  Min: {min(pressures):.2f} MPa
  Max: {max(pressures):.2f} MPa
  Mean: {np.mean(pressures):.2f} MPa
  P90: {np.percentile(pressures, 90):.2f} MPa
"""
        self.stats_text.setText(stats_text)
        
        # 更新属性表
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


def main():
    app = QApplication(sys.argv)
    
    app.setStyle('Fusion')
    app.setStyleSheet("""
        QMainWindow {
            background-color: #1e1e1e;
        }
        QWidget {
            background-color: #1e1e1e;
        }
    """)
    
    window = ResInsightVisualizerV2()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
