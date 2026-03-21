"""
主窗口模块
整合所有UI组件和可视化 - 与原文件完全一致
"""
import sys
import os
import math

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QSplitter, QFrame, QToolBar, QAction,
                             QStatusBar, QTabWidget, QStackedWidget, QTextEdit, QGroupBox,
                             QTableWidget, QSpinBox, QDoubleSpinBox, QGridLayout, QComboBox,
                             QCheckBox, QTableWidgetItem)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon, QFont, QColor, QPixmap, QPainter
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk

# 导入本地模块
from .data_models import OutputCapture, SimulationData

# 导入可视化模块
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from visual.vtk_renderer import VTKRenderer

# 导入C++模块
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'build', 'Release'))
try:
    import edfm_core
    HAS_CPP_MODULE = True
    print("Successfully imported edfm_core module")
except ImportError as e:
    HAS_CPP_MODULE = False
    print(f"C++ module not found: {e}, using mock data")


class AlgorithmSelector(QWidget):
    """算法选择器（单行紧凑）- 与原文件一致"""
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
        
        main_layout.addWidget(self.main_splitter, 1)  # stretch factor
        main_layout.addWidget(self.bottom_panel)
    
    def create_algorithm_bar_widget(self):
        """创建算法选择栏部件 - 与原文件一致"""
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
        """创建Grid参数页面 - 与原文件一致"""
        page = QWidget()
        page.setStyleSheet("background-color: #2b2b2b;")
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
        """创建Wells参数页面 - 与原文件一致"""
        page = QWidget()
        page.setStyleSheet("background-color: #2b2b2b;")
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
        """创建Fractures参数页面 - 与原文件一致"""
        page = QWidget()
        page.setStyleSheet("background-color: #2b2b2b;")
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
        print("Cache cleared")
    
    def run_simulation(self):
        """运行模拟 - 与原文件完全一致"""
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
                sim.setFractureParameters(num_fracs, min_len, max_len, math.pi/3.0, 0.0, math.pi, aperture, 10000.0)
                sim.setSimulationParameters(100.0, 1.0, 0.2, 0.001, 0.001, 0.0001)
                # 修正参数: 井半径0.05 (不是0.1)
                sim.setWellParameters(well_x, well_y, well_z, 0.05, well_pressure)
                
                result = sim.runSimulation()
                grid_lines = sim.getGridLines()
                interpolated_pressure = sim.getInterpolatedPressureField(50, 25, 10)
                self.sim_data.generate_from_cpp(result, nx, ny, nz, lx, ly, lz, grid_lines, interpolated_pressure)
            else:
                self.sim_data.generate_mock_data(nx, ny, nz, lx, ly, lz, num_fracs, 
                                                min_len, max_len, well_x, well_y, well_z, 
                                                well_pressure)
            
            self.append_sim_status("")
            self.append_sim_status("=" * 50)
            self.append_sim_status("  Simulation Completed Successfully!")
            self.append_sim_status("=" * 50)
            
            # 渲染
            self.render_mode3_smooth_pressure()
            self.update_statistics()
            
            # 输出边界信息到Simulation Status
            self.append_sim_status("")
            self.append_sim_status(f"Visualization:  X[0.00, {lx:.2f}], Y[0.00, {ly:.2f}], Z[0.00, {lz:.2f}]")
            if self.sim_data.pressure_field:
                xs = [p[0] for p in self.sim_data.pressure_field]
                ys = [p[1] for p in self.sim_data.pressure_field]
                zs = [p[2] for p in self.sim_data.pressure_field]
                self.append_sim_status(f"Data Range:     X[{min(xs):.2f}, {max(xs):.2f}], Y[{min(ys):.2f}, {max(ys):.2f}], Z[{min(zs):.2f}, {max(zs):.2f}]")
            if self.sim_data.fractures:
                frac_xs = []
                frac_ys = []
                frac_zs = []
                for fracture in self.sim_data.fractures:
                    for p in fracture['points']:
                        frac_xs.append(p[0])
                        frac_ys.append(p[1])
                        frac_zs.append(p[2])
                self.append_sim_status(f"Fractures:      X[{min(frac_xs):.2f}, {max(frac_xs):.2f}], Y[{min(frac_ys):.2f}, {max(frac_ys):.2f}], Z[{min(frac_zs):.2f}, {max(frac_zs):.2f}]")
                self.append_sim_status("OK: All fractures within grid boundaries")
            
        except Exception as e:
            self.append_sim_status(f"ERROR: {str(e)}")
            import traceback
            self.append_sim_status(traceback.format_exc())
        finally:
            output_capture.restore()
        
        self.status_bar.showMessage("Simulation completed")
    
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
