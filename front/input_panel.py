"""
输入面板模块
负责所有参数输入UI组件
"""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QGroupBox, QComboBox,
                             QCheckBox, QSpinBox, QDoubleSpinBox, QGridLayout,
                             QTabWidget, QStackedWidget, QFrame)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor, QPixmap, QPainter


def groupbox_style():
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


class AlgorithmSelector(QWidget):
    """算法选择器"""
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
        self.name_label.setStyleSheet("font-weight: bold; color: #333;")
        
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["Black Oil", "Compositional", "Thermal"])
        self.algo_combo.setEnabled(False)
        
        self.layout.addWidget(self.icon_label)
        self.layout.addWidget(self.name_label)
        self.layout.addWidget(self.algo_combo)
        self.layout.addStretch()
    
    def create_colorful_icon(self):
        """创建彩色图标"""
        pixmap = QPixmap(20, 20)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        gradient_colors = [QColor(255, 0, 0), QColor(255, 165, 0), 
                          QColor(255, 255, 0), QColor(0, 128, 0)]
        
        for i, color in enumerate(gradient_colors):
            painter.setBrush(color)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(2 + i*4, 2, 16, 16)
        
        painter.end()
        return pixmap


class GridInputPanel(QWidget):
    """网格参数输入面板 - 与原文件参数一致"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        group = QGroupBox("Grid Parameters")
        group.setStyleSheet(groupbox_style())
        grid = QGridLayout()
        
        # 与C++代码默认值一致 (150x15x2, 3000x300x40)
        self.spin_nx = QSpinBox()
        self.spin_nx.setRange(1, 200)
        self.spin_nx.setValue(150)
        
        self.spin_ny = QSpinBox()
        self.spin_ny.setRange(1, 100)
        self.spin_ny.setValue(15)
        
        self.spin_nz = QSpinBox()
        self.spin_nz.setRange(1, 50)
        self.spin_nz.setValue(2)
        
        # 域大小与C++代码默认值一致
        self.spin_lx = QDoubleSpinBox()
        self.spin_lx.setRange(1, 10000)
        self.spin_lx.setValue(3000)
        
        self.spin_ly = QDoubleSpinBox()
        self.spin_ly.setRange(1, 10000)
        self.spin_ly.setValue(300)
        
        self.spin_lz = QDoubleSpinBox()
        self.spin_lz.setRange(1, 1000)
        self.spin_lz.setValue(40)
        
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
    
    def get_values(self):
        """获取网格参数"""
        return {
            'nx': self.spin_nx.value(),
            'ny': self.spin_ny.value(),
            'nz': self.spin_nz.value(),
            'lx': self.spin_lx.value(),
            'ly': self.spin_ly.value(),
            'lz': self.spin_lz.value()
        }


class WellsInputPanel(QWidget):
    """井参数输入面板 - 与原文件参数一致"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        group = QGroupBox("Well Parameters")
        group.setStyleSheet(groupbox_style())
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
    
    def get_values(self):
        """获取井参数"""
        return {
            'x': self.spin_well_x.value(),
            'y': self.spin_well_y.value(),
            'z': self.spin_well_z.value(),
            'pressure': self.spin_well_pressure.value(),
            'radius': self.spin_well_radius.value()
        }


class FracturesInputPanel(QWidget):
    """裂缝参数输入面板 - 与原文件参数一致"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        group = QGroupBox("Fracture Parameters")
        group.setStyleSheet(groupbox_style())
        grid = QGridLayout()
        
        # 修正参数: 与C++代码默认值一致 - 10个裂缝, 长度30-80m, 开度0.01m
        self.spin_num_fracs = QSpinBox()
        self.spin_num_fracs.setRange(0, 200)
        self.spin_num_fracs.setValue(10)
        
        self.spin_min_len = QDoubleSpinBox()
        self.spin_min_len.setRange(1, 500)
        self.spin_min_len.setValue(30)
        
        self.spin_max_len = QDoubleSpinBox()
        self.spin_max_len.setRange(1, 500)
        self.spin_max_len.setValue(80)
        
        self.spin_aperture = QDoubleSpinBox()
        self.spin_aperture.setDecimals(4)
        self.spin_aperture.setRange(0.0, 1.0)
        self.spin_aperture.setValue(0.01)
        
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
    
    def get_values(self):
        """获取裂缝参数"""
        return {
            'num_fracs': self.spin_num_fracs.value(),
            'min_len': self.spin_min_len.value(),
            'max_len': self.spin_max_len.value(),
            'aperture': self.spin_aperture.value()
        }


def create_spinbox(minimum, maximum, value):
    spin = QSpinBox()
    spin.setRange(minimum, maximum)
    spin.setValue(value)
    return spin


def create_double_spinbox(minimum, maximum, value, decimals=2, step=None):
    spin = QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setDecimals(decimals)
    spin.setValue(value)
    if step is not None:
        spin.setSingleStep(step)
    return spin


class MatrixPropertiesPanel(QWidget):
    """基质属性参数面板 - 可复用"""
    def __init__(self, parent=None, prefix=""):
        super().__init__(parent)
        self.prefix = prefix
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        group = QGroupBox("Matrix Properties")
        group.setStyleSheet(groupbox_style())
        grid = QGridLayout()
        
        self.spin_porosity = create_double_spinbox(0.0, 1.0, 0.04, decimals=4)
        self.spin_perm_x = create_double_spinbox(0.0, 1000000, 0.005, decimals=6)
        self.spin_perm_y = create_double_spinbox(0.0, 1000000, 0.005, decimals=6)
        self.spin_perm_z = create_double_spinbox(0.0, 1000000, 0.005, decimals=6)
        
        grid.addWidget(QLabel("Porosity:"), 0, 0)
        grid.addWidget(self.spin_porosity, 0, 1)
        grid.addWidget(QLabel("Kx (Darcy):"), 1, 0)
        grid.addWidget(self.spin_perm_x, 1, 1)
        grid.addWidget(QLabel("Ky (Darcy):"), 2, 0)
        grid.addWidget(self.spin_perm_y, 2, 1)
        grid.addWidget(QLabel("Kz (Darcy):"), 3, 0)
        grid.addWidget(self.spin_perm_z, 3, 1)
        
        group.setLayout(grid)
        layout.addWidget(group)
    
    def get_values(self):
        return {
            'porosity': self.spin_porosity.value(),
            'perm_x': self.spin_perm_x.value(),
            'perm_y': self.spin_perm_y.value(),
            'perm_z': self.spin_perm_z.value()
        }


class FluidPropertiesPanel(QWidget):
    """流体属性参数面板 - 可复用"""
    def __init__(self, parent=None, prefix=""):
        super().__init__(parent)
        self.prefix = prefix
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        group = QGroupBox("Fluid Properties")
        group.setStyleSheet(groupbox_style())
        grid = QGridLayout()
        
        self.spin_mu_w = create_double_spinbox(0.0, 1000.0, 1.0, decimals=4)
        self.spin_mu_o = create_double_spinbox(0.0, 1000.0, 5.0, decimals=4)
        self.spin_mu_g = create_double_spinbox(0.0, 1000.0, 0.2, decimals=4)
        self.spin_cw = create_double_spinbox(0.0, 1.0, 1e-8, decimals=8, step=1e-8)
        self.spin_co = create_double_spinbox(0.0, 1.0, 1e-5, decimals=8, step=1e-6)
        self.spin_cg = create_double_spinbox(0.0, 1.0, 1e-3, decimals=6, step=1e-4)
        self.spin_p_ref = create_double_spinbox(0.0, 1000000, 100.0, decimals=2)
        self.spin_swi = create_double_spinbox(0.0, 1.0, 0.2, decimals=4)
        self.spin_sor = create_double_spinbox(0.0, 1.0, 0.2, decimals=4)
        self.spin_sgc = create_double_spinbox(0.0, 1.0, 0.05, decimals=4)
        
        grid.addWidget(QLabel("mu_w (cP):"), 0, 0)
        grid.addWidget(self.spin_mu_w, 0, 1)
        grid.addWidget(QLabel("mu_o (cP):"), 1, 0)
        grid.addWidget(self.spin_mu_o, 1, 1)
        grid.addWidget(QLabel("mu_g (cP):"), 2, 0)
        grid.addWidget(self.spin_mu_g, 2, 1)
        grid.addWidget(QLabel("cw (1/bar):"), 3, 0)
        grid.addWidget(self.spin_cw, 3, 1)
        grid.addWidget(QLabel("co (1/bar):"), 4, 0)
        grid.addWidget(self.spin_co, 4, 1)
        grid.addWidget(QLabel("cg (1/bar):"), 5, 0)
        grid.addWidget(self.spin_cg, 5, 1)
        grid.addWidget(QLabel("P_ref (bar):"), 6, 0)
        grid.addWidget(self.spin_p_ref, 6, 1)
        grid.addWidget(QLabel("Swi:"), 7, 0)
        grid.addWidget(self.spin_swi, 7, 1)
        grid.addWidget(QLabel("Sor:"), 8, 0)
        grid.addWidget(self.spin_sor, 8, 1)
        grid.addWidget(QLabel("Sgc:"), 9, 0)
        grid.addWidget(self.spin_sgc, 9, 1)
        
        group.setLayout(grid)
        layout.addWidget(group)
    
    def get_values(self):
        return {
            'mu_w': self.spin_mu_w.value(),
            'mu_o': self.spin_mu_o.value(),
            'mu_g': self.spin_mu_g.value(),
            'cw': self.spin_cw.value(),
            'co': self.spin_co.value(),
            'cg': self.spin_cg.value(),
            'p_ref': self.spin_p_ref.value(),
            'swi': self.spin_swi.value(),
            'sor': self.spin_sor.value(),
            'sgc': self.spin_sgc.value()
        }


class InitialStatePanel(QWidget):
    """初始状态参数面板 - 可复用"""
    def __init__(self, parent=None, prefix=""):
        super().__init__(parent)
        self.prefix = prefix
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        group = QGroupBox("Initial State")
        group.setStyleSheet(groupbox_style())
        grid = QGridLayout()
        
        self.spin_initial_pressure = create_double_spinbox(0.0, 1000000, 800.0, decimals=2)
        self.spin_initial_sw = create_double_spinbox(0.0, 1.0, 0.05, decimals=4)
        self.spin_initial_sg = create_double_spinbox(0.0, 1.0, 0.9, decimals=4)
        
        grid.addWidget(QLabel("Pressure (bar):"), 0, 0)
        grid.addWidget(self.spin_initial_pressure, 0, 1)
        grid.addWidget(QLabel("Sw:"), 1, 0)
        grid.addWidget(self.spin_initial_sw, 1, 1)
        grid.addWidget(QLabel("Sg:"), 2, 0)
        grid.addWidget(self.spin_initial_sg, 2, 1)
        
        group.setLayout(grid)
        layout.addWidget(group)
    
    def get_values(self):
        return {
            'initial_pressure': self.spin_initial_pressure.value(),
            'initial_sw': self.spin_initial_sw.value(),
            'initial_sg': self.spin_initial_sg.value()
        }


class NaturalFracturesPanel(QWidget):
    """天然裂缝参数面板 - 可复用"""
    def __init__(self, parent=None, prefix=""):
        super().__init__(parent)
        self.prefix = prefix
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        group = QGroupBox("Natural Fractures")
        group.setStyleSheet(groupbox_style())
        grid = QGridLayout()
        
        self.spin_num_fracs = create_spinbox(0, 500, 10)
        self.spin_min_len = create_double_spinbox(0.0, 100000.0, 30.0, decimals=2)
        self.spin_max_len = create_double_spinbox(0.0, 100000.0, 80.0, decimals=2)
        self.spin_aperture = create_double_spinbox(0.0, 10.0, 0.01, decimals=4)
        self.spin_perm = create_double_spinbox(0.0, 1000000.0, 1000.0, decimals=2)
        
        grid.addWidget(QLabel("裂缝数量:"), 0, 0)
        grid.addWidget(self.spin_num_fracs, 0, 1)
        grid.addWidget(QLabel("Min Length (m):"), 1, 0)
        grid.addWidget(self.spin_min_len, 1, 1)
        grid.addWidget(QLabel("Max Length (m):"), 2, 0)
        grid.addWidget(self.spin_max_len, 2, 1)
        grid.addWidget(QLabel("Aperture (m):"), 3, 0)
        grid.addWidget(self.spin_aperture, 3, 1)
        grid.addWidget(QLabel("Permeability (D):"), 4, 0)
        grid.addWidget(self.spin_perm, 4, 1)
        
        group.setLayout(grid)
        layout.addWidget(group)
    
    def get_values(self):
        return {
            'num_fracs': self.spin_num_fracs.value(),
            'min_len': self.spin_min_len.value(),
            'max_len': self.spin_max_len.value(),
            'aperture': self.spin_aperture.value(),
            'perm': self.spin_perm.value()
        }


class HydraulicFracturesPanel(QWidget):
    """人工裂缝参数面板 - 可复用"""
    def __init__(self, parent=None, prefix=""):
        super().__init__(parent)
        self.prefix = prefix
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        group = QGroupBox("Hydraulic Fractures (人工裂缝)")
        group.setStyleSheet(groupbox_style())
        grid = QGridLayout()
        
        self.spin_num_stages = create_spinbox(1, 50, 20)
        self.spin_half_len = create_double_spinbox(0.0, 10000.0, 120.0, decimals=2)
        self.spin_height = create_double_spinbox(0.0, 1000.0, 30.0, decimals=2)
        self.spin_aperture = create_double_spinbox(0.0, 10.0, 0.1, decimals=4)
        self.spin_perm = create_double_spinbox(0.0, 1000000.0, 1000.0, decimals=2)
        self.spin_conductivity = create_double_spinbox(0.0, 1000000.0, 100.0, decimals=2)
        
        grid.addWidget(QLabel("压裂段数:"), 0, 0)
        grid.addWidget(self.spin_num_stages, 0, 1)
        grid.addWidget(QLabel("半缝长 (m):"), 1, 0)
        grid.addWidget(self.spin_half_len, 1, 1)
        grid.addWidget(QLabel("缝高 (m):"), 2, 0)
        grid.addWidget(self.spin_height, 2, 1)
        grid.addWidget(QLabel("Aperture (m):"), 3, 0)
        grid.addWidget(self.spin_aperture, 3, 1)
        grid.addWidget(QLabel("Permeability (D):"), 4, 0)
        grid.addWidget(self.spin_perm, 4, 1)
        grid.addWidget(QLabel("Conductivity (D·m):"), 5, 0)
        grid.addWidget(self.spin_conductivity, 5, 1)
        
        group.setLayout(grid)
        layout.addWidget(group)
    
    def get_values(self):
        return {
            'num_stages': self.spin_num_stages.value(),
            'half_len': self.spin_half_len.value(),
            'height': self.spin_height.value(),
            'aperture': self.spin_aperture.value(),
            'perm': self.spin_perm.value(),
            'conductivity': self.spin_conductivity.value()
        }


class WellParametersPanel(QWidget):
    """井参数面板 - 可复用"""
    def __init__(self, parent=None, prefix=""):
        super().__init__(parent)
        self.prefix = prefix
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        group = QGroupBox("Well Parameters")
        group.setStyleSheet(groupbox_style())
        grid = QGridLayout()
        
        self.spin_well_x = create_double_spinbox(0.0, 100000.0, 500.0, decimals=2)
        self.spin_well_y = create_double_spinbox(0.0, 100000.0, 250.0, decimals=2)
        self.spin_well_z = create_double_spinbox(0.0, 10000.0, 50.0, decimals=2)
        self.spin_well_pressure = create_double_spinbox(0.0, 100000.0, 50.0, decimals=2)
        self.spin_well_radius = create_double_spinbox(0.001, 100.0, 0.05, decimals=3)
        self.spin_well_WI = create_double_spinbox(0.0, 1000000.0, 0.0, decimals=2)
        
        grid.addWidget(QLabel("X (m):"), 0, 0)
        grid.addWidget(self.spin_well_x, 0, 1)
        grid.addWidget(QLabel("Y (m):"), 1, 0)
        grid.addWidget(self.spin_well_y, 1, 1)
        grid.addWidget(QLabel("Z (m):"), 2, 0)
        grid.addWidget(self.spin_well_z, 2, 1)
        grid.addWidget(QLabel("P_bhp (bar):"), 3, 0)
        grid.addWidget(self.spin_well_pressure, 3, 1)
        grid.addWidget(QLabel("Radius (m):"), 4, 0)
        grid.addWidget(self.spin_well_radius, 4, 1)
        grid.addWidget(QLabel("WI:"), 5, 0)
        grid.addWidget(self.spin_well_WI, 5, 1)
        
        group.setLayout(grid)
        layout.addWidget(group)
    
    def get_values(self):
        return {
            'x': self.spin_well_x.value(),
            'y': self.spin_well_y.value(),
            'z': self.spin_well_z.value(),
            'pressure': self.spin_well_pressure.value(),
            'radius': self.spin_well_radius.value(),
            'WI': self.spin_well_WI.value()
        }


class SimulationControlPanel(QWidget):
    """模拟控制参数面板 - 可复用"""
    def __init__(self, parent=None, prefix=""):
        super().__init__(parent)
        self.prefix = prefix
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        group = QGroupBox("Simulation Control")
        group.setStyleSheet(groupbox_style())
        grid = QGridLayout()
        
        self.spin_simulation_time = create_double_spinbox(0.0, 1000000, 100.0, decimals=2)
        self.spin_time_step = create_double_spinbox(0.0, 1000000, 1.0, decimals=4)
        
        grid.addWidget(QLabel("Simulation Time (days):"), 0, 0)
        grid.addWidget(self.spin_simulation_time, 0, 1)
        grid.addWidget(QLabel("Time Step (days):"), 1, 0)
        grid.addWidget(self.spin_time_step, 1, 1)
        
        group.setLayout(grid)
        layout.addWidget(group)
    
    def get_values(self):
        return {
            'simulation_time': self.spin_simulation_time.value(),
            'time_step': self.spin_time_step.value()
        }


class ResultsPanel(QWidget):
    """结果显示面板 - 与原文件一致"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 视图模式选择
        view_group = QGroupBox("View Mode")
        view_group.setStyleSheet(groupbox_style())
        view_layout = QVBoxLayout()
        
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(["Pressure Field", "Fracture Mesh"])
        self.view_mode_combo.setStyleSheet("color: #cccccc; background-color: #3d3d3d;")
        
        view_layout.addWidget(QLabel("Select View:"))
        view_layout.addWidget(self.view_mode_combo)
        view_group.setLayout(view_layout)
        layout.addWidget(view_group)
        
        # 显示选项
        group = QGroupBox("Visualization Options")
        group.setStyleSheet(groupbox_style())
        vlayout = QVBoxLayout()
        
        self.combo_field = QComboBox()
        self.combo_field.addItems(["Pressure", "Temperature", "Stress"])
        self.combo_field.setStyleSheet("color: #cccccc; background-color: #3d3d3d;")
        
        self.check_show_grid = QCheckBox("Show Grid Lines")
        self.check_show_grid.setChecked(False)
        
        self.check_show_fractures = QCheckBox("Show Fractures")
        self.check_show_fractures.setChecked(False)
        
        vlayout.addWidget(QLabel("Display Field:"))
        vlayout.addWidget(self.combo_field)
        vlayout.addWidget(self.check_show_grid)
        vlayout.addWidget(self.check_show_fractures)
        
        group.setLayout(vlayout)
        layout.addWidget(group)
        layout.addStretch()
    
    def get_values(self):
        """获取显示参数"""
        return {
            'view_mode': self.view_mode_combo.currentText(),
            'show_grid': self.check_show_grid.isChecked(),
            'show_fractures': self.check_show_fractures.isChecked(),
            'field': self.combo_field.currentText()
        }
