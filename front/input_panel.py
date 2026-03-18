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
    
    def get_values(self):
        """获取裂缝参数"""
        return {
            'num_fracs': self.spin_num_fracs.value(),
            'min_len': self.spin_min_len.value(),
            'max_len': self.spin_max_len.value(),
            'aperture': self.spin_aperture.value()
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
