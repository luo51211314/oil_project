# EDFM 油藏模拟器 - Python原型

基于 Python + PyQt5 + VTK + C++ (pybind11) 的 ResInsight 风格油藏模拟器

## 项目概述

本项目实现了一个 ResInsight 风格的油藏模拟器，采用以下架构：
- **front/**: UI组件和输入处理 (PyQt5)
- **visual/**: VTK可视化渲染
- **src/**: C++ EDFM算法实现
- **build/**: 编译后的C++模块（自动生成）

## 功能特性

- **3D可视化**: 使用VTK实现平滑的压力场可视化
- **裂缝建模**: 支持嵌入式离散裂缝模型 (EDFM)
- **黑油模拟**: 三相（水-油-气）油藏模拟
- **交互式界面**: ResInsight风格的参数面板界面
- **实时输出**: 模拟状态和统计信息显示

## 项目结构

```
python_prototype/
├── front/                      # 前端UI模块
│   ├── __init__.py
│   ├── data_models.py         # 模拟数据结构
│   ├── input_panel.py         # 输入面板组件
│   └── main_window.py         # 主应用窗口
├── visual/                     # 可视化模块
│   ├── __init__.py
│   └── vtk_renderer.py        # VTK渲染组件
├── src/                        # C++源代码
│   └── edfm_3d_blackoil_lgr.cpp   # EDFM黑油算法
├── build/                      # 构建目录（自动生成）
│   └── Release/
│       └── edfm_core.pyd      # 编译后的Python模块
├── main.py                     # 应用程序入口
├── CMakeLists.txt             # CMake构建配置
└── README.md                  # 本文件
```

## 环境要求

### 前置条件

- **Python**: 3.9 或更高版本
- **CMake**: 3.15 或更高版本
- **C++编译器**: MSVC (Windows) 或 GCC/Clang (Linux/Mac)
- **Eigen3**: 3.4.0（仅头文件库）

### Python依赖

在conda/虚拟环境中安装以下包：

```bash
# 使用conda
conda install pyqt5 vtk numpy

# 使用pip
pip install PyQt5 vtk numpy

# pybind11（用于构建C++模块）
pip install pybind11
```

### C++依赖

1. **Eigen3**: 从 https://eigen.tuxfamily.org/ 下载
   - 解压到 `D:/eigen-3.4.0`（或在CMakeLists.txt中更新路径）

2. **pybind11**: 通过pip/conda安装，然后记录cmake路径：
   - 通常位于: `D:/Anaconda/envs/oil/Lib/site-packages/pybind11/share/cmake/pybind11`

## 构建C++模块

### Windows (Visual Studio)

```bash
# 进入构建目录
mkdir build
cd build

# 使用CMake配置
cmake .. -DCMAKE_BUILD_TYPE=Release

# 构建
cmake --build . --config Release
```

编译后的模块 `edfm_core.pyd` 将生成在 `build/Release/` 目录中。

### Linux/Mac

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

## 使用方法

### 运行应用程序

```bash
# 在python_prototype目录下
python main.py
```

### 基本工作流程

1. **设置网格参数**（Grid标签页）
   - Nx, Ny, Nz: 网格维度
   - Lx, Ly, Lz: 域大小（米）

2. **配置井参数**（Wells标签页）
   - 井位置（X, Y, Z）
   - 井底压力（MPa）
   - 井半径（米）

3. **定义裂缝**（Fractures标签页）
   - 裂缝数量
   - 长度范围（最小/最大）
   - 开度（米）

4. **运行模拟**
   - 点击绿色的 "▶ Run Simulation" 按钮
   - 在3D可视化区域查看结果

5. **可视化选项**（Results标签页）
   - 在 "Pressure Field" 和 "Fracture Mesh" 视图之间切换
   - 切换网格线和裂缝可见性

## 配置

### CMakeLists.txt

在 `CMakeLists.txt` 中更新以下路径以匹配您的系统：

```cmake
# Eigen3路径
set(EIGEN3_INCLUDE_DIR "D:/eigen-3.3.8")

# pybind11路径
set(pybind11_DIR "D:/Anaconda/envs/oil/Lib/site-packages/pybind11/share/cmake/pybind11")
```

### 默认参数

默认模拟参数在 `main_window.py` 中定义：
- 网格: 20 x 10 x 2 单元
- 域: 1000 x 500 x 50 米
- 裂缝: 100条（长度30-80m，开度0.001m）
- 井压力: 50 MPa
- 模拟时间: 100天

## 故障排除

### C++模块未找到

如果看到 "C++ module not found, using mock data"：
1. 确保C++模块构建成功
2. 检查 `edfm_core.pyd` 是否存在于 `build/Release/` 目录
3. 验证Python架构与编译模块匹配（64位）

### VTK渲染问题

如果3D可视化不显示：
1. 检查VTK安装: `python -c "import vtk; print(vtk.VTK_VERSION)"`
2. 确保显卡驱动已更新
3. 尝试软件渲染: `set VTK_OPENGL_HAS_OSMESA=1`

### CMake配置错误

如果CMake找不到pybind11：
1. 安装pybind11: `pip install pybind11`
2. 查找cmake目录: `python -c "import pybind11; print(pybind11.get_cmake_dir())"`
3. 使用正确路径更新 `CMakeLists.txt`

## 算法详情

EDFM（嵌入式离散裂缝模型）算法实现了：
- **LGR（局部网格细化）**: 裂缝周围的自适应网格细化
- **Newton-Raphson求解器**: 压力非线性方程求解
- **三相流动**: 水、油和气流相建模
- **PVT属性**: 压力-体积-温度关系

详细实现请参见 `src/edfm_3d_blackoil_lgr.cpp`。

## 许可证

本项目仅供研究和教育用途。

## 致谢

- ResInsight (https://resinsight.org/) - UI设计灵感
- VTK (https://vtk.org/) - 3D可视化
- Eigen (https://eigen.tuxfamily.org/) - 线性代数
- pybind11 (https://pybind11.readthedocs.io/) - Python-C++绑定
