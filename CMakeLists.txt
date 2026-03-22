cmake_minimum_required(VERSION 3.15)
project(edfm_python_bindings)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加Eigen
# lxw
# set(EIGEN3_INCLUDE_DIR "D:/eigen-3.3.8")
# ypw 
# set(EIGEN3_INCLUDE_DIR "D:/mingw64/eigen")
# wzq
set(EIGEN3_INCLUDE_DIR "D:/vtk/vcpkg/installed/x64-windows/include/eigen3")

include_directories(${EIGEN3_INCLUDE_DIR})

# pybind11
# lxw
# set(pybind11_DIR "D:/Anaconda/envs/oil/Lib/site-packages/pybind11/share/cmake/pybind11")
# ypw
# set(pybind11_DIR "D:/anaconda3/envs/transformers/Lib/site-packages/pybind11/share/cmake/pybind11")
# wzq
set(pybind11_DIR "D:/pybind11_site/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)

# 创建模块 - 包含完整的EDFM算法
pybind11_add_module(edfm_core
    src/lgr/edfm_3d_blackoil_lgr.cpp
)


# 创建新模块 - Corner Point Grid EDFM
pybind11_add_module(edfm_core_corner
    src/basic/Corner_Point_Grid_edfm_3d_water_oil.cpp
)


target_include_directories(edfm_core PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries(edfm_core PRIVATE pybind11::module)


target_include_directories(edfm_core_corner PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries(edfm_core_corner PRIVATE pybind11::module)