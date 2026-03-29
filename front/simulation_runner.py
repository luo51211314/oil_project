"""
子进程模拟运行器。
负责执行模拟、保留算法 stdout 输出，并将结果写入 JSON 供 UI 进程读取。
"""
import argparse
import csv
import importlib
import json
import math
import os
import sys

from .data_models import SimulationData


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE_SEARCH_DIRS = [
    os.path.join(PROJECT_ROOT, 'build', 'Release'),
    os.path.join(PROJECT_ROOT, 'build'),
    os.path.join(PROJECT_ROOT, 'build_nmake'),
    os.path.join(PROJECT_ROOT, 'build_nmake', 'Release'),
    os.path.join(PROJECT_ROOT, 'build_nmake', 'Debug'),
]
for module_dir in MODULE_SEARCH_DIRS:
    if os.path.isdir(module_dir) and module_dir not in sys.path:
        sys.path.append(module_dir)

_BLACK_OIL_MODULE = None
_BLACK_OIL_IMPORT_ERROR = None
_CORNER_EDFM_MODULE = None
_CORNER_EDFM_IMPORT_ERROR = None

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)


def run_simulation(params):
    """执行模拟并返回 SimulationData。"""
    algorithm = params.get('algorithm', 'black_oil')
    if algorithm == 'corner_edfm':
        return run_corner_edfm_simulation(params)
    return run_black_oil_simulation(params)


def load_black_oil_module():
    """按需加载 Black Oil C++ 模块，避免与其他 pybind 模块类型注册冲突。"""
    global _BLACK_OIL_MODULE, _BLACK_OIL_IMPORT_ERROR
    if _BLACK_OIL_MODULE is not None:
        return _BLACK_OIL_MODULE
    if _BLACK_OIL_IMPORT_ERROR is not None:
        raise RuntimeError(f"edfm_core module is not available: {_BLACK_OIL_IMPORT_ERROR}")

    try:
        _BLACK_OIL_MODULE = importlib.import_module('edfm_core')
        return _BLACK_OIL_MODULE
    except Exception as exc:
        _BLACK_OIL_IMPORT_ERROR = exc
        raise RuntimeError(f"edfm_core module is not available: {exc}") from exc


def load_corner_edfm_module():
    """按需加载 Corner EDFM C++ 模块，避免与 edfm_core 同时注册同名 pybind 类型。"""
    global _CORNER_EDFM_MODULE, _CORNER_EDFM_IMPORT_ERROR
    if _CORNER_EDFM_MODULE is not None:
        return _CORNER_EDFM_MODULE
    if _CORNER_EDFM_IMPORT_ERROR is not None:
        raise RuntimeError(f"edfm_core_corner module is not available: {_CORNER_EDFM_IMPORT_ERROR}")

    try:
        _CORNER_EDFM_MODULE = importlib.import_module('edfm_core_corner')
        return _CORNER_EDFM_MODULE
    except Exception as exc:
        _CORNER_EDFM_IMPORT_ERROR = exc
        raise RuntimeError(f"edfm_core_corner module is not available: {exc}") from exc


def run_black_oil_simulation(params):
    """执行 Black Oil 模拟并返回 SimulationData。"""
    sim_data = SimulationData()

    nx = int(params['nx'])
    ny = int(params['ny'])
    nz = int(params['nz'])
    lx = float(params['lx'])
    ly = float(params['ly'])
    lz = float(params['lz'])
    num_fracs = int(params['num_fracs'])
    min_len = float(params['min_len'])
    max_len = float(params['max_len'])
    aperture = float(params['aperture'])
    well_x = float(params['well_x'])
    well_y = float(params['well_y'])
    well_z = float(params['well_z'])
    well_pressure = float(params['well_pressure'])
    region_num_fracs = int(params.get('region_num_fracs', 0))
    region_x_min = float(params.get('region_x_min', 0.0))
    region_x_max = float(params.get('region_x_max', 0.0))
    region_y_min = float(params.get('region_y_min', 0.0))
    region_y_max = float(params.get('region_y_max', 0.0))
    region_z_min = float(params.get('region_z_min', 0.0))
    region_z_max = float(params.get('region_z_max', 0.0))

    try:
        edfm_core = load_black_oil_module()
    except RuntimeError as exc:
        print(f"{exc}, using mock data")
        edfm_core = None

    if edfm_core is not None:
        sim = edfm_core.EDFMSimulator()
        sim.setGridParameters(nx, ny, nz, lx, ly, lz)
        sim.setFractureParameters(num_fracs, min_len, max_len, math.pi / 3.0, 0.0, math.pi, aperture, 10000.0)
        sim.setSimulationParameters(100.0, 1.0, 0.2, 0.001, 0.001, 0.0001)
        sim.setWellParameters(well_x, well_y, well_z, 0.05, well_pressure)
        if region_num_fracs > 0:
            if hasattr(sim, 'setRegionFractureParameters'):
                sim.setRegionFractureParameters(
                    region_num_fracs,
                    region_x_min, region_x_max,
                    region_y_min, region_y_max,
                    region_z_min, region_z_max,
                )
            else:
                print("WARNING: edfm_core missing setRegionFractureParameters(); region fractures disabled")
        result = sim.runSimulation()
        grid_lines = []
        interpolated_pressure = []
        if hasattr(sim, 'getGridLines'):
            grid_lines = sim.getGridLines()
        else:
            print("WARNING: edfm_core missing getGridLines(); using coarse grid visualization")
        if hasattr(sim, 'getInterpolatedPressureField'):
            interpolated_pressure = sim.getInterpolatedPressureField(50, 25, 10)
        else:
            print("WARNING: edfm_core missing getInterpolatedPressureField(); using Python fallback interpolation")
        sim_data.generate_from_cpp(
            result,
            nx,
            ny,
            nz,
            lx,
            ly,
            lz,
            grid_lines,
            interpolated_pressure,
        )
        if not sim_data.interpolated_pressure:
            sim_data.interpolated_pressure = build_interpolated_pressure_from_leaf_data(sim_data)
    else:
        sim_data.generate_mock_data(
            nx, ny, nz, lx, ly, lz,
            num_fracs, min_len, max_len,
            well_x, well_y, well_z, well_pressure
        )

    return sim_data


def load_corner_grid_info(coord_file, zcorn_file):
    """从 COORD/ZCORN CSV 中静默提取网格维度和范围。"""
    coord_points = []
    with open(coord_file, 'r', encoding='utf-8') as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            keys = list(row.keys())
            coord_points.append((
                float(row[keys[3]]),
                float(row[keys[4]]),
                float(row[keys[5]]),
            ))

    zcorn_dims = []
    with open(zcorn_file, 'r', encoding='utf-8') as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            keys = list(row.keys())
            zcorn_dims.append((
                int(row[keys[0]]),
                int(row[keys[1]]),
                int(row[keys[2]]),
            ))

    if zcorn_dims:
        nx = max(item[0] for item in zcorn_dims)
        ny = max(item[1] for item in zcorn_dims)
        nz = max(item[2] for item in zcorn_dims)
    else:
        nx = ny = nz = 0

    if coord_points:
        xs = [point[0] for point in coord_points]
        ys = [point[1] for point in coord_points]
        zs = [point[2] for point in coord_points]
        lx = max(xs) - min(xs)
        ly = max(ys) - min(ys)
        lz = max(zs) - min(zs)
    else:
        lx = ly = lz = 0.0

    return nx, ny, nz, lx, ly, lz


def run_corner_edfm_simulation(params):
    """执行 Corner EDFM 模拟并返回 SimulationData。"""
    edfm_core_corner = load_corner_edfm_module()

    coord_file = params.get('coord_file', '')
    zcorn_file = params.get('zcorn_file', '')
    if not coord_file or not zcorn_file:
        raise ValueError("Corner EDFM requires both COORD and ZCORN files")

    sim = edfm_core_corner.EDFMSimulator()
    sim.setCornerPointFiles(coord_file, zcorn_file)
    sim.setFractureParameters(
        int(params.get('num_fracs', 0)),
        float(params.get('min_len', 30.0)),
        float(params.get('max_len', 80.0)),
        float(params.get('max_dip', math.pi / 3.0)),
        float(params.get('min_strike', 0.0)),
        float(params.get('max_strike', math.pi)),
        float(params.get('aperture', 0.001)),
        float(params.get('frac_perm', 1000.0)),
    )
    sim.setHydraulicFractureParameters(
        int(params.get('hf_count', 0)) if params.get('hf_enabled', True) else 0,
        float(params.get('hf_well_length', 800.0)),
        float(params.get('hf_length', 50.0)),
        float(params.get('hf_height', 40.0)),
        float(params.get('hf_aperture', 0.01)),
        float(params.get('hf_perm', 1000.0)),
        float(params.get('hf_center_x', -1.0)),
        float(params.get('hf_center_y', -1.0)),
        float(params.get('hf_center_z', -1.0)),
    )
    sim.setWellParameters(
        float(params.get('well_radius', 0.05)),
        float(params.get('well_pressure', 50.0)),
    )
    sim.setSimulationParameters(float(params.get('simulation_time', 100.0)))

    result = sim.runSimulation()
    nx, ny, nz, lx, ly, lz = load_corner_grid_info(coord_file, zcorn_file)

    sim_data = SimulationData()
    sim_data.generate_from_cpp(result, nx, ny, nz, lx, ly, lz, [], [])
    if not sim_data.interpolated_pressure:
        sim_data.interpolated_pressure = build_interpolated_pressure_from_leaf_data(sim_data)
    return sim_data


def build_interpolated_pressure_from_leaf_data(sim_data, nx=50, ny=25, nz=10):
    """旧版 edfm_core 缺少插值接口时，使用最近邻生成规则压力场。"""
    field_data = sim_data.pressure_field
    if not field_data:
        return []

    xs = [point[0] for point in field_data]
    ys = [point[1] for point in field_data]
    zs = [point[2] for point in field_data]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)

    interpolated = []
    for k in range(nz):
        z = min_z + (max_z - min_z) * k / (nz - 1) if nz > 1 else min_z
        for j in range(ny):
            y = min_y + (max_y - min_y) * j / (ny - 1) if ny > 1 else min_y
            for i in range(nx):
                x = min_x + (max_x - min_x) * i / (nx - 1) if nx > 1 else min_x
                nearest = min(
                    field_data,
                    key=lambda point: (
                        (point[0] - x) * (point[0] - x)
                        + (point[1] - y) * (point[1] - y)
                        + (point[2] - z) * (point[2] - z)
                    ),
                )
                interpolated.append((x, y, z, nearest[3]))
    return interpolated


def main():
    parser = argparse.ArgumentParser(description="Run EDFM simulation in a child process.")
    parser.add_argument("--output", required=True, help="Path to the JSON result file.")
    parser.add_argument("--params", required=True, help="JSON-encoded simulation parameters.")
    args = parser.parse_args()

    try:
        params = json.loads(args.params)
        sim_data = run_simulation(params)
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        sim_data.save_json(args.output)
        return 0
    except Exception as exc:
        print(f"RUNNER ERROR: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
