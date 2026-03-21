"""
子进程模拟运行器。
负责执行模拟、保留算法 stdout 输出，并将结果写入 JSON 供 UI 进程读取。
"""
import argparse
import json
import math
import os
import sys

from .data_models import SimulationData


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_RELEASE_DIR = os.path.join(PROJECT_ROOT, 'build', 'Release')
if BUILD_RELEASE_DIR not in sys.path:
    sys.path.insert(0, BUILD_RELEASE_DIR)

try:
    import edfm_core
    HAS_CPP_MODULE = True
except ImportError as exc:
    HAS_CPP_MODULE = False
    print(f"C++ module not found: {exc}, using mock data")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)


def run_simulation(params):
    """执行模拟并返回 SimulationData。"""
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

    if HAS_CPP_MODULE:
        sim = edfm_core.EDFMSimulator()
        sim.setGridParameters(nx, ny, nz, lx, ly, lz)
        sim.setFractureParameters(num_fracs, min_len, max_len, math.pi / 3.0, 0.0, math.pi, aperture, 10000.0)
        sim.setSimulationParameters(100.0, 1.0, 0.2, 0.001, 0.001, 0.0001)
        sim.setWellParameters(well_x, well_y, well_z, 0.05, well_pressure)
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


def build_interpolated_pressure_from_leaf_data(sim_data, nx=50, ny=25, nz=10):
    """旧版 edfm_core 缺少插值接口时，使用最近邻生成规则压力场。"""
    field_data = sim_data.pressure_field
    if not field_data:
        return []

    lx = float(sim_data.grid_info['Lx'])
    ly = float(sim_data.grid_info['Ly'])
    lz = float(sim_data.grid_info['Lz'])

    interpolated = []
    for k in range(nz):
        z = lz * k / (nz - 1) if nz > 1 else 0.0
        for j in range(ny):
            y = ly * j / (ny - 1) if ny > 1 else 0.0
            for i in range(nx):
                x = lx * i / (nx - 1) if nx > 1 else 0.0
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
