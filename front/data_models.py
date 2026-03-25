"""
数据模型模块
定义模拟数据结构和输出捕获
"""
import sys
import io
import json


def _safe_to_list(value):
    """避免对 numpy/pybind 数组做布尔判断。"""
    if value is None:
        return []
    return list(value)


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


class CornerPointCell:
    """角点网格单元数据结构
    
    角点编号约定：
    底面(zmin): 0:(xmin,ymin) 1:(xmax,ymin) 2:(xmax,ymax) 3:(xmin,ymax)
    顶面(zmax): 4:(xmin,ymin) 5:(xmax,ymin) 6:(xmax,ymax) 7:(xmin,ymax)
    """
    def __init__(self, cell_id=-1):
        self.id = cell_id
        self.corners = [(0.0, 0.0, 0.0) for _ in range(8)]
        self.pressure = 0.0
        self.ix = 0
        self.iy = 0
        self.iz = 0
    
    def set_corners(self, corners):
        """设置8个角点坐标"""
        if len(corners) == 8:
            self.corners = [(c[0], c[1], c[2]) for c in corners]
    
    def to_dict(self):
        return {
            'id': self.id,
            'corners': [list(c) for c in self.corners],
            'pressure': self.pressure,
            'ix': self.ix,
            'iy': self.iy,
            'iz': self.iz
        }
    
    @classmethod
    def from_dict(cls, data):
        cell = cls(data.get('id', -1))
        cell.corners = [tuple(c) for c in data.get('corners', [(0,0,0)]*8)]
        cell.pressure = data.get('pressure', 0.0)
        cell.ix = data.get('ix', 0)
        cell.iy = data.get('iy', 0)
        cell.iz = data.get('iz', 0)
        return cell


class CornerPointGridData:
    """角点网格数据结构"""
    def __init__(self):
        self.nx = 20
        self.ny = 10
        self.nz = 5
        self.lx = 1000.0
        self.ly = 500.0
        self.lz = 100.0
        self.cells = []
        self.fractures = []
        self.min_pressure = 0.0
        self.max_pressure = 200.0
    
    def to_dict(self):
        return {
            'nx': self.nx,
            'ny': self.ny,
            'nz': self.nz,
            'lx': self.lx,
            'ly': self.ly,
            'lz': self.lz,
            'cells': [c.to_dict() for c in self.cells],
            'fractures': self.fractures,
            'min_pressure': self.min_pressure,
            'max_pressure': self.max_pressure
        }
    
    @classmethod
    def from_dict(cls, data):
        grid = cls()
        grid.nx = data.get('nx', 20)
        grid.ny = data.get('ny', 10)
        grid.nz = data.get('nz', 5)
        grid.lx = data.get('lx', 1000.0)
        grid.ly = data.get('ly', 500.0)
        grid.lz = data.get('lz', 100.0)
        grid.cells = [CornerPointCell.from_dict(c) for c in data.get('cells', [])]
        grid.fractures = data.get('fractures', [])
        grid.min_pressure = data.get('min_pressure', 0.0)
        grid.max_pressure = data.get('max_pressure', 200.0)
        return grid


class SimulationData:
    """模拟数据结构"""
    def __init__(self):
        self.grid_info = {'nx': 20, 'ny': 20, 'nz': 5, 'Lx': 500.0, 'Ly': 500.0, 'Lz': 100.0}
        self.pressure_field = []
        self.temperature_field = []
        self.stress_field = []
        self.fractures = []
        self.wells = []
        self.grid_lines = []
        self.interpolated_pressure = []
        self.corner_point_grid = None
        
    def generate_from_cpp(self, sim_result, nx, ny, nz, lx, ly, lz, grid_lines=None, interpolated_pressure=None):
        """从C++结果生成数据"""
        self.grid_info = {'nx': nx, 'ny': ny, 'nz': nz, 'Lx': lx, 'Ly': ly, 'Lz': lz}
        self.pressure_field = list(sim_result.pressure_field)
        self.temperature_field = _safe_to_list(sim_result.temperature_field)
        self.stress_field = _safe_to_list(sim_result.stress_field)
        self.grid_lines = _safe_to_list(grid_lines)
        self.interpolated_pressure = _safe_to_list(interpolated_pressure)
        
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
    
    def generate_mock_data(self, nx, ny, nz, lx, ly, lz, num_fracs, min_len, max_len,
                          well_x, well_y, well_z, well_pressure):
        """生成模拟数据"""
        self.grid_info = {'nx': nx, 'ny': ny, 'nz': nz, 'Lx': lx, 'Ly': ly, 'Lz': lz}
        
        # 生成压力场
        self.pressure_field = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    x = (i + 0.5) * lx / nx
                    y = (j + 0.5) * ly / ny
                    z = (k + 0.5) * lz / nz
                    
                    # 从井压向四周递减
                    dist = ((x - well_x)**2 + (y - well_y)**2 + (z - well_z)**2)**0.5
                    max_dist = (lx**2 + ly**2 + lz**2)**0.5
                    pressure = well_pressure + (200.0 - well_pressure) * (dist / max_dist)
                    
                    self.pressure_field.append((x, y, z, pressure))
        
        # 生成裂缝
        self.fractures = []
        for i in range(num_fracs):
            x = lx * 0.2 + (lx * 0.6) * i / max(1, num_fracs - 1)
            y = ly / 2
            z = lz / 2
            
            length = min_len + (max_len - min_len) * (i % 3) / 2
            height = 30.0
            
            frac = {
                'id': i,
                'points': [
                    (x, y - length/2, z - height/2),
                    (x, y + length/2, z - height/2),
                    (x, y + length/2, z + height/2),
                    (x, y - length/2, z + height/2)
                ]
            }
            self.fractures.append(frac)
        
        print(f"Generated {len(self.pressure_field)} pressure points, {len(self.fractures)} fractures")

    def to_dict(self):
        """序列化模拟结果，供子进程返回给UI进程。"""
        return {
            'grid_info': dict(self.grid_info),
            'pressure_field': [list(point) for point in self.pressure_field],
            'temperature_field': [list(point) for point in self.temperature_field],
            'stress_field': [list(point) for point in self.stress_field],
            'grid_lines': [list(line) for line in self.grid_lines],
            'interpolated_pressure': [list(point) for point in self.interpolated_pressure],
            'fractures': [
                {
                    'id': fracture.get('id'),
                    'points': [list(point) for point in fracture.get('points', [])]
                }
                for fracture in self.fractures
            ],
            'wells': list(self.wells),
            'corner_point_grid': self.corner_point_grid.to_dict() if self.corner_point_grid else None,
        }

    def load_dict(self, payload):
        """从序列化结果恢复模拟数据。"""
        self.grid_info = dict(payload.get('grid_info', self.grid_info))
        self.pressure_field = [tuple(point) for point in payload.get('pressure_field', [])]
        self.temperature_field = [tuple(point) for point in payload.get('temperature_field', [])]
        self.stress_field = [tuple(point) for point in payload.get('stress_field', [])]
        self.grid_lines = [tuple(line) for line in payload.get('grid_lines', [])]
        self.interpolated_pressure = [
            tuple(point) for point in payload.get('interpolated_pressure', [])
        ]
        self.fractures = [
            {
                'id': fracture.get('id'),
                'points': [tuple(point) for point in fracture.get('points', [])]
            }
            for fracture in payload.get('fractures', [])
        ]
        self.wells = list(payload.get('wells', []))
        cpg_data = payload.get('corner_point_grid')
        if cpg_data:
            self.corner_point_grid = CornerPointGridData.from_dict(cpg_data)
        else:
            self.corner_point_grid = None

    def save_json(self, output_path):
        """将模拟结果写入JSON文件。"""
        with open(output_path, 'w', encoding='utf-8') as fp:
            json.dump(self.to_dict(), fp)

    def load_json(self, input_path):
        """从JSON文件加载模拟结果。"""
        with open(input_path, 'r', encoding='utf-8') as fp:
            self.load_dict(json.load(fp))


def load_corner_point_grid_from_csv(coord_csv_path, zcorn_csv_path):
    """从COORD和ZCORN CSV文件加载角点网格数据
    
    Args:
        coord_csv_path: COORD文件路径，包含网格点坐标
        zcorn_csv_path: ZCORN文件路径，包含每个单元的8个Z值
    
    Returns:
        CornerPointGridData对象
    """
    import csv
    
    grid = CornerPointGridData()
    
    coord_data = {}
    with open(coord_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 获取列名（处理可能的BOM或特殊字符）
            keys = list(row.keys())
            xi = int(row[keys[0]])  # X(I)
            yj = int(row[keys[1]])  # Y(J)
            zk = row[keys[2]].strip()       # Z(K) - 去除空白字符
            x = float(row[keys[3]]) # 坐标X
            y = float(row[keys[4]]) # 坐标Y
            z = float(row[keys[5]]) # 坐标Z
            
            key = (xi, yj, zk)
            coord_data[key] = (x, y, z)
    
    # 调试：打印前几个COORD数据
    print(f"COORD data samples (total {len(coord_data)} points):")
    for i, (k, v) in enumerate(list(coord_data.items())[:5]):
        print(f"  {k} -> {v}")
    
    max_i = max(k[0] for k in coord_data.keys())
    max_j = max(k[1] for k in coord_data.keys())
    
    zcorn_data = {}
    with open(zcorn_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys = list(row.keys())
            xi = int(row[keys[0]])  # X(I)
            yj = int(row[keys[1]])  # Y(J)
            zk = int(row[keys[2]])  # Z(K)
            z_values = [float(row[keys[i]]) for i in range(3, 11)]  # Z1-Z8
            zcorn_data[(xi, yj, zk)] = z_values
    
    if zcorn_data:
        max_k = max(k[2] for k in zcorn_data.keys())
    else:
        max_k = 1
    
    # COORD定义的是柱子坐标，每个柱子有顶和底
    # 柱子数量 = max_i x max_j
    # 单元数量 = (max_i - 1) x (max_j - 1) x max_k (从ZCORN确定)
    
    # 从ZCORN确定实际的网格维度
    if zcorn_data:
        zcorn_max_i = max(k[0] for k in zcorn_data.keys())
        zcorn_max_j = max(k[1] for k in zcorn_data.keys())
        zcorn_max_k = max(k[2] for k in zcorn_data.keys())
        grid.nx = zcorn_max_i
        grid.ny = zcorn_max_j
        grid.nz = zcorn_max_k
    else:
        grid.nx = max_i - 1
        grid.ny = max_j - 1
        grid.nz = max_k
    
    all_x = [c[0] for c in coord_data.values()]
    all_y = [c[1] for c in coord_data.values()]
    all_z = [c[2] for c in coord_data.values()]
    grid.lx = max(all_x) - min(all_x) if all_x else 1000.0
    grid.ly = max(all_y) - min(all_y) if all_y else 500.0
    grid.lz = max(all_z) - min(all_z) if all_z else 100.0
    
    print(f"Grid dimensions: {grid.nx}x{grid.ny}x{grid.nz}")
    print(f"COORD dimensions: {max_i}x{max_j}")
    
    cell_id = 0
    for iz in range(1, grid.nz + 1):
        for iy in range(1, grid.ny + 1):
            for ix in range(1, grid.nx + 1):
                cell = CornerPointCell(cell_id)
                cell.ix = ix - 1
                cell.iy = iy - 1
                cell.iz = iz - 1
                
                zcorn = zcorn_data.get((ix, iy, iz), [0]*8)
                
                # 获取当前单元的8个角点坐标
                # 角点顺序: 底面4个(逆时针), 顶面4个(逆时针)
                # p0-p3: 底面, p4-p7: 顶面
                # ZCORN顺序: Z1-Z4是顶面，Z5-Z8是底面
                
                # 从COORD获取XY坐标 (柱子坐标)
                # 单元(ix,iy)的4个角对应柱子: (ix,iy), (ix+1,iy), (ix+1,iy+1), (ix,iy+1)
                p0_xy = coord_data.get((ix, iy, '底'), coord_data.get((ix, iy, 'bottom'), None))
                p1_xy = coord_data.get((ix + 1, iy, '底'), coord_data.get((ix + 1, iy, 'bottom'), None))
                p2_xy = coord_data.get((ix + 1, iy + 1, '底'), coord_data.get((ix + 1, iy + 1, 'bottom'), None))
                p3_xy = coord_data.get((ix, iy + 1, '底'), coord_data.get((ix, iy + 1, 'bottom'), None))
                
                p4_xy = coord_data.get((ix, iy, '顶'), coord_data.get((ix, iy, 'top'), None))
                p5_xy = coord_data.get((ix + 1, iy, '顶'), coord_data.get((ix + 1, iy, 'top'), None))
                p6_xy = coord_data.get((ix + 1, iy + 1, '顶'), coord_data.get((ix + 1, iy + 1, 'top'), None))
                p7_xy = coord_data.get((ix, iy + 1, '顶'), coord_data.get((ix, iy + 1, 'top'), None))
                
                # 组合XYZ坐标 (XY来自COORD, Z来自ZCORN)
                # ZCORN: Z1,Z2,Z3,Z4是顶面(从西北开始顺时针), Z5,Z6,Z7,Z8是底面
                p0 = (p0_xy[0] if p0_xy else 0, p0_xy[1] if p0_xy else 0, zcorn[4])  # Z5
                p1 = (p1_xy[0] if p1_xy else 0, p1_xy[1] if p1_xy else 0, zcorn[5])  # Z6
                p2 = (p2_xy[0] if p2_xy else 0, p2_xy[1] if p2_xy else 0, zcorn[6])  # Z7
                p3 = (p3_xy[0] if p3_xy else 0, p3_xy[1] if p3_xy else 0, zcorn[7])  # Z8
                p4 = (p4_xy[0] if p4_xy else 0, p4_xy[1] if p4_xy else 0, zcorn[0])  # Z1
                p5 = (p5_xy[0] if p5_xy else 0, p5_xy[1] if p5_xy else 0, zcorn[1])  # Z2
                p6 = (p6_xy[0] if p6_xy else 0, p6_xy[1] if p6_xy else 0, zcorn[2])  # Z3
                p7 = (p7_xy[0] if p7_xy else 0, p7_xy[1] if p7_xy else 0, zcorn[3])  # Z4
                
                cell.corners = [p0, p1, p2, p3, p4, p5, p6, p7]
                
                # 默认压力值（仅用于占位）
                cell.pressure = 100.0
                
                grid.cells.append(cell)
                cell_id += 1
    
    if grid.cells:
        pressures = [c.pressure for c in grid.cells]
        grid.min_pressure = min(pressures)
        grid.max_pressure = max(pressures)
    
    print(f"Loaded {len(grid.cells)} corner point cells from CSV")
    print(f"Grid dimensions: {grid.nx}x{grid.ny}x{grid.nz}")
    
    return grid
