"""
数据模型模块
定义模拟数据结构和输出捕获
"""
import sys
import io


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
        self.grid_lines = []  # LGR网格线数据
        self.interpolated_pressure = []  # C++插值后的压力场数据
        
    def generate_from_cpp(self, sim_result, nx, ny, nz, lx, ly, lz, grid_lines=None, interpolated_pressure=None):
        """从C++结果生成数据"""
        self.grid_info = {'nx': nx, 'ny': ny, 'nz': nz, 'Lx': lx, 'Ly': ly, 'Lz': lz}
        self.pressure_field = list(sim_result.pressure_field)
        self.temperature_field = list(sim_result.temperature_field) if sim_result.temperature_field else []
        self.stress_field = list(sim_result.stress_field) if sim_result.stress_field else []
        self.grid_lines = list(grid_lines) if grid_lines else []
        self.interpolated_pressure = list(interpolated_pressure) if interpolated_pressure else []
        
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
