"""
VTK可视化渲染模块
负责所有VTK相关的3D渲染功能
"""
import vtk
import numpy as np


class VTKRenderer:
    """VTK渲染器 - 高内聚的可视化组件"""
    
    def __init__(self, vtk_widget):
        self.vtk_widget = vtk_widget
        self.renderer = vtk_widget.renderer
        self.cache = {
            'pressure_actor': None,
            'fracture_actors': [],
            'scalar_bar': None,
            'grid_lines_actor': None,
            'data_hash': None
        }
    
    def clear_cache(self):
        """清除VTK缓存 - 清除所有算法的数据"""
        # black_oil算法缓存
        self.cache['pressure_actor'] = None
        self.cache['fracture_actors'] = []
        self.cache['scalar_bar'] = None
        self.cache['grid_lines_actor'] = None
        self.cache['data_hash'] = None
        
        # corner point grid算法缓存
        self.cache['corner_actor'] = None
        self.cache['corner_surface_actor'] = None
        self.cache['corner_grid_hash'] = None
        self.cache['z_exaggeration'] = 1.0
        self.cache['z_min_for_exaggeration'] = 0.0
        
        # 井缓存
        self.cache['well_actors'] = []
        
        # 压力场缓存（corner point grid用）
        self.cache['pressure_field_actor'] = None
        self.cache['pressure_scalar_bar'] = None
    
    def render_mode3_smooth_pressure(self, sim_data):
        """渲染平滑压力场 - 使用C++插值数据"""
        renderer = self.renderer
        
        # 优先使用C++插值数据
        if sim_data.interpolated_pressure:
            field_data = sim_data.interpolated_pressure
        else:
            field_data = sim_data.pressure_field
        
        if not field_data:
            return
        
        # 计算数据哈希
        data_hash = hash(str(len(field_data)) + str(field_data[0]) + str(field_data[-1]))
        
        # 使用缓存
        if self.cache['data_hash'] == data_hash and self.cache['pressure_actor'] is not None:
            renderer.AddActor(self.cache['pressure_actor'])
            if self.cache['scalar_bar']:
                renderer.AddViewProp(self.cache['scalar_bar'])
            self.setup_camera(sim_data)
            self.vtk_widget.iren.Render()
            return
        
        renderer.RemoveAllViewProps()
        
        values = [p[3] for p in field_data]
        min_p, max_p = min(values), max(values)
        
        # 获取网格边界
        lx = sim_data.grid_info['Lx']
        ly = sim_data.grid_info['Ly']
        lz = sim_data.grid_info['Lz']
        
        # 使用C++插值数据时，网格尺寸固定为50x25x10
        if sim_data.interpolated_pressure:
            nx, ny, nz = 50, 25, 10
        else:
            nx, ny, nz = 50, 25, 10
        
        grid = vtk.vtkStructuredGrid()
        grid.SetDimensions(nx, ny, nz)
        
        points = vtk.vtkPoints()
        pressure_arr = vtk.vtkDoubleArray()
        pressure_arr.SetName("Pressure")
        
        # 使用C++插值数据 - 直接填充
        if sim_data.interpolated_pressure:
            for x, y, z, p in field_data:
                points.InsertNextPoint(x, y, z)
                pressure_arr.InsertNextValue(p)
        else:
            # === Python插值代码（已弃用，保留作为回退） ===
            # sorted_field = sorted(sim_data.pressure_field, key=lambda p: (p[0], p[1], p[2]))
            # min_x, max_x = 0.0, lx
            # min_y, max_y = 0.0, ly
            # min_z, max_z = 0.0, lz
            # 
            # for k in range(nz):
            #     z = min_z + (max_z - min_z) * k / (nz - 1) if nz > 1 else min_z
            #     for j in range(ny):
            #         y = min_y + (max_y - min_y) * j / (ny - 1) if ny > 1 else min_y
            #         for i in range(nx):
            #             x = min_x + (max_x - min_x) * i / (nx - 1) if nx > 1 else min_x
            #             points.InsertNextPoint(x, y, z)
            #             
            #             # 最近邻插值
            #             min_dist = float('inf')
            #             nearest_p = min_p
            #             for fx, fy, fz, fp in sorted_field:
            #                 dist = abs(x-fx) + abs(y-fy) + abs(z-fz)
            #                 if dist < min_dist:
            #                     min_dist = dist
            #                     nearest_p = fp
            #                     if dist < 1e-6:
            #                         break
            #             
            #             pressure_arr.InsertNextValue(nearest_p)
            # === Python插值代码结束 ===
            
            # 回退：直接使用原始压力点
            for x, y, z, p in field_data:
                points.InsertNextPoint(x, y, z)
                pressure_arr.InsertNextValue(p)
        
        grid.SetPoints(points)
        grid.GetPointData().SetScalars(pressure_arr)
        
        # 提取表面
        geom_filter = vtk.vtkGeometryFilter()
        geom_filter.SetInputData(grid)
        geom_filter.Update()
        
        # 颜色映射
        lut = vtk.vtkLookupTable()
        lut.SetTableRange(min_p, max_p)
        lut.SetHueRange(0.667, 0.0)
        lut.SetSaturationRange(1.0, 1.0)
        lut.SetValueRange(1.0, 1.0)
        lut.SetNumberOfTableValues(256)
        lut.Build()
        
        # 颜色条
        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetLookupTable(lut)
        scalar_bar.SetTitle("Pressure (MPa)")
        scalar_bar.SetNumberOfLabels(6)
        scalar_bar.SetWidth(0.08)
        scalar_bar.SetHeight(0.4)
        scalar_bar.GetLabelTextProperty().SetColor(1, 1, 1)
        scalar_bar.GetTitleTextProperty().SetColor(1, 1, 1)
        scalar_bar.SetPosition(0.88, 0.25)
        renderer.AddViewProp(scalar_bar)
        
        # Actor
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputConnection(geom_filter.GetOutputPort())
        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(min_p, max_p)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(1.0)
        actor.GetProperty().SetEdgeVisibility(0)
        
        renderer.AddActor(actor)
        
        # 保存缓存
        self.cache['data_hash'] = data_hash
        self.cache['pressure_actor'] = actor
        self.cache['scalar_bar'] = scalar_bar
        
        self.setup_camera(sim_data)
        self.vtk_widget.iren.Render()
    
    def render_fractures(self, sim_data):
        """渲染裂缝"""
        renderer = self.renderer
        
        # 检查缓存
        if self.cache['fracture_actors']:
            for actor in self.cache['fracture_actors']:
                renderer.AddActor(actor)
            return
        
        if not sim_data.fractures:
            return
        
        # black_oil模型不使用Z轴夸张，保持原始坐标
        # 创建裂缝面
        points = vtk.vtkPoints()
        polys = vtk.vtkCellArray()
        
        for fracture in sim_data.fractures:
            frac_points = fracture['points']
            n_points = len(frac_points)
            
            polygon = vtk.vtkPolygon()
            polygon.GetPointIds().SetNumberOfIds(n_points)
            
            base_idx = points.GetNumberOfPoints()
            for i, (x, y, z) in enumerate(frac_points):
                points.InsertNextPoint(x, y, z)  # 不使用Z轴夸张
                polygon.GetPointIds().SetId(i, base_idx + i)
            
            polys.InsertNextCell(polygon)
        
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetPolys(polys)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 0.5, 0.0)
        actor.GetProperty().SetOpacity(0.6)
        actor.GetProperty().SetEdgeVisibility(1)
        actor.GetProperty().SetEdgeColor(0.8, 0.4, 0.0)
        actor.GetProperty().SetLineWidth(1.0)
        
        renderer.AddActor(actor)
        self.cache['fracture_actors'].append(actor)
        
        # 裂缝边界线
        line_points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        
        for fracture in sim_data.fractures:
            frac_points = fracture['points']
            n_points = len(frac_points)
            
            for i in range(n_points):
                x0, y0, z0 = frac_points[i]
                x1, y1, z1 = frac_points[(i + 1) % n_points]
                
                p1 = line_points.InsertNextPoint(x0, y0, z0)  # 不使用Z轴夸张
                p2 = line_points.InsertNextPoint(x1, y1, z1)  # 不使用Z轴夸张
                
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
    
    def create_grid_lines(self, sim_data):
        """创建网格线 - 使用LGR真实网格数据"""
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        
        # 如果有真实的LGR网格线数据，使用它
        if sim_data.grid_lines:
            for line_data in sim_data.grid_lines:
                # line_data: (x1, y1, z1, x2, y2, z2)
                p1 = points.InsertNextPoint(line_data[0], line_data[1], line_data[2])
                p2 = points.InsertNextPoint(line_data[3], line_data[4], line_data[5])
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, p1)
                line.GetPointIds().SetId(1, p2)
                lines.InsertNextCell(line)
        else:
            # 回退到粗网格显示
            lx = sim_data.grid_info['Lx']
            ly = sim_data.grid_info['Ly']
            lz = sim_data.grid_info['Lz']
            nx = sim_data.grid_info['nx']
            ny = sim_data.grid_info['ny']
            nz = sim_data.grid_info['nz']
            
            # X方向线
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
            
            # Y方向线
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
            
            # Z方向线
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
        actor.GetProperty().SetColor(0.8, 0.8, 0.8)
        actor.GetProperty().SetLineWidth(1.0)
        actor.GetProperty().SetOpacity(0.8)
        
        self.cache['grid_lines_actor'] = actor
        return actor
    
    def toggle_grid_lines(self, show):
        """切换网格线显示"""
        if show:
            if self.cache['grid_lines_actor'] is None:
                return
            self.renderer.AddActor(self.cache['grid_lines_actor'])
        else:
            if self.cache['grid_lines_actor']:
                self.renderer.RemoveActor(self.cache['grid_lines_actor'])
        self.vtk_widget.iren.Render()
    
    def toggle_fractures(self, show):
        """切换裂缝显示"""
        if show:
            if self.cache['pressure_actor']:
                self.cache['pressure_actor'].GetProperty().SetOpacity(0.3)
            for actor in self.cache['fracture_actors']:
                self.renderer.AddActor(actor)
        else:
            if self.cache['pressure_actor']:
                self.cache['pressure_actor'].GetProperty().SetOpacity(1.0)
            for actor in self.cache['fracture_actors']:
                self.renderer.RemoveActor(actor)
        self.vtk_widget.iren.Render()
    
    def setup_camera(self, sim_data):
        """设置相机 - 与原文件一致"""
        camera = self.renderer.GetActiveCamera()
        
        lx = sim_data.grid_info['Lx']
        ly = sim_data.grid_info['Ly']
        lz = sim_data.grid_info['Lz']
        
        cx, cy, cz = lx / 2.0, ly / 2.0, lz / 2.0
        max_dim = max(lx, ly, lz)
        
        dist = max_dim * 2.5
        
        camera.SetPosition(cx + dist, cy + dist, cz + dist)
        camera.SetFocalPoint(cx, cy, cz)
        camera.SetViewUp(0, 0, 1)
        
        self.renderer.ResetCamera()
        camera.Zoom(1.1)
    
    def render_fracture_only(self, sim_data):
        """仅渲染裂缝"""
        self.renderer.RemoveAllViewProps()
        if sim_data.fractures:
            self.render_fractures(sim_data)
        self.setup_camera(sim_data)
        self.vtk_widget.iren.Render()
    
    def render_corner_point_grid(self, sim_data):
        """渲染角点网格几何模型
        
        使用vtkUnstructuredGrid渲染每个六面体单元，
        显示角点网格的几何结构（灰色，无压力场）
        对Z轴进行垂直夸张以显示地形变化
        """
        renderer = self.renderer
        
        if not sim_data.corner_point_grid or not sim_data.corner_point_grid.cells:
            return
        
        cpg = sim_data.corner_point_grid
        
        # 计算Z轴夸张系数（先算出来用于缓存判断）
        all_x = [c[0] for cell in cpg.cells for c in cell.corners]
        all_y = [c[1] for cell in cpg.cells for c in cell.corners]
        all_z = [c[2] for cell in cpg.cells for c in cell.corners]
        dx = max(all_x) - min(all_x)
        dy = max(all_y) - min(all_y)
        dz = max(all_z) - min(all_z)
        z_min = min(all_z)
        z_exaggeration = max(dx, dy) / dz * 0.3 if dz > 0 else 1.0
        z_exaggeration = min(z_exaggeration, 50)
        
        data_hash = hash(str(len(cpg.cells)) + str(cpg.cells[0].corners[0]) + str(z_exaggeration) + str(z_min) + "v2" if cpg.cells else 0)
        
        if self.cache.get('corner_grid_hash') == data_hash and self.cache.get('corner_actor') is not None:
            renderer.AddActor(self.cache['corner_actor'])
            renderer.AddActor(self.cache['corner_surface_actor'])
            self.setup_camera_for_corner_grid(cpg)
            self.vtk_widget.iren.Render()
            return
        
        # 只移除网格相关的actor，保留其他actor（如压力场）
        if self.cache.get('corner_actor'):
            renderer.RemoveActor(self.cache['corner_actor'])
        if self.cache.get('corner_surface_actor'):
            renderer.RemoveActor(self.cache['corner_surface_actor'])
        
        print(f"Z exaggeration: {z_exaggeration:.1f}x")
        print(f"Z origin for exaggeration: {z_min}")
        
        points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()
        
        # 收集所有网格的Z坐标用于调试
        all_grid_z_exag = []
        
        for cell in cpg.cells:
            point_ids = []
            for corner in cell.corners:
                # 应用Z轴夸张：先减去原点，再乘以系数
                z_exag = (corner[2] - z_min) * z_exaggeration + z_min
                pid = points.InsertNextPoint(corner[0], corner[1], z_exag)
                point_ids.append(pid)
                all_grid_z_exag.append(z_exag)
            
            # VTK Hexahedron vertex order: 0,1,2,3,4,5,6,7
            cells.InsertNextCell(8)
            for pid in point_ids:
                cells.InsertCellPoint(pid)
        
        # 打印网格实际渲染的Z范围
        print(f"网格实际渲染Z范围: {min(all_grid_z_exag):.3f} 到 {max(all_grid_z_exag):.3f}")
        
        grid = vtk.vtkUnstructuredGrid()
        grid.SetPoints(points)
        grid.SetCells(vtk.VTK_HEXAHEDRON, cells)
        
        # 创建线框显示 - 提取所有边
        extract_edges = vtk.vtkExtractEdges()
        extract_edges.SetInputData(grid)
        extract_edges.Update()
        
        # 使用白色线框显示
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputConnection(extract_edges.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 1.0, 1.0)  # 白色线框
        actor.GetProperty().SetLineWidth(1.0)
        
        renderer.AddActor(actor)
        
        # 同时添加半透明表面，增加立体感
        geom_filter = vtk.vtkGeometryFilter()
        geom_filter.SetInputData(grid)
        geom_filter.Update()
        
        surface_mapper = vtk.vtkDataSetMapper()
        surface_mapper.SetInputConnection(geom_filter.GetOutputPort())
        
        surface_actor = vtk.vtkActor()
        surface_actor.SetMapper(surface_mapper)
        surface_actor.GetProperty().SetColor(0.5, 0.5, 0.5)  # 中灰色表面
        surface_actor.GetProperty().SetOpacity(0.15)  # 很透明
        surface_actor.GetProperty().SetEdgeVisibility(0)
        
        renderer.AddActor(surface_actor)
        
        self.cache['corner_grid_hash'] = data_hash
        self.cache['corner_actor'] = actor
        self.cache['corner_surface_actor'] = surface_actor
        self.cache['z_exaggeration'] = z_exaggeration  # 保存Z轴夸张系数
        self.cache['z_min_for_exaggeration'] = z_min  # 保存Z轴原点用于夸张
        
        self.setup_camera_for_corner_grid(cpg)
        self.vtk_widget.iren.Render()
    
    def setup_camera_for_corner_grid(self, cpg):
        """为角点网格设置相机 - 优化视角以显示Z方向变化"""
        camera = self.renderer.GetActiveCamera()
        
        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')
        
        for cell in cpg.cells:
            for corner in cell.corners:
                min_x = min(min_x, corner[0])
                max_x = max(max_x, corner[0])
                min_y = min(min_y, corner[1])
                max_y = max(max_y, corner[1])
                min_z = min(min_z, corner[2])
                max_z = max(max_z, corner[2])
        
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        cz = (min_z + max_z) / 2.0
        
        dx = max_x - min_x
        dy = max_y - min_y
        dz = max_z - min_z
        
        # 计算Z方向相对于XY的比例，如果Z很小，使用侧视图
        z_ratio = dz / max(dx, dy) if max(dx, dy) > 0 else 1.0
        
        if z_ratio < 0.1:
            # Z方向很小，使用倾斜侧视图（从侧面看，带一点角度）
            dist = max(dx, dy) * 1.8
            # 从Y负方向看，同时向X方向偏移一点，抬高Z
            camera.SetPosition(cx + dx * 0.3, cy - dist, cz + dz * 8)
            camera.SetFocalPoint(cx, cy, cz)
            camera.SetViewUp(0, 0, 1)
        else:
            # 正常视角，稍微倾斜
            max_dim = max(dx, dy, dz)
            dist = max_dim * 2.5
            camera.SetPosition(cx + dist * 0.8, cy + dist * 0.6, cz + dist * 0.4)
            camera.SetFocalPoint(cx, cy, cz)
            camera.SetViewUp(0, 0, 1)
        
        self.renderer.ResetCamera()
        camera.Zoom(1.0)

    def render_corner_fractures(self, sim_data):
        """渲染裂缝 - 从CSV加载的四边形面 - 用于Corner Point Grid，应用Z轴夸张"""
        renderer = self.renderer
        
        if not sim_data.fractures:
            return
        
        z_exaggeration = self.cache.get('z_exaggeration', 1.0)
        z_min = self.cache.get('z_min_for_exaggeration', 0.0)
        
        # 清除之前的裂缝actor
        if 'fracture_actors' in self.cache:
            for actor in self.cache['fracture_actors']:
                renderer.RemoveActor(actor)
        
        self.cache['fracture_actors'] = []
        
        # 获取网格边界用于调试
        grid_min_x = grid_max_x = grid_min_y = grid_max_y = grid_min_z = grid_max_z = 0.0
        if sim_data.corner_point_grid and sim_data.corner_point_grid.cells:
            all_corners = [c for cell in sim_data.corner_point_grid.cells for c in cell.corners]
            grid_min_x = min(c[0] for c in all_corners)
            grid_max_x = max(c[0] for c in all_corners)
            grid_min_y = min(c[1] for c in all_corners)
            grid_max_y = max(c[1] for c in all_corners)
            grid_min_z = min(c[2] for c in all_corners)
            grid_max_z = max(c[2] for c in all_corners)
        
        # 计算夸张后的网格Z边界
        grid_min_z_exag = (grid_min_z - z_min) * z_exaggeration + z_min
        grid_max_z_exag = (grid_max_z - z_min) * z_exaggeration + z_min
        
        for frac in sim_data.fractures:
            points = vtk.vtkPoints()
            p0 = list(frac['points'][0])
            p1 = list(frac['points'][1])
            p2 = list(frac['points'][2])
            p3 = list(frac['points'][3])
            
            # 查找裂缝中心所在的网格单元
            center_x_frac = (p0[0] + p1[0] + p2[0] + p3[0]) / 4
            center_y_frac = (p0[1] + p1[1] + p2[1] + p3[1]) / 4
            center_z_frac = (p0[2] + p1[2] + p2[2] + p3[2]) / 4
            
            found_cell = None
            for cell in sim_data.corner_point_grid.cells:
                cell_min_x = min(c[0] for c in cell.corners)
                cell_max_x = max(c[0] for c in cell.corners)
                cell_min_y = min(c[1] for c in cell.corners)
                cell_max_y = max(c[1] for c in cell.corners)
                cell_min_z = min(c[2] for c in cell.corners)
                cell_max_z = max(c[2] for c in cell.corners)
                
                if (cell_min_x <= center_x_frac <= cell_max_x and
                    cell_min_y <= center_y_frac <= cell_max_y and
                    cell_min_z <= center_z_frac <= cell_max_z):
                    found_cell = cell
                    break
            
            # 如果找到网格单元，裁剪裂缝顶点到网格单元边界内
            if found_cell:
                cell_min_x = min(c[0] for c in found_cell.corners)
                cell_max_x = max(c[0] for c in found_cell.corners)
                cell_min_y = min(c[1] for c in found_cell.corners)
                cell_max_y = max(c[1] for c in found_cell.corners)
                cell_min_z = min(c[2] for c in found_cell.corners)
                cell_max_z = max(c[2] for c in found_cell.corners)
                
                # 裁剪每个顶点
                for pt in [p0, p1, p2, p3]:
                    pt[0] = max(cell_min_x, min(cell_max_x, pt[0]))
                    pt[1] = max(cell_min_y, min(cell_max_y, pt[1]))
                    pt[2] = max(cell_min_z, min(cell_max_z, pt[2]))
            
            # 计算夸张后的裂缝顶点
            p0_exag_z = (p0[2] - z_min) * z_exaggeration + z_min
            p1_exag_z = (p1[2] - z_min) * z_exaggeration + z_min
            p2_exag_z = (p2[2] - z_min) * z_exaggeration + z_min
            p3_exag_z = (p3[2] - z_min) * z_exaggeration + z_min
            
            # 针对裂缝65和96的详细调试
            if frac['id'] in [65, 96]:
                print(f"\n=== 裂缝 {frac['id']} 详细调试 ===")
                print(f"网格原始边界:")
                print(f"  X: [{grid_min_x:.3f}, {grid_max_x:.3f}]")
                print(f"  Y: [{grid_min_y:.3f}, {grid_max_y:.3f}]")
                print(f"  Z: [{grid_min_z:.3f}, {grid_max_z:.3f}]")
                print(f"网格夸张后Z边界: [{grid_min_z_exag:.3f}, {grid_max_z_exag:.3f}]")
                print(f"\n裂缝原始顶点:")
                print(f"  点0: ({p0[0]:.3f}, {p0[1]:.3f}, {p0[2]:.3f})")
                print(f"  点1: ({p1[0]:.3f}, {p1[1]:.3f}, {p1[2]:.3f})")
                print(f"  点2: ({p2[0]:.3f}, {p2[1]:.3f}, {p2[2]:.3f})")
                print(f"  点3: ({p3[0]:.3f}, {p3[1]:.3f}, {p3[2]:.3f})")
                print(f"\n裂缝夸张后顶点:")
                print(f"  点0: ({p0[0]:.3f}, {p0[1]:.3f}, {p0_exag_z:.3f})")
                print(f"  点1: ({p1[0]:.3f}, {p1[1]:.3f}, {p1_exag_z:.3f})")
                print(f"  点2: ({p2[0]:.3f}, {p2[1]:.3f}, {p2_exag_z:.3f})")
                print(f"  点3: ({p3[0]:.3f}, {p3[1]:.3f}, {p3_exag_z:.3f})")
                
                # 检查每个点是否超出边界
                for i, (pt, z_exag) in enumerate([(p0, p0_exag_z), (p1, p1_exag_z), (p2, p2_exag_z), (p3, p3_exag_z)]):
                    x, y, z = pt
                    issues = []
                    if x < grid_min_x or x > grid_max_x:
                        issues.append(f"X={x:.3f} 超出 [{grid_min_x:.3f}, {grid_max_x:.3f}]")
                    if y < grid_min_y or y > grid_max_y:
                        issues.append(f"Y={y:.3f} 超出 [{grid_min_y:.3f}, {grid_max_y:.3f}]")
                    if z < grid_min_z or z > grid_max_z:
                        issues.append(f"Z={z:.3f} 超出 [{grid_min_z:.3f}, {grid_max_z:.3f}]")
                    if z_exag < grid_min_z_exag or z_exag > grid_max_z_exag:
                        issues.append(f"Z夸张后={z_exag:.3f} 超出 [{grid_min_z_exag:.3f}, {grid_max_z_exag:.3f}]")
                    if issues:
                        print(f"  点{i} 问题: {', '.join(issues)}")
                
                # 查找裂缝中心所在的网格单元
                center_x_frac = (p0[0] + p1[0] + p2[0] + p3[0]) / 4
                center_y_frac = (p0[1] + p1[1] + p2[1] + p3[1]) / 4
                center_z_frac = (p0[2] + p1[2] + p2[2] + p3[2]) / 4
                print(f"\n裂缝中心: ({center_x_frac:.3f}, {center_y_frac:.3f}, {center_z_frac:.3f})")
                
                # 查找包含裂缝中心的网格单元
                found_cell = None
                for cell_idx, cell in enumerate(sim_data.corner_point_grid.cells):
                    cell_min_x = min(c[0] for c in cell.corners)
                    cell_max_x = max(c[0] for c in cell.corners)
                    cell_min_y = min(c[1] for c in cell.corners)
                    cell_max_y = max(c[1] for c in cell.corners)
                    cell_min_z = min(c[2] for c in cell.corners)
                    cell_max_z = max(c[2] for c in cell.corners)
                    
                    if (cell_min_x <= center_x_frac <= cell_max_x and
                        cell_min_y <= center_y_frac <= cell_max_y and
                        cell_min_z <= center_z_frac <= cell_max_z):
                        found_cell = cell
                        print(f"\n找到包含裂缝的网格单元 (索引 {cell_idx}):")
                        print(f"  网格单元原始Z范围: [{cell_min_z:.3f}, {cell_max_z:.3f}]")
                        cell_min_z_exag_cell = (cell_min_z - z_min) * z_exaggeration + z_min
                        cell_max_z_exag_cell = (cell_max_z - z_min) * z_exaggeration + z_min
                        print(f"  网格单元夸张后Z范围: [{cell_min_z_exag_cell:.3f}, {cell_max_z_exag_cell:.3f}]")
                        print(f"  网格单元8个顶点:")
                        for j, corner in enumerate(cell.corners):
                            z_exag_corner = (corner[2] - z_min) * z_exaggeration + z_min
                            print(f"    顶点{j}: ({corner[0]:.3f}, {corner[1]:.3f}, {corner[2]:.3f}) -> 夸张后Z: {z_exag_corner:.3f}")
                        break
                
                if not found_cell:
                    print("\n警告: 未找到包含裂缝中心的网格单元!")
                    # 查找最近的网格单元
                    min_dist = float('inf')
                    nearest_cell_idx = -1
                    for cell_idx, cell in enumerate(sim_data.corner_point_grid.cells):
                        cell_cx = sum(c[0] for c in cell.corners) / 8
                        cell_cy = sum(c[1] for c in cell.corners) / 8
                        cell_cz = sum(c[2] for c in cell.corners) / 8
                        dist = ((cell_cx - center_x_frac)**2 + (cell_cy - center_y_frac)**2 + (cell_cz - center_z_frac)**2)**0.5
                        if dist < min_dist:
                            min_dist = dist
                            nearest_cell_idx = cell_idx
                    if nearest_cell_idx >= 0:
                        cell = sim_data.corner_point_grid.cells[nearest_cell_idx]
                        print(f"\n最近的网格单元 (索引 {nearest_cell_idx}, 距离 {min_dist:.3f}):")
                        cell_min_z = min(c[2] for c in cell.corners)
                        cell_max_z = max(c[2] for c in cell.corners)
                        print(f"  网格单元原始Z范围: [{cell_min_z:.3f}, {cell_max_z:.3f}]")
                        print(f"  裂缝Z: {center_z_frac:.3f}")
                
                print("="*50)
            
            points.InsertNextPoint(p0[0], p0[1], p0_exag_z)
            points.InsertNextPoint(p1[0], p1[1], p1_exag_z)
            points.InsertNextPoint(p2[0], p2[1], p2_exag_z)
            points.InsertNextPoint(p3[0], p3[1], p3_exag_z)
            
            quad = vtk.vtkQuad()
            for i in range(4):
                quad.GetPointIds().SetId(i, i)
            
            cells = vtk.vtkCellArray()
            cells.InsertNextCell(quad)
            
            grid = vtk.vtkUnstructuredGrid()
            grid.SetPoints(points)
            grid.SetCells(vtk.VTK_QUAD, cells)
            
            mapper = vtk.vtkDataSetMapper()
            mapper.SetInputData(grid)
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # 红色裂缝
            actor.GetProperty().SetOpacity(0.8)
            actor.GetProperty().SetEdgeVisibility(1)
            actor.GetProperty().SetEdgeColor(0.8, 0.0, 0.0)
            actor.GetProperty().SetLineWidth(2.0)
            
            renderer.AddActor(actor)
            self.cache['fracture_actors'].append(actor)
            
            # 添加裂缝ID标签
            center_x = (p0[0] + p1[0] + p2[0] + p3[0]) / 4
            center_y = (p0[1] + p1[1] + p2[1] + p3[1]) / 4
            center_z = (p0_exag_z + p1_exag_z + p2_exag_z + p3_exag_z) / 4
            
            text_source = vtk.vtkVectorText()
            text_source.SetText(str(frac['id']))
            
            text_mapper = vtk.vtkPolyDataMapper()
            text_mapper.SetInputConnection(text_source.GetOutputPort())
            
            text_actor = vtk.vtkFollower()
            text_actor.SetMapper(text_mapper)
            text_actor.SetPosition(center_x, center_y, center_z)
            text_actor.SetScale(5.0)
            text_actor.GetProperty().SetColor(0.0, 1.0, 0.0)  # 绿色文字
            
            camera = self.renderer.GetActiveCamera()
            text_actor.SetCamera(camera)
            
            renderer.AddActor(text_actor)
            self.cache['fracture_actors'].append(text_actor)
        
        self.vtk_widget.iren.Render()

    def hide_fractures(self):
        """隐藏裂缝"""
        if 'fracture_actors' in self.cache:
            for actor in self.cache['fracture_actors']:
                self.renderer.RemoveActor(actor)
            self.vtk_widget.iren.Render()

    def render_wells(self, sim_data):
        """渲染井 - 显示为球体"""
        renderer = self.renderer
        
        if not sim_data.wells:
            return
        
        # 清除之前的井actor
        if 'well_actors' in self.cache:
            for actor in self.cache['well_actors']:
                renderer.RemoveActor(actor)
        
        self.cache['well_actors'] = []
        
        for well in sim_data.wells:
            sphere = vtk.vtkSphereSource()
            # 直接使用原始坐标，不使用Z轴夸张
            sphere.SetCenter(well['x'], well['y'], well['z'])
            sphere.SetRadius(5.0)  # 井的显示半径
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            # 根据井类型设置颜色
            if well['type'] == 'Fracture':
                actor.GetProperty().SetColor(0.0, 1.0, 0.0)  # 绿色 - 裂缝井
            else:
                actor.GetProperty().SetColor(1.0, 1.0, 0.0)  # 黄色 - 基质井
            
            renderer.AddActor(actor)
            self.cache['well_actors'].append(actor)
        
        self.vtk_widget.iren.Render()

    def hide_wells(self):
        """隐藏井"""
        if 'well_actors' in self.cache:
            for actor in self.cache['well_actors']:
                self.renderer.RemoveActor(actor)
            self.vtk_widget.iren.Render()

    def render_corner_wells(self, sim_data):
        """渲染角点网格的井 - 使用Z轴夸张"""
        renderer = self.renderer
        
        if not sim_data.wells:
            return
        
        # 获取Z轴夸张系数
        z_exaggeration = self.cache.get('z_exaggeration', 1.0)
        z_min = self.cache.get('z_min_for_exaggeration', 0.0)
        
        # 清除之前的井actor
        if 'well_actors' in self.cache:
            for actor in self.cache['well_actors']:
                renderer.RemoveActor(actor)
        
        self.cache['well_actors'] = []
        
        for well in sim_data.wells:
            sphere = vtk.vtkSphereSource()
            # 应用Z轴夸张：先减去原点，再乘以系数
            z_exag = (well['z'] - z_min) * z_exaggeration + z_min
            sphere.SetCenter(well['x'], well['y'], z_exag)
            sphere.SetRadius(5.0)  # 井的显示半径
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            # 根据井类型设置颜色
            if well['type'] == 'Fracture':
                actor.GetProperty().SetColor(0.0, 1.0, 0.0)  # 绿色 - 裂缝井
            else:
                actor.GetProperty().SetColor(1.0, 1.0, 0.0)  # 黄色 - 基质井
            
            renderer.AddActor(actor)
            self.cache['well_actors'].append(actor)
        
        self.vtk_widget.iren.Render()

    def render_pressure_field(self, sim_data):
        """渲染压力场 - black_oil模型使用"""
        renderer = self.renderer
        
        if not sim_data.pressure_field:
            print("No pressure field data")
            return
        
        print(f"Rendering pressure field with {len(sim_data.pressure_field)} cells")
        
        # black_oil不使用Z轴夸张
        z_exaggeration = 1.0
        
        if 'pressure_field_actor' in self.cache and self.cache['pressure_field_actor']:
            renderer.RemoveActor(self.cache['pressure_field_actor'])
            self.cache['pressure_field_actor'] = None
        
        # black_oil使用点云方式渲染
        self._render_pressure_field_points(sim_data, z_exaggeration)

    def render_corner_pressure_field(self, sim_data):
        """渲染角点网格压力场 - corner point grid模型专用"""
        renderer = self.renderer
        
        if not hasattr(sim_data, 'cell_geometry_with_pressure') or sim_data.cell_geometry_with_pressure is None:
            print("No cell geometry data for corner point grid pressure field")
            return
        
        cell_data = sim_data.cell_geometry_with_pressure
        n_cells = cell_data.shape[0]
        print(f"Rendering corner point grid pressure field with {n_cells} cells")
        
        # 获取Z轴夸张参数
        z_exaggeration = self.cache.get('z_exaggeration', 1.0)
        z_min = self.cache.get('z_min_for_exaggeration', 0.0)
        print(f"Using z_exaggeration: {z_exaggeration}")
        print(f"Using z_min: {z_min}")
        
        # 清除之前的压力场actor
        if 'pressure_field_actor' in self.cache and self.cache['pressure_field_actor']:
            renderer.RemoveActor(self.cache['pressure_field_actor'])
            self.cache['pressure_field_actor'] = None
        if 'pressure_scalar_bar' in self.cache and self.cache['pressure_scalar_bar']:
            renderer.RemoveViewProp(self.cache['pressure_scalar_bar'])
            self.cache['pressure_scalar_bar'] = None
        
        try:
            # 创建非结构化网格
            grid = vtk.vtkUnstructuredGrid()
            points = vtk.vtkPoints()
            
            pressure_arr = vtk.vtkDoubleArray()
            pressure_arr.SetName("Pressure")
            
            pressure_values = []
            point_id = 0
            
            # VTK六面体顶点顺序: 0,1,2,3,4,5,6,7
            for i in range(n_cells):
                p = float(cell_data[i, 28])
                pressure_values.append(p)
                
                # 插入8个角点 - 使用Z轴夸张
                for j in range(8):
                    x = float(cell_data[i, 4 + j*3 + 0])
                    y = float(cell_data[i, 4 + j*3 + 1])
                    z = float(cell_data[i, 4 + j*3 + 2])
                    z_exag = (z - z_min) * z_exaggeration + z_min
                    points.InsertNextPoint(x, y, z_exag)
                
                # 创建六面体单元
                hexa = vtk.vtkHexahedron()
                for j in range(8):
                    hexa.GetPointIds().SetId(j, point_id + j)
                
                grid.InsertNextCell(hexa.GetCellType(), hexa.GetPointIds())
                pressure_arr.InsertNextValue(p)
                
                point_id += 8
            
            grid.SetPoints(points)
            grid.GetCellData().SetScalars(pressure_arr)
            
            print(f"Created {points.GetNumberOfPoints()} points, {grid.GetNumberOfCells()} cells")
            print(f"Pressure range: {min(pressure_values):.2f} - {max(pressure_values):.2f}")
            
            # 将CellData转换为PointData用于平滑着色
            cell_to_point = vtk.vtkCellDataToPointData()
            cell_to_point.SetInputData(grid)
            cell_to_point.Update()
            
            # 提取外表面
            geometry_filter = vtk.vtkGeometryFilter()
            geometry_filter.SetInputData(cell_to_point.GetOutput())
            geometry_filter.Update()
            
            surface = geometry_filter.GetOutput()
            print(f"Surface has {surface.GetNumberOfCells()} cells")
            
            # 创建mapper和actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(surface)
            
            # 显式设置标量数据
            mapper.SetScalarModeToUsePointData()
            mapper.SelectColorArray("Pressure")
            mapper.SetScalarVisibility(True)
            
            pressure_min = min(pressure_values) if pressure_values else 700.0
            pressure_max = max(pressure_values) if pressure_values else 800.0
            mapper.SetScalarRange(pressure_min, pressure_max)
            print(f"Mapper scalar range: {pressure_min} - {pressure_max}")
            
            lut = vtk.vtkLookupTable()
            lut.SetHueRange(0.667, 0.0)
            lut.SetSaturationRange(1.0, 1.0)
            lut.SetValueRange(1.0, 1.0)
            lut.SetNumberOfColors(256)
            lut.Build()
            mapper.SetLookupTable(lut)
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetOpacity(0.9)
            
            renderer.AddActor(actor)
            self.cache['pressure_field_actor'] = actor
            print(f"Added corner pressure field surface actor")
            
            self._add_pressure_scalar_bar(renderer, lut, pressure_min, pressure_max)
            
            self.vtk_widget.iren.Render()
            print("Corner pressure field render complete")
        except Exception as e:
            print(f"ERROR in render_corner_pressure_field: {e}")
            import traceback
            traceback.print_exc()

    def _render_pressure_field_cells(self, sim_data, z_exaggeration):
        """使用C++导出的单元格几何数据渲染压力场 - 提取外表面显示"""
        try:
            renderer = self.renderer
            
            cell_data = sim_data.cell_geometry_with_pressure
            n_cells = cell_data.shape[0]
            print(f"Rendering {n_cells} cells as surface")
            
            # 使用和网格相同的z_exaggeration（从cache获取）
            use_z_exaggeration = self.cache.get('z_exaggeration', z_exaggeration)
            z_min = self.cache.get('z_min_for_exaggeration', 0.0)
            print(f"Using z_exaggeration: {use_z_exaggeration}")
            print(f"Using z_min: {z_min}")
            
            # 调试：对比前3个单元的Z坐标
            if sim_data.corner_point_grid and sim_data.corner_point_grid.cells:
                print("\n=== 调试：对比Z坐标 ===")
                for i in range(min(3, n_cells, len(sim_data.corner_point_grid.cells))):
                    print(f"\n单元 {i}:")
                    print(f"  网格Z坐标 (Python): {[c[2] for c in sim_data.corner_point_grid.cells[i].corners]}")
                    print(f"  网格夸张后Z: {[(c[2] - z_min) * use_z_exaggeration + z_min for c in sim_data.corner_point_grid.cells[i].corners]}")
                    print(f"  压力场Z坐标 (C++): {[float(cell_data[i, 4 + j*3 + 2]) for j in range(8)]}")
                    print(f"  压力场夸张后Z: {[(float(cell_data[i, 4 + j*3 + 2]) - z_min) * use_z_exaggeration + z_min for j in range(8)]}")
                print("========================\n")
            
            # 创建非结构化网格
            grid = vtk.vtkUnstructuredGrid()
            points = vtk.vtkPoints()
            
            pressure_arr = vtk.vtkDoubleArray()
            pressure_arr.SetName("Pressure")
            
            pressure_values = []
            point_id = 0
            
            # VTK六面体顶点顺序: 0,1,2,3,4,5,6,7
            for i in range(n_cells):
                p = float(cell_data[i, 28])
                pressure_values.append(p)
                
                # 插入8个角点 - 使用和网格相同的z_exaggeration：先减去原点，再乘以系数
                for j in range(8):
                    x = float(cell_data[i, 4 + j*3 + 0])
                    y = float(cell_data[i, 4 + j*3 + 1])
                    z = float(cell_data[i, 4 + j*3 + 2])
                    z_exag = (z - z_min) * use_z_exaggeration + z_min
                    points.InsertNextPoint(x, y, z_exag)
                
                # 创建六面体单元
                hexa = vtk.vtkHexahedron()
                for j in range(8):
                    hexa.GetPointIds().SetId(j, point_id + j)
                
                grid.InsertNextCell(hexa.GetCellType(), hexa.GetPointIds())
                pressure_arr.InsertNextValue(p)
                
                point_id += 8
            
            grid.SetPoints(points)
            grid.GetCellData().SetScalars(pressure_arr)
            
            print(f"Created {points.GetNumberOfPoints()} points, {grid.GetNumberOfCells()} cells")
            print(f"Pressure range: {min(pressure_values):.2f} - {max(pressure_values):.2f}")
            
            # 将CellData转换为PointData用于平滑着色
            cell_to_point = vtk.vtkCellDataToPointData()
            cell_to_point.SetInputData(grid)
            cell_to_point.Update()
            
            # 提取外表面
            geometry_filter = vtk.vtkGeometryFilter()
            geometry_filter.SetInputData(cell_to_point.GetOutput())
            geometry_filter.Update()
            
            surface = geometry_filter.GetOutput()
            print(f"Surface has {surface.GetNumberOfCells()} cells")
            
            # 创建mapper和actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(surface)
            
            # 显式设置标量数据
            mapper.SetScalarModeToUsePointData()
            mapper.SelectColorArray("Pressure")
            mapper.SetScalarVisibility(True)
            
            pressure_min = min(pressure_values) if pressure_values else 700.0
            pressure_max = max(pressure_values) if pressure_values else 800.0
            mapper.SetScalarRange(pressure_min, pressure_max)
            print(f"Mapper scalar range: {pressure_min} - {pressure_max}")
            
            lut = vtk.vtkLookupTable()
            lut.SetHueRange(0.667, 0.0)
            lut.SetSaturationRange(1.0, 1.0)
            lut.SetValueRange(1.0, 1.0)
            lut.SetNumberOfColors(256)
            lut.Build()
            mapper.SetLookupTable(lut)
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetOpacity(0.9)
            
            renderer.AddActor(actor)
            self.cache['pressure_field_actor'] = actor
            print(f"Added pressure field surface actor")
            
            self._add_pressure_scalar_bar(renderer, lut, pressure_min, pressure_max)
            
            self.vtk_widget.iren.Render()
            print("Render complete")
        except Exception as e:
            print(f"ERROR in _render_pressure_field_cells: {e}")
            import traceback
            traceback.print_exc()

    def _render_pressure_field_points(self, sim_data, z_exaggeration):
        """回退方案：使用点云显示压力场"""
        renderer = self.renderer
        
        points = vtk.vtkPoints()
        pressure_arr = vtk.vtkDoubleArray()
        pressure_arr.SetName("Pressure")
        
        pressure_values = []
        for x, y, z, p in sim_data.pressure_field:
            points.InsertNextPoint(x, y, z * z_exaggeration)
            pressure_arr.InsertNextValue(p)
            pressure_values.append(p)
        
        pressure_min = min(pressure_values)
        pressure_max = max(pressure_values)
        print(f"Pressure range: {pressure_min} - {pressure_max}")
        
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.GetPointData().SetScalars(pressure_arr)
        
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(8.0)
        sphere.SetPhiResolution(8)
        sphere.SetThetaResolution(8)
        
        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(polydata)
        glyph.SetSourceConnection(sphere.GetOutputPort())
        glyph.SetScaleModeToDataScalingOff()
        glyph.Update()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        mapper.SetScalarRange(pressure_min, pressure_max)
        
        lut = vtk.vtkLookupTable()
        lut.SetHueRange(0.667, 0.0)
        lut.SetSaturationRange(1.0, 1.0)
        lut.SetValueRange(1.0, 1.0)
        lut.SetNumberOfColors(256)
        lut.Build()
        mapper.SetLookupTable(lut)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        renderer.AddActor(actor)
        self.cache['pressure_field_actor'] = actor
        print(f"Added pressure field actor (point cloud)")
        
        self._add_pressure_scalar_bar(renderer, lut, pressure_min, pressure_max)
        self.vtk_widget.iren.Render()

    def _add_pressure_scalar_bar(self, renderer, lut, pressure_min, pressure_max):
        """添加压力颜色条"""
        if 'pressure_scalar_bar' in self.cache and self.cache['pressure_scalar_bar']:
            renderer.RemoveViewProp(self.cache['pressure_scalar_bar'])
        
        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetLookupTable(lut)
        scalar_bar.SetTitle("Pressure (bar)")
        scalar_bar.SetNumberOfLabels(5)
        scalar_bar.SetPosition(0.82, 0.15)
        scalar_bar.SetWidth(0.12)
        scalar_bar.SetHeight(0.6)
        
        scalar_bar.GetTitleTextProperty().SetColor(1, 1, 1)
        scalar_bar.GetLabelTextProperty().SetColor(1, 1, 1)
        
        renderer.AddActor2D(scalar_bar)
        self.cache['pressure_scalar_bar'] = scalar_bar
        print(f"Added scalar bar")
        
        # 添加颜色条
        if 'pressure_scalar_bar' in self.cache and self.cache['pressure_scalar_bar']:
            renderer.RemoveViewProp(self.cache['pressure_scalar_bar'])
        
        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetLookupTable(lut)
        scalar_bar.SetTitle("Pressure (bar)")
        scalar_bar.SetNumberOfLabels(5)
        scalar_bar.SetPosition(0.82, 0.15)
        scalar_bar.SetWidth(0.12)
        scalar_bar.SetHeight(0.6)
        
        # 设置颜色条文字颜色为白色（适应黑色背景）
        scalar_bar.GetTitleTextProperty().SetColor(1, 1, 1)
        scalar_bar.GetLabelTextProperty().SetColor(1, 1, 1)
        
        renderer.AddActor2D(scalar_bar)
        self.cache['pressure_scalar_bar'] = scalar_bar
        print(f"Added scalar bar")
        
        self.vtk_widget.iren.Render()
        print(f"Render called")

    def hide_pressure_field(self):
        """隐藏压力场"""
        if 'pressure_field_actor' in self.cache and self.cache['pressure_field_actor']:
            self.renderer.RemoveActor(self.cache['pressure_field_actor'])
        if 'pressure_scalar_bar' in self.cache and self.cache['pressure_scalar_bar']:
            self.renderer.RemoveViewProp(self.cache['pressure_scalar_bar'])
        self.vtk_widget.iren.Render()

    def toggle_grid_visibility(self, visible):
        """切换网格显示"""
        if 'corner_actor' in self.cache and self.cache['corner_actor']:
            self.cache['corner_actor'].SetVisibility(visible)
        self.vtk_widget.iren.Render()

    def toggle_fractures_visibility(self, visible):
        """切换裂缝显示"""
        if 'fracture_actors' in self.cache:
            for actor in self.cache['fracture_actors']:
                actor.SetVisibility(visible)
        self.vtk_widget.iren.Render()

    def toggle_wells_visibility(self, visible):
        """切换井显示"""
        if 'well_actors' in self.cache:
            for actor in self.cache['well_actors']:
                actor.SetVisibility(visible)
        self.vtk_widget.iren.Render()

    def toggle_pressure_visibility(self, visible):
        """切换压力场显示"""
        if 'pressure_field_actor' in self.cache and self.cache['pressure_field_actor']:
            self.cache['pressure_field_actor'].SetVisibility(visible)
        if 'pressure_scalar_bar' in self.cache and self.cache['pressure_scalar_bar']:
            self.cache['pressure_scalar_bar'].SetVisibility(visible)
        self.vtk_widget.iren.Render()
