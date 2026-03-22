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
        """清除VTK缓存"""
        self.cache['pressure_actor'] = None
        self.cache['fracture_actors'] = []
        self.cache['scalar_bar'] = None
        self.cache['grid_lines_actor'] = None
        self.cache['data_hash'] = None
    
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
                points.InsertNextPoint(x, y, z)
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
                
                p1 = line_points.InsertNextPoint(x0, y0, z0)
                p2 = line_points.InsertNextPoint(x1, y1, z1)
                
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
        """渲染角点网格压力场
        
        使用vtkUnstructuredGrid渲染每个六面体单元，
        支持非规则地质曲面（角点网格）
        """
        renderer = self.renderer
        
        if not sim_data.corner_point_grid or not sim_data.corner_point_grid.cells:
            return
        
        cpg = sim_data.corner_point_grid
        
        data_hash = hash(str(len(cpg.cells)) + str(cpg.cells[0].corners[0]) if cpg.cells else 0)
        
        if self.cache['data_hash'] == data_hash and self.cache['pressure_actor'] is not None:
            renderer.AddActor(self.cache['pressure_actor'])
            if self.cache['scalar_bar']:
                renderer.AddViewProp(self.cache['scalar_bar'])
            self.setup_camera_for_corner_grid(cpg)
            self.vtk_widget.iren.Render()
            return
        
        renderer.RemoveAllViewProps()
        
        points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()
        pressure_arr = vtk.vtkDoubleArray()
        pressure_arr.SetName("Pressure")
        
        for cell in cpg.cells:
            point_ids = []
            for corner in cell.corners:
                pid = points.InsertNextPoint(corner[0], corner[1], corner[2])
                point_ids.append(pid)
            
            # VTK Hexahedron vertex order: 0,1,2,3,4,5,6,7
            # Our corners: 0,1,2,3 (bottom), 4,5,6,7 (top)
            # VTK expects: bottom face CCW, then top face CCW
            cells.InsertNextCell(8)
            for pid in point_ids:
                cells.InsertCellPoint(pid)
            
            pressure_arr.InsertNextValue(cell.pressure)
        
        grid = vtk.vtkUnstructuredGrid()
        grid.SetPoints(points)
        grid.SetCells(vtk.VTK_HEXAHEDRON, cells)
        grid.GetCellData().SetScalars(pressure_arr)
        
        min_p = cpg.min_pressure
        max_p = cpg.max_pressure
        
        lut = vtk.vtkLookupTable()
        lut.SetTableRange(min_p, max_p)
        lut.SetHueRange(0.667, 0.0)
        lut.SetSaturationRange(1.0, 1.0)
        lut.SetValueRange(1.0, 1.0)
        lut.SetNumberOfTableValues(256)
        lut.Build()
        
        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetLookupTable(lut)
        scalar_bar.SetTitle("Pressure (bar)")
        scalar_bar.SetNumberOfLabels(6)
        scalar_bar.SetWidth(0.08)
        scalar_bar.SetHeight(0.4)
        scalar_bar.GetLabelTextProperty().SetColor(1, 1, 1)
        scalar_bar.GetTitleTextProperty().SetColor(1, 1, 1)
        scalar_bar.SetPosition(0.88, 0.25)
        renderer.AddViewProp(scalar_bar)
        
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputData(grid)
        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(min_p, max_p)
        mapper.SetScalarModeToUseCellData()
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(0.9)
        actor.GetProperty().SetEdgeVisibility(1)
        actor.GetProperty().SetEdgeColor(0.3, 0.3, 0.3)
        actor.GetProperty().SetLineWidth(0.5)
        
        renderer.AddActor(actor)
        
        self.cache['data_hash'] = data_hash
        self.cache['pressure_actor'] = actor
        self.cache['scalar_bar'] = scalar_bar
        
        self.setup_camera_for_corner_grid(cpg)
        self.vtk_widget.iren.Render()
    
    def setup_camera_for_corner_grid(self, cpg):
        """为角点网格设置相机"""
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
        
        max_dim = max(max_x - min_x, max_y - min_y, max_z - min_z)
        dist = max_dim * 2.5
        
        camera.SetPosition(cx + dist, cy + dist, cz + dist)
        camera.SetFocalPoint(cx, cy, cz)
        camera.SetViewUp(0, 0, 1)
        
        self.renderer.ResetCamera()
        camera.Zoom(1.1)
