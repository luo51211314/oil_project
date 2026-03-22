// =============================================================================
// 文件名: edfm_3d_blackoil.cpp
// 描述: 3D 三相(油气水) 黑油模型 EDFM 求解器
// 依赖: Eigen 3.3+
// 编译: g++ -O3 -std=c++17 edfm_3d_blackoil.cpp -o edfm_3d -I /path/to/eigen
// =============================================================================
#include <array>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <random>
#include <tuple>
#include <map>

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/SparseLU>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/AutoDiff>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

using namespace std;
using namespace Eigen;

// =============================================================================
// 1. 常量与基础数据结构
// =============================================================================

const double EPSILON = 1e-8;
const double PI = 3.14159265358979323846;

enum FaceID {
    XM = 0, XP = 1,
    YM = 2, YP = 3,
    ZM = 4, ZP = 5
};

// 3D 点结构
struct Point3 {
    double x = 0.0, y = 0.0, z = 0.0;

    Point3 operator+(const Point3& other) const { return {x + other.x, y + other.y, z + other.z}; }
    Point3 operator-(const Point3& other) const { return {x - other.x, y - other.y, z - other.z}; }
    Point3 operator*(double s) const { return {x * s, y * s, z * s}; }
    Point3 operator/(double s) const { return {x / s, y / s, z / s}; }

    double dot(const Point3& other) const { return x * other.x + y * other.y + z * other.z; }
    Point3 cross(const Point3& other) const {
        return {
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        };
    }
    double norm() const { return std::sqrt(x*x + y*y + z*z); }
};

// 裂缝定义（几何输入）
struct Fracture {
    int id = -1;
    Point3 vertices[4];   // 四个顶点，定义一个四边形面
    double aperture = 0.0;
    double perm = 0.0;

    // 新增：是否为人工裂缝
    bool is_hydraulic = false;
};


// 全局 face 几何对象（阶段A新增）
struct FaceGeom {
    int id = -1;

    // owner / neighbor: 面两侧的 cell
    // 边界面时 neighbor = -1
    int owner = -1;
    int neighbor = -1;

    // 该 face 在 owner / neighbor 中对应的局部 face 槽位
    int local_owner_face = -1;
    int local_neighbor_face = -1;

    // 这里先按四边形面存；后续角点网格阶段仍然可沿用
    std::array<Point3, 4> vertices{};

    Point3 center{0, 0, 0};
    Point3 normal{0, 0, 0};   // 法向朝 owner 外侧
    double area = 0.0;

    Point3 bbox_min{0, 0, 0};
    Point3 bbox_max{0, 0, 0};
};

// 网格单元（阶段A升级版）
// 仍保留 dx,dy,dz / center / vol 等字段，以保证后续旧函数暂时还能编译运行。
// 但从这一阶段开始，真实几何信息应优先来自 corners / face_ids / bbox。
struct Cell {
    int id = -1;             // 全局索引
    int ix = 0, iy = 0, iz = 0;     // 逻辑 IJK 索引

    Point3 center{0, 0, 0};  // 几何中心（阶段A先用8角点平均）
    double dx = 0.0, dy = 0.0, dz = 0.0; // 兼容字段；后续不再作为真几何依据
    double vol = 0.0;

    double phi = 0.0;
    double K[3] = {0.0, 0.0, 0.0};
    double depth = 0.0;

    // 角点网格核心几何：8个角点
    // 约定编号：
    // 0:(xmin,ymin,zmin) 1:(xmax,ymin,zmin) 2:(xmax,ymax,zmin) 3:(xmin,ymax,zmin)
    // 4:(xmin,ymin,zmax) 5:(xmax,ymin,zmax) 6:(xmax,ymax,zmax) 7:(xmin,ymax,zmax)
    std::array<Point3, 8> corners{};

    // 该 cell 的 6 个局部面，对应到全局 faces 向量中的 face id
    std::array<int, 6> face_ids{{-1, -1, -1, -1, -1, -1}};

    // 包围盒，后续做 fracture-cell 候选筛选会用到
    Point3 bbox_min{0, 0, 0};
    Point3 bbox_max{0, 0, 0};
};

// 裂缝段 (EDFM 离散后的最小单元)
struct Segment {
    int id;             // 全局段索引
    int frac_id;        // 所属的大裂缝ID
    int cell_id;        // 所在的基质网格ID
    double area;        // 该段在网格内的截面积
    Point3 center;      // 该段的几何中心
    Point3 normal;      // 法向量
    double aperture;
    double perm;
    double T_mf;        // 基质-裂缝传导率
     // 六个宿主 cell 面上的几何信息
    // 0:x-, 1:x+, 2:y-, 3:y+, 4:z-, 5:z+
    double face_trace_len[6] = {0, 0, 0, 0, 0, 0};  //这个 segment 在宿主 cell 的第 f 个面上的交线长度
    double face_center_dist[6] = {0, 0, 0, 0, 0, 0}; //segment 面积质心到该 face-trace 的距离
    std::vector<Point3> poly;   // 新增
};

// 连接关系 (用于构建 Jacobian)
struct Connection {
    int u; // 单元 u (可以是基质或裂缝段)
    int v; // 单元 v
    double T; // 传导率
    // 类型: 0=Matrix-Matrix, 1=Matrix-Fracture, 2=Fracture-Fracture
    int type; 
};

// 流体物性参数
struct FluidProps {
    // 粘度 (cP)
    double mu_w = 1.0;
    double mu_o = 5.0;
    double mu_g = 0.2;
    
    // 压缩系数 (1/bar)
    double cw = 1e-8;
    double co = 1e-5;
    double cg = 1e-3;
    
    // 参考压力 (bar)
    double P_ref = 100.0;
    
    // 相对渗透率端点
    double Swi = 0.05;
    double Sor = 0.01;
    double Sgc = 0.05;
};

typedef Eigen::Matrix<double, 3, 1> Deriv3;
typedef Eigen::AutoDiffScalar<Deriv3> AD3;

// 状态变量 (每个计算节点：基质或裂缝段)
template <typename T>
struct StateT {
    T P;  // 油相压力
    T Sw; // 水饱和度
    T Sg; // 气饱和度
    // So = 1 - Sw - Sg;
};
typedef StateT<double> State;
typedef StateT<AD3> StateAD3;

template <typename T>
struct PropertiesT {
    T Bw, Bo, Bg;
    T krw, kro, krg;
    T lw, lo, lg;
};

// 阶段A新增：几何辅助函数
// ==========================

Point3 pointMin(const Point3& a, const Point3& b) {
    return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)};
}

Point3 pointMax(const Point3& a, const Point3& b) {
    return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)};
}

Point3 averagePoints(const std::array<Point3, 4>& pts) {
    Point3 c{0, 0, 0};
    for (const auto& p : pts) c = c + p;
    return c / 4.0;
}

Point3 averagePoints(const std::array<Point3, 8>& pts) {
    Point3 c{0, 0, 0};
    for (const auto& p : pts) c = c + p;
    return c / 8.0;
}

// quad 面法向（单位向量），用 Newell 方法更稳一些
Point3 quadUnitNormal(const std::array<Point3, 4>& q) {
    Point3 n{0, 0, 0};
    for (int i = 0; i < 4; ++i) {
        const Point3& p = q[i];
        const Point3& r = q[(i + 1) % 4];
        n.x += (p.y - r.y) * (p.z + r.z);
        n.y += (p.z - r.z) * (p.x + r.x);
        n.z += (p.x - r.x) * (p.y + r.y);
    }
    double nn = n.norm();
    if (nn < EPSILON) return {0, 0, 0};
    return n / nn;
}

double triangleArea3D(const Point3& a, const Point3& b, const Point3& c) {
    return 0.5 * ((b - a).cross(c - a)).norm();
}

double quadArea3D(const std::array<Point3, 4>& q) {
    return triangleArea3D(q[0], q[1], q[2]) + triangleArea3D(q[0], q[2], q[3]);
}

// 调整四边形顶点顺序，使其法向朝 cell_center 外侧
void orientQuadOutward(std::array<Point3, 4>& q, const Point3& cell_center) {
    Point3 fc = averagePoints(q);
    Point3 n = quadUnitNormal(q);
    if (n.norm() < EPSILON) return;

    // 若法向没有指向“远离 cell center”的方向，则翻转顶点顺序
    if (n.dot(fc - cell_center) < 0.0) {
        std::reverse(q.begin(), q.end());
    }
}

double signedTetVolumeFromOrigin(const Point3& a, const Point3& b, const Point3& c) {
    return a.dot(b.cross(c)) / 6.0;
}

// 通过“外向有序的六个 face 三角化”估算一般六面体体积
double hexaVolumeFromOrientedFaces(const std::array<std::array<Point3, 4>, 6>& face_quads) {
    double v = 0.0;
    for (const auto& q : face_quads) {
        v += signedTetVolumeFromOrigin(q[0], q[1], q[2]);
        v += signedTetVolumeFromOrigin(q[0], q[2], q[3]);
    }
    return std::abs(v);
}

// 局部 face 槽位 -> 8角点编号映射
std::array<int, 4> getLocalFaceCornerIds(int face_id) {
    switch (face_id) {
        case XM: return {0, 4, 7, 3};
        case XP: return {1, 2, 6, 5};
        case YM: return {0, 1, 5, 4};
        case YP: return {3, 7, 6, 2};
        case ZM: return {0, 3, 2, 1};
        case ZP: return {4, 5, 6, 7};
        default: return {0, 1, 2, 3};
    }
}

std::array<Point3, 4> getCellFaceVertices(const Cell& c, int face_id) {
    auto ids = getLocalFaceCornerIds(face_id);
    return {c.corners[ids[0]], c.corners[ids[1]], c.corners[ids[2]], c.corners[ids[3]]};
}

// 用 corners 预计算 cell 的 center / bbox / vol / depth
void computeCellDerivedGeometry(Cell& c) {
    c.center = averagePoints(c.corners);

    c.bbox_min = c.corners[0];
    c.bbox_max = c.corners[0];
    for (int n = 1; n < 8; ++n) {
        c.bbox_min = pointMin(c.bbox_min, c.corners[n]);
        c.bbox_max = pointMax(c.bbox_max, c.corners[n]);
    }

    // 为兼容你后续尚未改造的旧函数，阶段A先继续填 dx/dy/dz
    c.dx = c.bbox_max.x - c.bbox_min.x;
    c.dy = c.bbox_max.y - c.bbox_min.y;
    c.dz = c.bbox_max.z - c.bbox_min.z;

    std::array<std::array<Point3, 4>, 6> face_quads;
    for (int f = 0; f < 6; ++f) {
        face_quads[f] = getCellFaceVertices(c, f);
        orientQuadOutward(face_quads[f], c.center);
    }

    c.vol = hexaVolumeFromOrientedFaces(face_quads);
    c.depth = c.center.z;
}

void computeFaceDerivedGeometry(FaceGeom& f, const Point3& owner_center) {
    orientQuadOutward(f.vertices, owner_center);

    f.center = averagePoints(f.vertices);
    f.normal = quadUnitNormal(f.vertices);
    f.area = quadArea3D(f.vertices);

    f.bbox_min = f.vertices[0];
    f.bbox_max = f.vertices[0];
    for (int i = 1; i < 4; ++i) {
        f.bbox_min = pointMin(f.bbox_min, f.vertices[i]);
        f.bbox_max = pointMax(f.bbox_max, f.vertices[i]);
    }
}


double normalProjectedPerm(const Cell& c, const Point3& n_unit)
{
    // 当前仍假设 K 为对角渗透率张量
    // k_n = n^T K n
    return
        n_unit.x * n_unit.x * c.K[0] +
        n_unit.y * n_unit.y * c.K[1] +
        n_unit.z * n_unit.z * c.K[2];
}

double centerToFaceNormalDistance(const Cell& c, const FaceGeom& f)
{
    // FaceGeom::normal 约定为单位法向
    // 对 TPFA，这里取 center 到面中心沿 face 法向的投影距离
    double d = std::abs((f.center - c.center).dot(f.normal));
    return std::max(d, 1e-10);
}

double computeMMTransmissibilityTPFA(const Cell& cu,
                                     const Cell& cv,
                                     const FaceGeom& f)
{
    // 共享面面积
    double Af = f.area;
    if (Af <= EPSILON) return 0.0;

    // 法向渗透率（face.normal 已为单位向量）
    double kn_u = normalProjectedPerm(cu, f.normal);
    double kn_v = normalProjectedPerm(cv, f.normal);

    if (kn_u <= EPSILON || kn_v <= EPSILON) return 0.0;

    // 两侧中心到共享面的法向距离
    double du = centerToFaceNormalDistance(cu, f);
    double dv = centerToFaceNormalDistance(cv, f);

    // 两个半传导率
    double tau_u = kn_u * Af / du;
    double tau_v = kn_v * Af / dv;

    if (tau_u <= EPSILON || tau_v <= EPSILON) return 0.0;

    // 调和组合
    return (tau_u * tau_v) / std::max(EPSILON, tau_u + tau_v);
}


Point3 mapHexTrilinear(const Cell& cell, double u, double v, double w)
{
    const auto& c = cell.corners;

    double N0 = (1.0 - u) * (1.0 - v) * (1.0 - w);
    double N1 = u * (1.0 - v) * (1.0 - w);
    double N2 = u * v * (1.0 - w);
    double N3 = (1.0 - u) * v * (1.0 - w);
    double N4 = (1.0 - u) * (1.0 - v) * w;
    double N5 = u * (1.0 - v) * w;
    double N6 = u * v * w;
    double N7 = (1.0 - u) * v * w;

    return c[0] * N0 + c[1] * N1 + c[2] * N2 + c[3] * N3
         + c[4] * N4 + c[5] * N5 + c[6] * N6 + c[7] * N7;
}


double averageDistanceCellToPlane(const Cell& cell,
                                  const Point3& planePoint,
                                  const Point3& unitNormal,
                                  int nxs = 2, int nys = 2, int nzs = 2)
{
    double sum = 0.0;
    int count = 0;

    for (int k = 0; k < nzs; ++k) {
        for (int j = 0; j < nys; ++j) {
            for (int i = 0; i < nxs; ++i) {
                double u = (i + 0.5) / (double)nxs;
                double v = (j + 0.5) / (double)nys;
                double w = (k + 0.5) / (double)nzs;

                Point3 p = mapHexTrilinear(cell, u, v, w);
                double d = std::abs((p - planePoint).dot(unitNormal));
                sum += d;
                count++;
            }
        }
    }

    if (count == 0) return 0.0;
    return sum / (double)count;
}

double computeMatrixFractureTransmissibility(const Cell& cell,
                                             const Segment& seg,
                                             int nxs = 2, int nys = 2, int nzs = 2)
{
    if (seg.area <= EPSILON) return 0.0;

    // 裂缝法向应为单位向量；这里再做一次保护
    Point3 n = seg.normal;
    double nn = n.norm();
    if (nn < EPSILON) return 0.0;
    n = n * (1.0 / nn);

    // 基质渗透率沿裂缝法向的投影
    double Kn = normalProjectedPerm(cell, n);
    if (Kn <= EPSILON) return 0.0;

    // 在一般六面体内部做采样，求 cell 内各采样点到 fracture plane 的平均距离
    // 这里 planePoint 直接用 seg.center，比 poly[0] 更稳
    double d_avg = averageDistanceCellToPlane(cell, seg.center, n, nxs, nys, nzs);

    double scale = std::max(1.0, std::max(cell.dx, std::max(cell.dy, cell.dz)));
    d_avg = std::max(d_avg, 1e-10 * scale);

    // 延续你当前代码的 EDFM 近似：
    // Tmf = 2 * A * Kn / d_avg
    double Tmf = 2.0 * seg.area * (Kn / d_avg);

    if (!std::isfinite(Tmf) || Tmf <= EPSILON) return 0.0;
    return Tmf;
}



Point3 fractureCenter(const Fracture& f)
{
    Point3 c{0, 0, 0};
    for (int i = 0; i < 4; ++i) c = c + f.vertices[i];
    return c / 4.0;
}


void buildPlaneBasisFromNormal(const Point3& normal, Point3& e1, Point3& e2)
{
    Point3 n = normal;
    double nn = n.norm();
    if (nn < EPSILON) {
        e1 = {1, 0, 0};
        e2 = {0, 1, 0};
        return;
    }
    n = n * (1.0 / nn);

    // 选一个与 n 不太平行的参考方向
    Point3 ref;
    if (std::abs(n.x) <= std::abs(n.y) && std::abs(n.x) <= std::abs(n.z)) {
        ref = {1, 0, 0};
    } else if (std::abs(n.y) <= std::abs(n.x) && std::abs(n.y) <= std::abs(n.z)) {
        ref = {0, 1, 0};
    } else {
        ref = {0, 0, 1};
    }

    e1 = n.cross(ref);
    double ne1 = e1.norm();
    if (ne1 < EPSILON) {
        // 极端退化保护
        ref = {0, 1, 0};
        e1 = n.cross(ref);
        ne1 = e1.norm();
        if (ne1 < EPSILON) {
            e1 = {1, 0, 0};
            e2 = {0, 1, 0};
            return;
        }
    }
    e1 = e1 * (1.0 / ne1);

    e2 = n.cross(e1);
    double ne2 = e2.norm();
    if (ne2 < EPSILON) {
        e2 = {0, 1, 0};
    } else {
        e2 = e2 * (1.0 / ne2);
    }
}

bool computeSegmentLocalPlaneExtents(const Segment& seg, double& Lu, double& Lv)
{
    Lu = 0.0;
    Lv = 0.0;

    if (seg.poly.size() < 3) return false;

    Point3 e1, e2;
    buildPlaneBasisFromNormal(seg.normal, e1, e2);

    double umin = 1e100, umax = -1e100;
    double vmin = 1e100, vmax = -1e100;

    for (const auto& p : seg.poly) {
        Point3 d = p - seg.center;
        double u = d.dot(e1);
        double v = d.dot(e2);

        umin = std::min(umin, u);
        umax = std::max(umax, u);
        vmin = std::min(vmin, v);
        vmax = std::max(vmax, v);
    }

    Lu = std::max(0.0, umax - umin);
    Lv = std::max(0.0, vmax - vmin);

    // 退化保护：
    // 如果某个方向数值太小，但 area 是正的，用 area/另一方向 做一个修复
    if (Lu <= EPSILON && Lv > EPSILON && seg.area > EPSILON) {
        Lu = seg.area / Lv;
    }
    if (Lv <= EPSILON && Lu > EPSILON && seg.area > EPSILON) {
        Lv = seg.area / Lu;
    }

    return (Lu > EPSILON && Lv > EPSILON);
}


double computeSegmentEquivalentRadius(const Segment& seg, double rw)
{
    double Lu = 0.0, Lv = 0.0;
    bool ok = computeSegmentLocalPlaneExtents(seg, Lu, Lv);

    double re = 0.0;

    if (ok) {
        re = 0.14 * std::sqrt(Lu * Lu + Lv * Lv);
    } else {
        // 再退化保护：如果局部长宽提取失败，则用面积反推一个等效尺度
        if (seg.area > EPSILON) {
            double Leq = std::sqrt(seg.area);
            re = 0.14 * std::sqrt(2.0) * Leq;
        } else {
            re = 1.1 * rw;
        }
    }

    // 必须保证 re > rw，否则 log(re/rw) 会出问题
    re = std::max(re, 1.1 * rw);
    return re;
}




// =============================================================================
// 2. 几何计算模块 (3D Sutherland-Hodgman 剪裁)
// =============================================================================

// 计算多边形面积  a和b叉积的模就是以a，b为边的平行四边形面积，这里是定一个点，拆成了多个三角形
double polygonArea(const std::vector<Point3>& poly) {
    if (poly.size() < 3) return 0.0;
    Point3 total = {0, 0, 0};
    Point3 v0 = poly[0];
    for (size_t i = 1; i < poly.size() - 1; ++i) {
        Point3 v1 = poly[i];
        Point3 v2 = poly[i+1];
        total = total + (v1 - v0).cross(v2 - v0);
    }
    return 0.5 * total.norm();
}
//计算面积质心
Point3 polygonCenter(const std::vector<Point3>& poly) {
    Point3 c{0, 0, 0};
    if (poly.empty()) return c;
    if (poly.size() < 3) {
        for (const auto& p : poly) c = c + p;
        return c * (1.0 / poly.size());
    }

    Point3 v0 = poly[0];
    double A_total = 0.0;
    Point3 C_total{0, 0, 0};

    for (size_t i = 1; i + 1 < poly.size(); ++i) {
        Point3 v1 = poly[i];
        Point3 v2 = poly[i + 1];

        double Ai = 0.5 * ((v1 - v0).cross(v2 - v0)).norm();
        if (Ai < EPSILON) continue;

        Point3 triC = (v0 + v1 + v2) * (1.0 / 3.0);
        C_total = C_total + triC * Ai;
        A_total += Ai;
    }

    if (A_total < EPSILON) {
        for (const auto& p : poly) c = c + p;
        return c * (1.0 / poly.size());
    }

    return C_total * (1.0 / A_total);
}

// 检查点是否在平面的内侧
bool isInside(const Point3& p, const Point3& planeNormal, double planeD) {
    return (planeNormal.dot(p) + planeD) >= -1e-9;
}

// 计算线段与平面的交点  加入容差！！！
Point3 intersectPlane(const Point3& p1, const Point3& p2,
                      const Point3& planeNormal, double planeD) {
    double d1 = planeNormal.dot(p1) + planeD;
    double d2 = planeNormal.dot(p2) + planeD;

    const double eps = 1e-12;

    if (std::abs(d1) < eps) return p1;
    if (std::abs(d2) < eps) return p2;

    double denom = d1 - d2;
    if (std::abs(denom) < eps) return p1; // 退化保护

    double t = d1 / (d1 - d2);
    return p1 + (p2 - p1) * t;
}

// Sutherland-Hodgman 多边形剪裁 (针对一个平面)
std::vector<Point3> clipPolygonCurrentPlane(const std::vector<Point3>& inputPoly, const Point3& normal, double d) {
    std::vector<Point3> outputPoly;
    if (inputPoly.empty()) return outputPoly;

    for (size_t i = 0; i < inputPoly.size(); ++i) {
        Point3 cur = inputPoly[i];
        Point3 prev = inputPoly[(i + inputPoly.size() - 1) % inputPoly.size()];

        bool curIn = isInside(cur, normal, d);
        bool prevIn = isInside(prev, normal, d);

        if (curIn) {
            if (!prevIn) {
                outputPoly.push_back(intersectPlane(prev, cur, normal, d));
            }
            outputPoly.push_back(cur);
        } else if (prevIn) {
            outputPoly.push_back(intersectPlane(prev, cur, normal, d));
        }
    }
    return outputPoly;
}


bool bboxOverlap(const Point3& a_min, const Point3& a_max,
                 const Point3& b_min, const Point3& b_max,
                 double tol = 1e-12)
{
    if (a_max.x < b_min.x - tol || b_max.x < a_min.x - tol) return false;
    if (a_max.y < b_min.y - tol || b_max.y < a_min.y - tol) return false;
    if (a_max.z < b_min.z - tol || b_max.z < a_min.z - tol) return false;
    return true;
}

// 清理裁剪后的 polygon：
// 1) 去掉连续重复点
// 2) 若首尾重复，去掉尾点
std::vector<Point3> cleanupPolygon3D(const std::vector<Point3>& poly, double tol)
{
    std::vector<Point3> out;
    if (poly.empty()) return out;

    for (const auto& p : poly) {
        if (out.empty() || (p - out.back()).norm() > tol) {
            out.push_back(p);
        }
    }

    if (out.size() >= 2 && (out.front() - out.back()).norm() <= tol) {
        out.pop_back();
    }

    return out;
}

// 对于某个 cell 的某个局部 face，构造“朝 cell 内部”的裁剪平面
// 输出平面方程： n_in · x + d >= 0  表示在 cell 内部
bool getInwardPlaneForCellFace(const Cell& cell,
                               const std::vector<FaceGeom>& faces,
                               int local_face_id,
                               Point3& n_in,
                               double& d)
{
    int fid = cell.face_ids[local_face_id];
    if (fid < 0) return false;

    const FaceGeom& fg = faces[fid];

    // fg.normal 约定为“朝 owner 外侧”
    // 对当前 cell 来说，要得到“朝内部”的法向
    if (fg.owner == cell.id) {
        n_in = fg.normal * (-1.0);
    } else if (fg.neighbor == cell.id) {
        n_in = fg.normal;
    } else {
        return false;
    }

    // face 上任取一点即可
    const Point3& p0 = fg.vertices[0];
    d = -n_in.dot(p0);
    return true;
}

// 用一个一般六面体 cell 的 6 个真实面，逐面裁剪 polygon
std::vector<Point3> clipPolygonByCell(const std::vector<Point3>& inputPoly,
                                      const Cell& cell,
                                      const std::vector<FaceGeom>& faces)
{
    std::vector<Point3> poly = inputPoly;

    double scale = std::max(1.0, std::max(cell.dx, std::max(cell.dy, cell.dz)));
    double tol = 1e-10 * scale;

    for (int lf = 0; lf < 6; ++lf) {
        Point3 n_in;
        double d = 0.0;
        if (!getInwardPlaneForCellFace(cell, faces, lf, n_in, d)) {
            poly.clear();
            return poly;
        }

        poly = clipPolygonCurrentPlane(poly, n_in, d);
        poly = cleanupPolygon3D(poly, tol);

        if (poly.size() < 3) {
            poly.clear();
            return poly;
        }
    }

    return poly;
}



std::vector<Point3> clipFractureBox(const Fracture& frac,
                                    const Cell& cell,
                                    const std::vector<FaceGeom>& faces)
{
    std::vector<Point3> poly;
    for (int i = 0; i < 4; ++i) poly.push_back(frac.vertices[i]);

    double scale = std::max(1.0, std::max(cell.dx, std::max(cell.dy, cell.dz)));
    double tol = 1e-10 * scale;

    poly = cleanupPolygon3D(poly, tol);
    if (poly.size() < 3) return {};

    poly = clipPolygonByCell(poly, cell, faces);
    poly = cleanupPolygon3D(poly, tol);

    if (poly.size() < 3) return {};
    return poly;
}



void pushUniquePoint(std::vector<Point3>& pts, const Point3& p, double tol) {   //点去重辅助
    for (const auto& q : pts) {
        if ((p - q).norm() <= tol) return;
    }
    pts.push_back(p);
}




// 3D三角形内点判断（假设点已与三角形近共面）
bool pointInTriangle3D(const Point3& p,
                       const Point3& a,
                       const Point3& b,
                       const Point3& c,
                       double tol)
{
    Point3 v0 = b - a;
    Point3 v1 = c - a;
    Point3 v2 = p - a;

    double d00 = v0.dot(v0);
    double d01 = v0.dot(v1);
    double d11 = v1.dot(v1);
    double d20 = v2.dot(v0);
    double d21 = v2.dot(v1);

    double denom = d00 * d11 - d01 * d01;
    if (std::abs(denom) < EPSILON) return false;

    double v = (d11 * d20 - d01 * d21) / denom;
    double w = (d00 * d21 - d01 * d20) / denom;
    double u = 1.0 - v - w;

    return (u >= -tol && v >= -tol && w >= -tol);
}

// 凸四边形内点判断：拆成两个三角形
bool pointInConvexQuad3D(const Point3& p,
                         const std::array<Point3, 4>& q,
                         double tol)
{
    return pointInTriangle3D(p, q[0], q[1], q[2], tol) ||
           pointInTriangle3D(p, q[0], q[2], q[3], tol);
}




bool pointOnCellFace(const Point3& p,
                     const Cell& cell,
                     const std::vector<FaceGeom>& faces,
                     int face_id,
                     double tol)
{
    int fid = cell.face_ids[face_id];
    if (fid < 0) return false;

    const FaceGeom& fg = faces[fid];
    if (fg.area <= EPSILON) return false;

    // 1) 先判断是否在该 face 所在平面上
    const Point3& p0 = fg.vertices[0];
    double dist_to_plane = std::abs((p - p0).dot(fg.normal));
    if (dist_to_plane > tol) return false;

    // 2) 再判断是否落在 face 四边形内部
    return pointInConvexQuad3D(p, fg.vertices, tol);
}

bool extractTraceOnFace(const std::vector<Point3>& poly,
                        const Cell& cell,
                        const std::vector<FaceGeom>& faces,
                        int face_id,
                        Point3& a,
                        Point3& b,
                        double& len)
{
    len = 0.0;
    if (poly.size() < 2) return false;

    int fid = cell.face_ids[face_id];
    if (fid < 0) return false;

    const FaceGeom& fg = faces[fid];
    if (fg.area <= EPSILON) return false;

    double scale = std::max(1.0, std::max(cell.dx, std::max(cell.dy, cell.dz)));
    double tol = 1e-8 * scale;

    const Point3& p0 = fg.vertices[0];
    const Point3& n = fg.normal;
    double planeD = -n.dot(p0);

    std::vector<Point3> pts;

    auto try_add_point = [&](const Point3& q) {
        if (pointOnCellFace(q, cell, faces, face_id, tol)) {
            pushUniquePoint(pts, q, tol);
        }
    };

    // 1) 先收集本来就在该 face 上的 polygon 顶点
    for (const auto& p : poly) {
        try_add_point(p);
    }

    // 2) 再检查 polygon 边与 face 平面的交点
    for (size_t i = 0; i < poly.size(); ++i) {
        const Point3& p1 = poly[i];
        const Point3& p2 = poly[(i + 1) % poly.size()];

        double d1 = n.dot(p1) + planeD;
        double d2 = n.dot(p2) + planeD;

        bool on1 = std::abs(d1) <= tol;
        bool on2 = std::abs(d2) <= tol;

        if (on1 && on2) {
            try_add_point(p1);
            try_add_point(p2);
            continue;
        }

        // 真正跨过该平面
        if ((d1 > tol && d2 < -tol) || (d1 < -tol && d2 > tol)) {
            Point3 q = intersectPlane(p1, p2, n, planeD);
            try_add_point(q);
        }
    }

    if (pts.size() < 2) return false;

    // 3) 取最远两点作为 trace 端点
    double best = -1.0;
    Point3 pa{0, 0, 0}, pb{0, 0, 0};

    for (size_t i = 0; i < pts.size(); ++i) {
        for (size_t j = i + 1; j < pts.size(); ++j) {
            double d = (pts[i] - pts[j]).norm();
            if (d > best) {
                best = d;
                pa = pts[i];
                pb = pts[j];
            }
        }
    }

    if (best <= EPSILON) return false;

    a = pa;
    b = pb;
    len = best;
    return true;
}

double pointToLineDistance3D(const Point3& p, const Point3& a, const Point3& b) { //点到3D无限直线的距离
    Point3 ab = b - a;
    double lab = ab.norm();
    if (lab < EPSILON) return 0.0;
    return ((p - a).cross(ab)).norm() / lab;
}

void fillSegmentFaceGeom(Segment& seg,
                         const std::vector<Point3>& poly,
                         const Cell& cell,
                         const std::vector<FaceGeom>& faces)
{
    for (int f = 0; f < 6; ++f) {
        seg.face_trace_len[f] = 0.0;
        seg.face_center_dist[f] = 0.0;

        Point3 a, b;
        double len = 0.0;

        if (extractTraceOnFace(poly, cell, faces, f, a, b, len)) {
            seg.face_trace_len[f] = len;

            double dist = pointToLineDistance3D(seg.center, a, b);
            seg.face_center_dist[f] = std::max(dist, 1e-10);
        }
    }
}

std::pair<int,int> getSharedFaces(const Cell& c1, const Cell& c2) {  //判断两个相邻 cell 共享哪两个面
    int dix = c2.ix - c1.ix;
    int diy = c2.iy - c1.iy;
    int diz = c2.iz - c1.iz;

    if (dix ==  1 && diy == 0 && diz == 0) return {XP, XM};
    if (dix == -1 && diy == 0 && diz == 0) return {XM, XP};

    if (dix == 0 && diy ==  1 && diz == 0) return {YP, YM};
    if (dix == 0 && diy == -1 && diz == 0) return {YM, YP};

    if (dix == 0 && diy == 0 && diz ==  1) return {ZP, ZM};
    if (dix == 0 && diy == 0 && diz == -1) return {ZM, ZP};

    return {-1, -1};
}


bool getSharedFacePairByFaceIds(const Cell& c1,
                                const Cell& c2,
                                int& lf1,
                                int& lf2,
                                int& shared_fid)
{
    lf1 = -1;
    lf2 = -1;
    shared_fid = -1;

    for (int f1 = 0; f1 < 6; ++f1) {
        int id1 = c1.face_ids[f1];
        if (id1 < 0) continue;

        for (int f2 = 0; f2 < 6; ++f2) {
            int id2 = c2.face_ids[f2];
            if (id2 < 0) continue;

            if (id1 == id2) {
                lf1 = f1;
                lf2 = f2;
                shared_fid = id1;
                return true;
            }
        }
    }

    return false;
}






double pointToSegmentDistance3D(const Point3& p, const Point3& a, const Point3& b) //点到有限线段的最短距离
{
    Point3 ab = b - a;
    double ab2 = ab.dot(ab);
    if (ab2 < EPSILON) {
        return (p - a).norm();   // 退化成一个点
    }

    double t = (p - a).dot(ab) / ab2;    //利用q点，p-q垂直于b-a，p-q与b-a的点积为0
    t = std::max(0.0, std::min(1.0, t));  //把t限制在线段上

    Point3 q = a + ab * t;   // 线段上最近点
    return (p - q).norm();
}


bool intersectTwoPlanes(const Point3& n1, const Point3& p1,   //求两个无限平面的交线
                        const Point3& n2, const Point3& p2,
                        Point3& linePoint, Point3& lineDir)
{
    // 原始方向（未归一化）
    Point3 dir = n1.cross(n2);
    double dir2 = dir.dot(dir);

    // 平行或近似平行
    if (dir2 < 1e-14) {
        return false;
    }

    double d1 = n1.dot(p1);
    double d2 = n2.dot(p2);

    // 交线上一点
    Point3 term1 = n2.cross(dir) * d1;
    Point3 term2 = dir.cross(n1) * d2;
    linePoint = (term1 + term2) * (1.0 / dir2);  //交线上某点，即X0

    // 把方向单位化，后面裁剪更方便
    lineDir = dir * (1.0 / std::sqrt(dir2));  //交线方向（单位向量）

    return true;
}

bool clipLineByConvexPolygon(const std::vector<Point3>& poly,  //把一条直线x(t)=linePoint+tlineDir 用一个凸多边形裁剪，求出这条直线落在多边形内部的那一段参数区间[tmin,tmax]
                             const Point3& normal,
                             const Point3& linePoint,
                             const Point3& lineDir,
                             double& tmin,
                             double& tmax)
{
    if (poly.size() < 3) return false;

    Point3 centroid = polygonCenter(poly);   // 你现在这个 polygonCenter 已经是面积质心版本

    tmin = -1e100;
    tmax =  1e100;

    for (size_t i = 0; i < poly.size(); ++i) {
        const Point3& vi = poly[i];
        const Point3& vj = poly[(i + 1) % poly.size()];

        Point3 e = vj - vi;

        // 候选边法向量（在 polygon 平面内）
        Point3 m = normal.cross(e);

        // 退化边跳过
        if (m.norm() < EPSILON) continue;

        // 调整方向，使其指向 polygon 内部
        if (m.dot(centroid - vi) < 0.0) {
            m = m * (-1.0);
        }

        double c = m.dot(linePoint - vi);
        double den = m.dot(lineDir);

        const double tol = 1e-12;

        // 直线与这条边界平行
        if (std::abs(den) < tol) {
            // 若 linePoint 在外侧，则整条线都在外面
            if (c < -tol) return false;
            // 否则这条边对 t 没约束
            continue;
        }

        // 不等式： c + den * t >= 0
        double tbound = -c / den;

        if (den > 0.0) {
            // t >= tbound
            tmin = std::max(tmin, tbound);
        } else {
            // t <= tbound
            tmax = std::min(tmax, tbound);
        }

        if (tmin > tmax) return false;
    }

    return true;
}

bool computeCrossIntersectionSegment(const Segment& seg1,  //
                                     const Segment& seg2,
                                     Point3& a,
                                     Point3& b,
                                     double& ell_int)
{
    ell_int = 0.0;

    // 1) 先求两平面的交线
    Point3 linePoint, lineDir;
    if (!intersectTwoPlanes(seg1.normal, seg1.center,
                            seg2.normal, seg2.center,
                            linePoint, lineDir)) {
        return false;
    }

    // 2) 把交线裁到 seg1.poly 内
    double t1min, t1max;
    if (!clipLineByConvexPolygon(seg1.poly, seg1.normal,
                                 linePoint, lineDir,
                                 t1min, t1max)) {
        return false;
    }

    // 3) 把交线裁到 seg2.poly 内
    double t2min, t2max;
    if (!clipLineByConvexPolygon(seg2.poly, seg2.normal,
                                 linePoint, lineDir,
                                 t2min, t2max)) {
        return false;
    }

    // 4) 两段区间求交集
    double ta = std::max(t1min, t2min);
    double tb = std::min(t1max, t2max);

    if (tb - ta <= 1e-10) {
        return false;   // 无交线，或只接触于一点，或数值退化
    }

    a = linePoint + lineDir * ta;
    b = linePoint + lineDir * tb;
    ell_int = (b - a).norm();

    if (ell_int <= EPSILON) return false;

    return true;
}

bool computeCrossGeom(const Segment& seg1,
                      const Segment& seg2,
                      double& ell_int,
                      double& L1,
                      double& L2)
{
    Point3 a, b;
    if (!computeCrossIntersectionSegment(seg1, seg2, a, b, ell_int)) {
        return false;
    }

    L1 = pointToSegmentDistance3D(seg1.center, a, b);
    L2 = pointToSegmentDistance3D(seg2.center, a, b);

    // 防止除零
    L1 = std::max(L1, 1e-10);
    L2 = std::max(L2, 1e-10);

    return true;
}

// =============================================================================
// 3. 物理模型辅助函数 (PVT & RelPerm)
// =============================================================================

FluidProps g_props;

// 计算体积系数 B (Formation Volume Factor)
// B = exp(C * (P - Pref))
template <typename T>
void calcPVT(const T& P, T& Bw, T& Bo, T& Bg, T& dBw_dP, T& dBo_dP, T& dBg_dP) {
    T dP = P - g_props.P_ref;
    using std::exp;
    Bw = exp(-g_props.cw * dP);
    Bo = exp(-g_props.co * dP);
    Bg = exp(-g_props.cg * dP);
}

// 计算相对渗透率 (Corey Model)
template <typename T>
void calcRelPerm(const T& Sw, const T& Sg, T& krw, T& kro, T& krg) {
    auto clamp01_T = [](const T& v) -> T {
        T zero(0.0), one(1.0);
        return (v < zero) ? zero : ((v > one) ? one : v);
    };
    T Sw_norm = (Sw - g_props.Swi) / (1.0 - g_props.Swi - g_props.Sor);
    T Sg_norm = (Sg - g_props.Sgc) / (1.0 - g_props.Sgc - g_props.Swi - g_props.Sor);
    Sw_norm = clamp01_T(Sw_norm);
    Sg_norm = clamp01_T(Sg_norm);
    
    krw = Sw_norm * Sw_norm;
    krg = Sg_norm * Sg_norm;
    T So_norm = clamp01_T(T(1.0) - Sw_norm - Sg_norm);
    kro = So_norm * So_norm;
}

// =============================================================================
// Python Result Structure
// =============================================================================

struct SimulationResult {
    py::array_t<double> pressure_field;
    py::array_t<double> temperature_field;
    py::array_t<double> stress_field;
    py::array_t<double> fracture_vertices;
    py::array_t<int> fracture_cells;
};

// =============================================================================
// 4. 模拟器类 (Simulator)
// =============================================================================

// =============================================================================
// 4. 模拟器类 (Simulator) - 性能优化版
// =============================================================================

class Simulator {
public:
    // --- 基础数据 ---
    int Nx, Ny, Nz;
    double Lx, Ly, Lz;
    double dx, dy, dz;
    std::vector<Cell> cells;
    std::vector<Fracture> fractures;
    std::vector<Segment> segments;
    std::vector<FaceGeom> faces;
    // --- 拓扑连接优化 ---
    // 原有的 connections 用于全局遍历，新增 adj 用于快速查找邻居
    std::vector<Connection> connections;
    
    // 邻接表结构：adj[u] 包含所有连接到 u 的 {v, T, type}
    struct Neighbor {
        int v;      // 邻居节点索引
        double T;   // 传导率
        int conn_idx; // 在全局 connections 中的下标（可选，用于调试）
    };
    std::vector<std::vector<Neighbor>> adj; // adj[u] -> list of neighbors

    // 状态量
    int n_matrix;
    int n_frac_nodes;
    int n_total;
    
    std::vector<State> states; 
    std::vector<State> states_prev; 
    
    struct Well {
        int target_node_idx; 
        double WI; 
        double P_bhp;
    };
    std::vector<Well> wells;
    // 井的快速查找：well_map[node_idx] -> well_idx
    std::map<int, int> well_map; 

    SparseMatrix<double> J;
    struct CellOffsets { int diag[3][3]; };
    struct ConnOffsets { int off_uv[3][3]; int off_vu[3][3]; };
    std::vector<CellOffsets> cell_J_idx;
    std::vector<ConnOffsets> conn_J_idx;

    Simulator() {
        Lx = 3000; Ly = 300; Lz = 40;
        Nx = 150; Ny = 15; Nz = 2; 
        dx = Lx/Nx; dy = Ly/Ny; dz = Lz/Nz;
    }


    int getNextFractureId() const {
        int max_id = -1;
        for (const auto& f : fractures) {
            max_id = std::max(max_id, f.id);
        }
        return max_id + 1;
    }




    void buildGridFaces() {
        faces.clear();

        for (auto& c : cells) {
            c.face_ids = {{-1, -1, -1, -1, -1, -1}};
        }

        auto add_face = [&](int owner, int local_owner_face,
                            int neighbor, int local_neighbor_face) {
            FaceGeom f;
            f.id = (int)faces.size();
            f.owner = owner;
            f.neighbor = neighbor;
            f.local_owner_face = local_owner_face;
            f.local_neighbor_face = local_neighbor_face;

            f.vertices = getCellFaceVertices(cells[owner], local_owner_face);
            computeFaceDerivedGeometry(f, cells[owner].center);

            faces.push_back(f);

            cells[owner].face_ids[local_owner_face] = f.id;
            if (neighbor >= 0) {
                cells[neighbor].face_ids[local_neighbor_face] = f.id;
            }
        };

        for (int k = 0; k < Nz; ++k) {
            for (int j = 0; j < Ny; ++j) {
                for (int i = 0; i < Nx; ++i) {
                    int u = k * Nx * Ny + j * Nx + i;

                    // x- 边界面
                    if (i == 0) {
                        add_face(u, XM, -1, -1);
                    }
                    // y- 边界面
                    if (j == 0) {
                        add_face(u, YM, -1, -1);
                    }
                    // z- 边界面
                    if (k == 0) {
                        add_face(u, ZM, -1, -1);
                    }

                    // x+ 面：若有右邻居则创建内部面，否则创建边界面
                    if (i < Nx - 1) {
                        int v = u + 1;
                        add_face(u, XP, v, XM);
                    } else {
                        add_face(u, XP, -1, -1);
                    }

                    // y+ 面
                    if (j < Ny - 1) {
                        int v = u + Nx;
                        add_face(u, YP, v, YM);
                    } else {
                        add_face(u, YP, -1, -1);
                    }

                    // z+ 面
                    if (k < Nz - 1) {
                        int v = u + Nx * Ny;
                        add_face(u, ZP, v, ZM);
                    } else {
                        add_face(u, ZP, -1, -1);
                    }
                }
            }
        }
    }


    void initGrid() {
        n_matrix = Nx * Ny * Nz;
        cells.resize(n_matrix);

        for (int k = 0; k < Nz; ++k) {
            for (int j = 0; j < Ny; ++j) {
                for (int i = 0; i < Nx; ++i) {
                    int id = k * Nx * Ny + j * Nx + i;

                    Cell c;
                    c.id = id;
                    c.ix = i;
                    c.iy = j;
                    c.iz = k;

                    // 先构造规则网格的 8 个角点
                    double x0 = i * dx;
                    double x1 = (i + 1) * dx;
                    double y0 = j * dy;
                    double y1 = (j + 1) * dy;
                    double z0 = k * dz;
                    double z1 = (k + 1) * dz;

                    c.corners[0] = {x0, y0, z0};
                    c.corners[1] = {x1, y0, z0};
                    c.corners[2] = {x1, y1, z0};
                    c.corners[3] = {x0, y1, z0};
                    c.corners[4] = {x0, y0, z1};
                    c.corners[5] = {x1, y0, z1};
                    c.corners[6] = {x1, y1, z1};
                    c.corners[7] = {x0, y1, z1};

                    // 物性
                    c.phi = 0.04;
                    c.K[0] = 0.005;
                    c.K[1] = 0.005;
                    c.K[2] = 0.005;

                    // 用 corners 反算几何量
                    computeCellDerivedGeometry(c);

                    cells[id] = c;
                }
            }
        }

        // 构建全局 face 几何与 cell-face 映射
        buildGridFaces();
    }



    void generateFractures(int total_fracs = 60, 
                       double min_L = 30.0, double max_L = 80.0, 
                       double max_dip = PI/3.0, 
                       double min_strike = 0.0, double max_strike = PI,
                       double aperture_val = 0.01, double perm_val = 1000.0,
                       double range_x_min = 0.0, double range_x_max = -1.0,
                       double range_y_min = 0.0, double range_y_max = -1.0,
                       double range_z_min = 0.0, double range_z_max = -1.0) {
    
        double use_max_x = (range_x_max < 0) ? Lx : range_x_max;
        double use_max_y = (range_y_max < 0) ? Ly : range_y_max;
        double use_max_z = (range_z_max < 0) ? Lz : range_z_max;

        fractures.clear();

        std::mt19937 rng(42);
        std::uniform_real_distribution<double> distX(range_x_min, use_max_x);
        std::uniform_real_distribution<double> distY(range_y_min, use_max_y);
        std::uniform_real_distribution<double> distZ(range_z_min, use_max_z);
        std::uniform_real_distribution<double> distAngle(min_strike, max_strike);
        std::uniform_real_distribution<double> distDip(0, max_dip);
        std::uniform_real_distribution<double> distL(min_L, max_L);

        auto inBox = [&](const Point3& p) -> bool {
            return (p.x >= 0.0 && p.x <= Lx &&
                    p.y >= 0.0 && p.y <= Ly &&
                    p.z >= 0.0 && p.z <= Lz);
        };

        auto fracVerticesInBox = [&](const Fracture& f) -> bool {
            return inBox(f.vertices[0]) && inBox(f.vertices[1]) &&
                inBox(f.vertices[2]) && inBox(f.vertices[3]);
        };

        for (int i = 0; i < total_fracs; ++i) {
            Fracture f;
            f.id = i; // 天然裂缝从 0 开始连续编号
            f.aperture = aperture_val;
            f.perm = perm_val;
            f.is_hydraulic = false; // 明确标记为天然裂缝

            int tries = 0;
            while (true) {
                if (++tries > 200000) {
                    std::cerr << "Failed to place natural fracture " << i
                            << " fully inside domain. "
                            << "Consider reducing distL/distDip or enlarging domain.\n";
                    return;
                }

                Point3 center = {distX(rng), distY(rng), distZ(rng)};
                double len = distL(rng);
                double height = distL(rng) * 0.5;
                double strike = distAngle(rng);
                double dip = distDip(rng);

                Point3 u = {cos(strike), sin(strike), 0};
                Point3 n_horiz = {-sin(strike), cos(strike), 0};
                Point3 v = {n_horiz.x * cos(dip), n_horiz.y * cos(dip), -sin(dip)};

                f.vertices[0] = center - u*(len/2) - v*(height/2);
                f.vertices[1] = center + u*(len/2) - v*(height/2);
                f.vertices[2] = center + u*(len/2) + v*(height/2);
                f.vertices[3] = center - u*(len/2) + v*(height/2);

                if (fracVerticesInBox(f)) {
                    fractures.push_back(f);
                    break;
                }
            }
        }
    }

    void generateHydraulicFractures(int total_fracs = 20,
                                double well_length = 2000.0,
                                double hf_len = 120.0,
                                double hf_height = 40.0,
                                double aperture_val = 0.01,
                                double perm_val = 10000.0,
                                double x_center = -1.0,
                                double y_center = -1.0,
                                double z_center = -1.0,
                                int start_id = -1) {
    
        if (total_fracs <= 0) {
            std::cout << "No hydraulic fractures requested." << std::endl;
            return;
        }

        double xc = (x_center < 0.0) ? (Lx / 2.0) : x_center;
        double yc = (y_center < 0.0) ? (Ly / 2.0) : y_center;
        double zc = (z_center < 0.0) ? (Lz / 2.0) : z_center;

        double y_min = yc - hf_len / 2.0;
        double y_max = yc + hf_len / 2.0;
        double z_min = zc - hf_height / 2.0;
        double z_max = zc + hf_height / 2.0;

        if (y_min < 0.0 || y_max > Ly || z_min < 0.0 || z_max > Lz) {
            std::cerr << "Hydraulic fracture geometry exceeds domain in y/z direction." << std::endl;
            return;
        }

        if (total_fracs > 1) {
            double x_start_check = xc - well_length / 2.0;
            double x_end_check   = xc + well_length / 2.0;
            if (x_start_check < 0.0 || x_end_check > Lx) {
                std::cerr << "Hydraulic fracture distribution exceeds domain in x direction." << std::endl;
                return;
            }
        } else {
            if (xc < 0.0 || xc > Lx) {
                std::cerr << "Hydraulic fracture center exceeds domain in x direction." << std::endl;
                return;
            }
        }

        // 若未显式指定 start_id，则自动从当前已有裂缝编号之后开始
        int base_id = (start_id >= 0) ? start_id : getNextFractureId();

        double spacing = 0.0;
        double x_start = xc;
        if (total_fracs > 1) {
            x_start = xc - well_length / 2.0;
            spacing = well_length / (total_fracs - 1);
        }

        for (int k = 0; k < total_fracs; ++k) {
            Fracture f;
            f.id = base_id + k;
            f.aperture = aperture_val;
            f.perm = perm_val;
            f.is_hydraulic = true;

            double x_curr = (total_fracs == 1) ? xc : (x_start + k * spacing);

            // 裂缝面位于 x = x_curr，为垂直于 x 方向的一组竖直裂缝面
            f.vertices[0] = {x_curr, yc - hf_len/2.0, zc - hf_height/2.0};
            f.vertices[1] = {x_curr, yc + hf_len/2.0, zc - hf_height/2.0};
            f.vertices[2] = {x_curr, yc + hf_len/2.0, zc + hf_height/2.0};
            f.vertices[3] = {x_curr, yc - hf_len/2.0, zc + hf_height/2.0};

            fractures.push_back(f);
        }

        std::cout << "Generated " << total_fracs << " hydraulic fractures. "
                << "ID range: [" << base_id << ", " << (base_id + total_fracs - 1) << "]"
                << std::endl;
    }


    void processGeometry() {
        segments.clear();
        int seg_id_counter = 0;

        for (const auto& frac : fractures) {
            // 1. fracture 自身 bbox
            Point3 fmin = frac.vertices[0];
            Point3 fmax = frac.vertices[0];
            for (int i = 1; i < 4; ++i) {
                fmin = pointMin(fmin, frac.vertices[i]);
                fmax = pointMax(fmax, frac.vertices[i]);
            }

            // 2. fracture 平面法向
            Point3 vec1 = frac.vertices[1] - frac.vertices[0];
            Point3 vec2 = frac.vertices[3] - frac.vertices[0];
            Point3 normal = vec1.cross(vec2);
            double nn = normal.norm();
            if (nn < EPSILON) {
                std::cerr << "Warning: fracture " << frac.id << " has degenerate geometry, skipped.\n";
                continue;
            }
            normal = normal * (1.0 / nn);

            // 3. 遍历所有 cell，用 bbox 粗筛，再精确裁剪
            for (int cell_idx = 0; cell_idx < n_matrix; ++cell_idx) {
                const Cell& cell = cells[cell_idx];

                // 粗筛：fracture bbox vs cell bbox
                if (!bboxOverlap(fmin, fmax, cell.bbox_min, cell.bbox_max)) {
                    continue;
                }

                // 精确裁剪：fracture quad vs arbitrary hexahedron
                std::vector<Point3> poly = clipFractureBox(frac, cell, faces);
                if (poly.size() < 3) continue;

                double area = polygonArea(poly);
                if (area <= 1e-6) continue;

                Segment seg;
                seg.id = seg_id_counter++;
                seg.frac_id = frac.id;
                seg.cell_id = cell_idx;
                seg.area = area;
                seg.center = polygonCenter(poly);
                seg.normal = normal;
                seg.aperture = frac.aperture;
                seg.perm = frac.perm;
                seg.poly = poly;

                // 阶段C已重构：基于真实 FaceGeom 的局部 face-trace 几何
                fillSegmentFaceGeom(seg, poly, cell, faces);

                // 阶段D2：统一用一般六面体版 Tmf 计算
                seg.T_mf = computeMatrixFractureTransmissibility(cell, seg, 2, 2, 2);

                segments.push_back(seg);
            }
        }

        n_frac_nodes = (int)segments.size();
        n_total = n_matrix + n_frac_nodes;
        std::cout << "Generated " << n_frac_nodes << " fracture segments." << std::endl;
    }



    // --- 优化点：构建连接时同时构建邻接表 ---
    void buildConnections() {
        connections.clear();
        adj.assign(n_total, std::vector<Neighbor>());

        auto add_conn = [&](int u, int v, double T, int type) {
            if (T <= EPSILON) return;
            connections.push_back({u, v, T, type});
            int conn_idx = (int)connections.size() - 1;
            adj[u].push_back({v, T, conn_idx});
            adj[v].push_back({u, T, conn_idx});
        };

        // =========================================================
        // 1. Matrix-Matrix —— face-based TPFA（阶段D1）
        // =========================================================
        for (const auto& f : faces) {
            if (f.owner < 0 || f.neighbor < 0) continue;

            int u = f.owner;
            int v = f.neighbor;

            double Tmm = computeMMTransmissibilityTPFA(cells[u], cells[v], f);
            if (Tmm > EPSILON) {
                add_conn(u, v, Tmm, 0);
            }
        }

        // =========================================================
        // 2. Matrix-Fracture —— 继续使用阶段D2的 T_mf
        // =========================================================
        for (int s = 0; s < n_frac_nodes; ++s) {
            int u = segments[s].cell_id;
            int v = n_matrix + s;
            add_conn(u, v, segments[s].T_mf, 1);
        }

        // =========================================================
        // 3. Fracture-Fracture (Intra) —— 阶段D3重写版
        // 同一条大裂缝在相邻宿主 cell 中的两个 segment，
        // 通过 shared global face id 来识别共享界面
        // =========================================================
        std::map<int, std::vector<int>> frac_seg_map;
        for (int s = 0; s < n_frac_nodes; ++s) {
            frac_seg_map[segments[s].frac_id].push_back(s);
        }

        for (auto& entry : frac_seg_map) {
            const std::vector<int>& segs = entry.second;

            for (size_t i = 0; i < segs.size(); ++i) {
                for (size_t j = i + 1; j < segs.size(); ++j) {
                    int s1 = segs[i];
                    int s2 = segs[j];

                    int c1 = segments[s1].cell_id;
                    int c2 = segments[s2].cell_id;
                    if (c1 == c2) continue; // 同一宿主 cell 内不属于 Intra

                    int lf1 = -1, lf2 = -1, shared_fid = -1;
                    if (!getSharedFacePairByFaceIds(cells[c1], cells[c2], lf1, lf2, shared_fid)) {
                        continue;
                    }

                    // 理论上共享面应是内部面
                    if (shared_fid < 0) continue;
                    const FaceGeom& fg = faces[shared_fid];
                    if (fg.owner < 0 || fg.neighbor < 0) continue;

                    // 两个 segment 都必须在共享 face 上有有效 trace
                    double ell1 = segments[s1].face_trace_len[lf1];
                    double ell2 = segments[s2].face_trace_len[lf2];
                    if (ell1 <= EPSILON || ell2 <= EPSILON) continue;

                    // 数值上允许略有差异，取较小值更保守
                    double ell = std::min(ell1, ell2);
                    if (ell <= EPSILON) continue;

                    // 各自中心到共享界面 trace 的距离
                    double L1 = segments[s1].face_center_dist[lf1];
                    double L2 = segments[s2].face_center_dist[lf2];
                    if (L1 <= EPSILON || L2 <= EPSILON) continue;

                    // 过流面积 A = aperture * shared-trace-length
                    double Af1 = segments[s1].aperture * ell;
                    double Af2 = segments[s2].aperture * ell;
                    if (Af1 <= EPSILON || Af2 <= EPSILON) continue;

                    // 两侧半传导率
                    double tau1 = segments[s1].perm * Af1 / L1;
                    double tau2 = segments[s2].perm * Af2 / L2;
                    if (tau1 <= EPSILON || tau2 <= EPSILON) continue;

                    // 调和组合
                    double Tff = (tau1 * tau2) / std::max(EPSILON, tau1 + tau2);

                    if (Tff > EPSILON) {
                        add_conn(n_matrix + s1, n_matrix + s2, Tff, 2);
                    }
                }
            }
        }

        // =========================================================
        // 4. Fracture-Fracture (Inter/Cross) —— 暂时保持现有逻辑
        // =========================================================
        std::map<int, std::vector<int>> cell_seg_map;
        for (int s = 0; s < n_frac_nodes; ++s) {
            cell_seg_map[segments[s].cell_id].push_back(s);
        }

        for (auto& entry : cell_seg_map) {
            const std::vector<int>& segs = entry.second;
            if (segs.size() < 2) continue;

            for (size_t i = 0; i < segs.size(); ++i) {
                for (size_t j = i + 1; j < segs.size(); ++j) {
                    int s1 = segs[i];
                    int s2 = segs[j];

                    if (segments[s1].frac_id == segments[s2].frac_id) continue;

                    double ell_int = 0.0;
                    double L1 = 0.0;
                    double L2 = 0.0;

                    bool ok = computeCrossGeom(segments[s1], segments[s2], ell_int, L1, L2);
                    if (!ok) continue;

                    if (ell_int <= EPSILON) continue;

                    L1 = std::max(L1, 1e-10);
                    L2 = std::max(L2, 1e-10);

                    double A1 = segments[s1].aperture * ell_int;
                    double A2 = segments[s2].aperture * ell_int;
                    if (A1 <= EPSILON || A2 <= EPSILON) continue;

                    double tau1 = segments[s1].perm * A1 / L1;
                    double tau2 = segments[s2].perm * A2 / L2;
                    if (tau1 <= EPSILON || tau2 <= EPSILON) continue;

                    double T_cross = (tau1 * tau2) / std::max(EPSILON, tau1 + tau2);

                    if (T_cross > EPSILON) {
                        add_conn(n_matrix + s1, n_matrix + s2, T_cross, 2);
                    }
                }
            }
        }

        std::cout << "Built " << connections.size() << " connections." << std::endl;
    }

    void setupWells() {
        wells.clear();
        well_map.clear();

        // 只收集人工裂缝，不再依赖固定 ID 范围
        std::vector<int> target_fracs;
        std::map<int, Point3> frac_center_map;

        for (const auto& f : fractures) {
            if (f.is_hydraulic) {
                target_fracs.push_back(f.id);
                frac_center_map[f.id] = fractureCenter(f);
            }
        }

        std::sort(target_fracs.begin(), target_fracs.end());

        for (int fid : target_fracs) {
            std::vector<int> cands;
            for (int s = 0; s < n_frac_nodes; ++s) {
                if (segments[s].frac_id == fid) {
                    cands.push_back(s);
                }
            }

            if (cands.empty()) continue;

            // 选最接近该裂缝几何中心的 segment
            int best_s = -1;
            double best_d2 = 1e100;

            Point3 fc = frac_center_map[fid];
            for (int s : cands) {
                Point3 d = segments[s].center - fc;
                double d2 = d.dot(d);
                if (d2 < best_d2) {
                    best_d2 = d2;
                    best_s = s;
                }
            }

            if (best_s != -1) {
                double rw = 0.05;

                double re = computeSegmentEquivalentRadius(segments[best_s], rw);
                double kf = segments[best_s].perm;
                double b  = segments[best_s].aperture;

                double denom = std::log(re / rw);
                if (denom <= EPSILON) continue;

                Well w;
                w.target_node_idx = n_matrix + best_s;
                w.WI = 2.0 * PI * kf * b / denom;
                w.P_bhp = 50.0;

                if (!std::isfinite(w.WI) || w.WI <= EPSILON) continue;

                wells.push_back(w);
                well_map[w.target_node_idx] = (int)wells.size() - 1;
            }
        }

        std::cout << "Setup " << wells.size() << " well connections." << std::endl;
    }

    // --- 物理计算辅助 ---
    void initState() {
        states.resize(n_total); states_prev.resize(n_total);
        for(int i=0; i<n_total; ++i) {
            states[i].P = 800.0; states[i].Sw = 0.05; states[i].Sg = 0.9;
            states_prev[i] = states[i];
        }
    }

    template <typename T>
    PropertiesT<T> getProps(const StateT<T>& s) const {
        PropertiesT<T> p;
        T dummy;
        calcPVT(s.P, p.Bw, p.Bo, p.Bg, dummy, dummy, dummy);
        calcRelPerm(s.Sw, s.Sg, p.krw, p.kro, p.krg);
        p.lw = p.krw / (g_props.mu_w * p.Bw);
        p.lo = p.kro / (g_props.mu_o * p.Bo);
        p.lg = p.krg / (g_props.mu_g * p.Bg);
        return p;
    }

    Eigen::Matrix<AD3, 3, 1> computeAccumulation_AD(double dt, const State& s_old_val, const StateAD3& s_new, const PropertiesT<AD3>& p_new, double vol, double phi) const {
        StateAD3 s_old;
        s_old.P.value() = s_old_val.P;    s_old.P.derivatives().setZero();
        s_old.Sw.value() = s_old_val.Sw;  s_old.Sw.derivatives().setZero();
        s_old.Sg.value() = s_old_val.Sg;  s_old.Sg.derivatives().setZero();

        PropertiesT<AD3> p_old = getProps(s_old);

        AD3 accum = vol * phi / dt;
        Eigen::Matrix<AD3, 3, 1> R;
        R(0) = accum * (s_new.Sw / p_new.Bw - s_old.Sw / p_old.Bw);
        R(1) = accum * ((AD3(1.0) - s_new.Sw - s_new.Sg) / p_new.Bo - (AD3(1.0) - s_old.Sw - s_old.Sg) / p_old.Bo);
        R(2) = accum * (s_new.Sg / p_new.Bg - s_old.Sg / p_old.Bg);
        
        return R;
    }

    Eigen::Matrix<AD3, 3, 1> computeWell_AD(const Well& w, const StateAD3& s_new, const PropertiesT<AD3>& pu) const {
        Eigen::Matrix<AD3, 3, 1> R;
        R(0) = AD3(0.0); R(1) = AD3(0.0); R(2) = AD3(0.0);
        
        AD3 dP = s_new.P - w.P_bhp;
        if (dP.value() > 0.0) {
            R(0) = w.WI * pu.lw * dP;
            R(1) = w.WI * pu.lo * dP;
            R(2) = w.WI * pu.lg * dP;
        }
        return R;
    }

    struct FluxAD {
        double val[3];
        Eigen::Vector3d d_du[3];
        Eigen::Vector3d d_dv[3];
    };

    FluxAD computeFlux_FastAD(double T_trans, const StateAD3& su, const StateAD3& sv, const PropertiesT<AD3>& pu, const PropertiesT<AD3>& pv) const {
        FluxAD res;
        double dP_val = su.P.value() - sv.P.value();
        bool u_is_upwind = (dP_val >= 0.0);

        AD3 dP_u = su.P - sv.P.value(); 
        AD3 dP_v = su.P.value() - sv.P; 

        if (u_is_upwind) {
            AD3 Fu0 = T_trans * pu.lw * dP_u;
            AD3 Fu1 = T_trans * pu.lo * dP_u;
            AD3 Fu2 = T_trans * pu.lg * dP_u;

            res.val[0] = Fu0.value(); res.d_du[0] = Fu0.derivatives();
            res.val[1] = Fu1.value(); res.d_du[1] = Fu1.derivatives();
            res.val[2] = Fu2.value(); res.d_du[2] = Fu2.derivatives();

            res.d_dv[0] = T_trans * pu.lw.value() * dP_v.derivatives();
            res.d_dv[1] = T_trans * pu.lo.value() * dP_v.derivatives();
            res.d_dv[2] = T_trans * pu.lg.value() * dP_v.derivatives();
        } else {
            AD3 Fv0 = T_trans * pv.lw * dP_v;
            AD3 Fv1 = T_trans * pv.lo * dP_v;
            AD3 Fv2 = T_trans * pv.lg * dP_v;

            res.val[0] = Fv0.value(); res.d_dv[0] = Fv0.derivatives();
            res.val[1] = Fv1.value(); res.d_dv[1] = Fv1.derivatives();
            res.val[2] = Fv2.value(); res.d_dv[2] = Fv2.derivatives();

            res.d_du[0] = T_trans * pv.lw.value() * dP_u.derivatives();
            res.d_du[1] = T_trans * pv.lo.value() * dP_u.derivatives();
            res.d_du[2] = T_trans * pv.lg.value() * dP_u.derivatives();
        }
        return res;
    }

    void buildJacobianPattern() {
        J.resize(3*n_total, 3*n_total);
        std::vector<Triplet<double>> trips;
        trips.reserve(n_total * 9 + connections.size() * 18);

        for(int i=0; i<n_total; ++i) {
            for(int eq=0; eq<3; ++eq) {
                for(int var=0; var<3; ++var) {
                    trips.emplace_back(3*i+eq, 3*i+var, 0.0);
                }
            }
        }
        for(const auto& conn : connections) {
            int u = conn.u, v = conn.v;
            for(int eq=0; eq<3; ++eq) {
                for(int var=0; var<3; ++var) {
                    trips.emplace_back(3*u+eq, 3*v+var, 0.0);
                    trips.emplace_back(3*v+eq, 3*u+var, 0.0);
                }
            }
        }
        
        J.setFromTriplets(trips.begin(), trips.end());
        J.makeCompressed();

        auto get_val_idx = [&](int r, int c) -> int {
            int col_start = J.outerIndexPtr()[c];
            int col_end   = J.outerIndexPtr()[c+1];
            for (int k = col_start; k < col_end; ++k) {
                if (J.innerIndexPtr()[k] == r) return k;
            }
            return -1;
        };

        cell_J_idx.resize(n_total);
        for(int i=0; i<n_total; ++i) {
            for(int eq=0; eq<3; ++eq) {
                for(int var=0; var<3; ++var) {
                    cell_J_idx[i].diag[eq][var] = get_val_idx(3*i+eq, 3*i+var);
                }
            }
        }

        conn_J_idx.resize(connections.size());
        for(size_t i=0; i<connections.size(); ++i) {
            int u = connections[i].u;
            int v = connections[i].v;
            for(int eq=0; eq<3; ++eq) {
                for(int var=0; var<3; ++var) {
                    conn_J_idx[i].off_uv[eq][var] = get_val_idx(3*u+eq, 3*v+var);
                    conn_J_idx[i].off_vu[eq][var] = get_val_idx(3*v+eq, 3*u+var);
                }
            }
        }
        std::cout << "Jacobian static pattern built. Nonzeros: " << J.nonZeros() << std::endl;
    }

    // =========================================================================
    // 替换原有的 solveStep，现在返回 bool 表示是否收敛成功
    // =========================================================================
    bool solveStep(double dt, double& step_oil, double& step_water, double& step_gas, int& iter_out) {
        int max_iter = 15;      // 增加最大迭代次数
        double tol = 1e-3;      // 稍微放宽一点容差，防止在数值噪音处死循环
        
        // 备份初始状态，以便迭代失败时恢复
        std::vector<State> states_backup = states;
        
        // 临时累积量，只有收敛才加到总量里
        double curr_oil = 0, curr_water = 0, curr_gas = 0;

        for(int iter=0; iter<max_iter; ++iter) {
            iter_out = iter + 1; // [Modified: 记录当前迭代步数]

            std::fill(J.valuePtr(), J.valuePtr() + J.nonZeros(), 0.0);
            VectorXd Rg(3*n_total);
            Rg.setZero();
            
            std::vector<StateAD3> states_ad(n_total);
            std::vector<PropertiesT<AD3>> props_ad(n_total);

            for (int i = 0; i < n_total; ++i) {
                states_ad[i].P.value() = states[i].P;    states_ad[i].P.derivatives()  = Eigen::Vector3d::Unit(0);
                states_ad[i].Sw.value() = states[i].Sw;  states_ad[i].Sw.derivatives() = Eigen::Vector3d::Unit(1);
                states_ad[i].Sg.value() = states[i].Sg;  states_ad[i].Sg.derivatives() = Eigen::Vector3d::Unit(2);
                props_ad[i] = getProps(states_ad[i]);
            }

            for (int i=0; i<n_total; ++i) {
                double vol = (i < n_matrix) ? cells[i].vol : (segments[i - n_matrix].area * segments[i - n_matrix].aperture);
                double phi = (i < n_matrix) ? cells[i].phi : 1.0;
                
                auto R_acc = computeAccumulation_AD(dt, states_prev[i], states_ad[i], props_ad[i], vol, phi);
                
                if (well_map.count(i)) {
                    auto R_well = computeWell_AD(wells[well_map[i]], states_ad[i], props_ad[i]);
                    R_acc(0) += R_well(0);
                    R_acc(1) += R_well(1);
                    R_acc(2) += R_well(2);
                }

                for (int eq = 0; eq < 3; ++eq) {
                    Rg(3*i + eq) += R_acc(eq).value();
                    Deriv3 derivs = R_acc(eq).derivatives();
                    for(int var = 0; var < 3; ++var) {
                        J.valuePtr()[cell_J_idx[i].diag[eq][var]] += derivs(var);
                    }
                }
            }

            for (size_t c_idx = 0; c_idx < connections.size(); ++c_idx) {
                const auto& conn = connections[c_idx];
                int u = conn.u;
                int v = conn.v;
                
                auto F_uv = computeFlux_FastAD(conn.T, states_ad[u], states_ad[v], props_ad[u], props_ad[v]);
                
                for (int eq = 0; eq < 3; ++eq) {
                    Rg(3*u + eq) += F_uv.val[eq];
                    Rg(3*v + eq) -= F_uv.val[eq];
                    
                    for (int var = 0; var < 3; ++var) {
                        double dF_dXu = F_uv.d_du[eq](var);
                        double dF_dXv = F_uv.d_dv[eq](var);
                        
                        J.valuePtr()[cell_J_idx[u].diag[eq][var]] += dF_dXu;
                        J.valuePtr()[conn_J_idx[c_idx].off_vu[eq][var]] -= dF_dXu;
                        
                        J.valuePtr()[conn_J_idx[c_idx].off_uv[eq][var]] += dF_dXv;
                        J.valuePtr()[cell_J_idx[v].diag[eq][var]] -= dF_dXv;
                    }
                }
            }

            double max_resid = Rg.lpNorm<Infinity>();

            if(max_resid < tol) {
                // 收敛成功！计算本步产量
                for(const auto& w : wells) {
                    int u = w.target_node_idx;
                    double dP = states[u].P - w.P_bhp;
                    if(dP > 0) {
                        PropertiesT<double> p = getProps(states[u]);
                        step_water += w.WI * p.lw * dP * dt;
                        step_oil   += w.WI * p.lo * dP * dt;
                        step_gas   += w.WI * p.lg * dP * dt;
                    }
                }
                return true; // 成功
            }

            BiCGSTAB<SparseMatrix<double>, IncompleteLUT<double>> solver;
            solver.preconditioner().setDroptol(1e-5);
            solver.preconditioner().setFillfactor(40);
            solver.setTolerance(1e-5);
            solver.setMaxIterations(500);
            
            solver.compute(J);
            if (solver.info() != Success) {
                states = states_backup; 
                return false;
            }
            
            VectorXd delta = solver.solve(-Rg);
            
            if (solver.info() != Success) {
                states = states_backup; 
                return false;
            }
            
            // 5. 更新与阻尼 (Damping) - 关键修改！
            // [Modified: 引入牛顿回溯 Line Search 以确保残差下降]
            double alpha = 1.0; 
            bool accepted = false;
            double norm_old = Rg.norm();
            std::vector<State> states_before_ls = states;

            for(int ls = 0; ls < 3; ++ls) {
                double damping = 1.0; 
                double max_delta_P = 0;
                double max_delta_S = 0;
                for(int i=0; i<n_total; ++i) {
                    max_delta_P = std::max(max_delta_P, std::abs(delta(3*i+0) * alpha));
                    max_delta_S = std::max(max_delta_S, std::abs(delta(3*i+1) * alpha));
                    max_delta_S = std::max(max_delta_S, std::abs(delta(3*i+2) * alpha));
                }

                if(max_delta_P > 20.0) damping = std::min(damping, 20.0 / max_delta_P);
                if(max_delta_S > 0.1)  damping = std::min(damping, 0.1 / max_delta_S);

                for(int i=0; i<n_total; ++i) {
                    states[i].P  = states_before_ls[i].P  + delta(3*i+0) * damping * alpha;
                    states[i].Sw = states_before_ls[i].Sw + delta(3*i+1) * damping * alpha;
                    states[i].Sg = states_before_ls[i].Sg + delta(3*i+2) * damping * alpha;
                    states[i].P = std::max(1.0, states[i].P); 
                    states[i].Sw = std::max(0.0, std::min(1.0, states[i].Sw));
                    states[i].Sg = std::max(0.0, std::min(1.0, states[i].Sg));
                    if(states[i].Sw + states[i].Sg > 1.0) {
                        double sum = states[i].Sw + states[i].Sg;
                        states[i].Sw /= sum; states[i].Sg /= sum;
                    }
                }

                double norm_new = 0;
                std::vector<StateAD3> states_ad_ls(n_total);
                std::vector<PropertiesT<AD3>> props_ad_ls(n_total);
                for(int i=0; i<n_total; ++i) {
                    states_ad_ls[i].P.value() = states[i].P;    states_ad_ls[i].P.derivatives().setZero();
                    states_ad_ls[i].Sw.value() = states[i].Sw;  states_ad_ls[i].Sw.derivatives().setZero();
                    states_ad_ls[i].Sg.value() = states[i].Sg;  states_ad_ls[i].Sg.derivatives().setZero();
                    props_ad_ls[i] = getProps(states_ad_ls[i]);
                }
                VectorXd Rg_ls(3*n_total);
                Rg_ls.setZero();
                for (int i = 0; i < n_total; ++i) {
                    double vol = (i < n_matrix) ? cells[i].vol : (segments[i - n_matrix].area * segments[i - n_matrix].aperture);
                    double phi = (i < n_matrix) ? cells[i].phi : 1.0;
                    auto R_acc = computeAccumulation_AD(dt, states_prev[i], states_ad_ls[i], props_ad_ls[i], vol, phi);
                    if (well_map.count(i)) {
                        auto R_well = computeWell_AD(wells[well_map[i]], states_ad_ls[i], props_ad_ls[i]);
                        R_acc(0) += R_well(0); R_acc(1) += R_well(1); R_acc(2) += R_well(2);
                    }
                    for (int eq = 0; eq < 3; ++eq) Rg_ls(3*i + eq) += R_acc(eq).value();
                }
                for (size_t c_idx = 0; c_idx < connections.size(); ++c_idx) {
                    const auto& conn = connections[c_idx];
                    int u = conn.u; int v = conn.v;
                    auto F_uv = computeFlux_FastAD(conn.T, states_ad_ls[u], states_ad_ls[v], props_ad_ls[u], props_ad_ls[v]);
                    for (int eq = 0; eq < 3; ++eq) {
                        Rg_ls(3*u + eq) += F_uv.val[eq];
                        Rg_ls(3*v + eq) -= F_uv.val[eq];
                    }
                }
                norm_new = Rg_ls.norm();

                if(norm_new < norm_old || iter == 0) {
                    accepted = true;
                    break;
                } else {
                    alpha *= 0.5;
                    states = states_before_ls;
                }
            }

            if(!accepted && iter > 0) {
                states = states_backup;
                return false;
            }
        }
        
        // 达到最大迭代次数仍未收敛
        states = states_backup; // 恢复状态
        return false;
    }

    // =========================================================================
    // 新增功能：导出静态网格和裂缝几何信息，供 OpenCV 可视化使用
    // =========================================================================
    void exportStaticGeometry() {
        // 1. 保留旧版网格基础信息，兼容现有脚本
        std::ofstream gridFile("grid_info.csv");
        gridFile << Nx << "," << Ny << "," << Nz << ","
                << Lx << "," << Ly << "," << Lz << ","
                << dx << "," << dy << "," << dz << "\n";
        gridFile.close();

        // 2. 新增：导出 cell 几何
        std::ofstream cellFile("cell_geometry.csv");
        cellFile
            << "cell_id,ix,iy,iz,"
            << "cx,cy,cz,vol,"
            << "bbox_min_x,bbox_min_y,bbox_min_z,"
            << "bbox_max_x,bbox_max_y,bbox_max_z,";

        for (int n = 0; n < 8; ++n) {
            cellFile << "c" << n << "_x,c" << n << "_y,c" << n << "_z,";
        }
        cellFile << "face0,face1,face2,face3,face4,face5\n";

        for (const auto& c : cells) {
            cellFile
                << c.id << ","
                << c.ix << "," << c.iy << "," << c.iz << ","
                << c.center.x << "," << c.center.y << "," << c.center.z << ","
                << c.vol << ","
                << c.bbox_min.x << "," << c.bbox_min.y << "," << c.bbox_min.z << ","
                << c.bbox_max.x << "," << c.bbox_max.y << "," << c.bbox_max.z << ",";

            for (int n = 0; n < 8; ++n) {
                cellFile << c.corners[n].x << "," << c.corners[n].y << "," << c.corners[n].z << ",";
            }

            cellFile
                << c.face_ids[0] << "," << c.face_ids[1] << "," << c.face_ids[2] << ","
                << c.face_ids[3] << "," << c.face_ids[4] << "," << c.face_ids[5] << "\n";
        }
        cellFile.close();

        // 3. 新增：导出 face 几何
        std::ofstream faceFile("face_geometry.csv");
        faceFile
            << "face_id,owner,neighbor,local_owner_face,local_neighbor_face,"
            << "cx,cy,cz,nx,ny,nz,area,"
            << "bbox_min_x,bbox_min_y,bbox_min_z,"
            << "bbox_max_x,bbox_max_y,bbox_max_z,"
            << "v0_x,v0_y,v0_z,v1_x,v1_y,v1_z,v2_x,v2_y,v2_z,v3_x,v3_y,v3_z\n";

        for (const auto& f : faces) {
            faceFile
                << f.id << ","
                << f.owner << ","
                << f.neighbor << ","
                << f.local_owner_face << ","
                << f.local_neighbor_face << ","
                << f.center.x << "," << f.center.y << "," << f.center.z << ","
                << f.normal.x << "," << f.normal.y << "," << f.normal.z << ","
                << f.area << ","
                << f.bbox_min.x << "," << f.bbox_min.y << "," << f.bbox_min.z << ","
                << f.bbox_max.x << "," << f.bbox_max.y << "," << f.bbox_max.z << ",";

            for (int i = 0; i < 4; ++i) {
                faceFile << f.vertices[i].x << "," << f.vertices[i].y << "," << f.vertices[i].z;
                if (i < 3) faceFile << ",";
            }
            faceFile << "\n";
        }
        faceFile.close();

        // 4. 保留旧版裂缝几何导出
        std::ofstream fracFile("fracture_geometry.csv");
        fracFile << "id,x0,y0,z0,x1,y1,z1,x2,y2,z2,x3,y3,z3\n";
        for (const auto& f : fractures) {
            fracFile << f.id;
            for (int i = 0; i < 4; ++i) {
                fracFile << ","
                        << f.vertices[i].x << ","
                        << f.vertices[i].y << ","
                        << f.vertices[i].z;
            }
            fracFile << "\n";
        }
        fracFile.close();

        std::cout << "Geometry exported: grid_info.csv, cell_geometry.csv, face_geometry.csv, fracture_geometry.csv" << std::endl;
    }

    // 导出井信息 (well_info.csv)
    void exportWells() {
        std::ofstream wellFile("well_info.csv");
        // 输出格式: ID, 节点索引, 类型(基质/裂缝), X, Y, Z, 井指数,井底流压
        wellFile << "well_id,node_idx,type,x,y,z,WI,P_bhp\n";
        
        for(size_t i=0; i<wells.size(); ++i) {
            int u = wells[i].target_node_idx;
            double x, y, z;
            std::string type;

            if (u < n_matrix) {
                // 如果井在基质网格中
                x = cells[u].center.x;
                y = cells[u].center.y;
                z = cells[u].center.z;
                type = "Matrix";
            } else {
                // 如果井在裂缝段中 (你的代码主要是这种情况)
                int seg_idx = u - n_matrix;
                x = segments[seg_idx].center.x;
                y = segments[seg_idx].center.y;
                z = segments[seg_idx].center.z;
                type = "Fracture";
            }
            
            wellFile << i << "," << u << "," << type << ","
                     << x << "," << y << "," << z << ","
                     << wells[i].WI << "," << wells[i].P_bhp << "\n";
        }
        wellFile.close();
        std::cout << "Wells exported: well_info.csv" << std::endl;
    }

    // =========================================================================
    // 替换原有的 run，实现自动时间步长控制 (Auto Time-Stepping)
    // =========================================================================
    void run(double total_time_days) {
        std::ofstream file("output_sim.csv");
        file << "Time,CumOil,CumWater,CumGas,AvgPressure,DT\n";

        double current_time = 0.0;
        double dt = 0.001; // 初始步长设得非常小！ (1e-3 天)
        double dt_min = 1e-6;
        double dt_max = 10.0;
        
        // [Modified: 设定工业级目标迭代次数]
        int target_iter = 6; 
        double tot_oil=0, tot_water=0, tot_gas=0;
        int step_count = 0;

        std::cout << std::fixed << std::setprecision(4);

        while(current_time < total_time_days) {
            step_count++;
            
            // 尝试求解一步
            double step_oil=0, step_water=0, step_gas=0;
            bool success = false;
            int actual_iter = 0; // [Modified: 接收实际迭代次数]

            // 如果这一步加上去超过总时间，就截断 dt
            if(current_time + dt > total_time_days) {
                dt = total_time_days - current_time;
            }

            // 重试循环
            while(!success) {
                if(dt < dt_min) {
                    std::cerr << "Time step too small, simulation failed." << std::endl;
                    return;
                }

                std::cout << "Step " << step_count << " @ T=" << current_time 
                          << " trying dt=" << dt << " ... " << std::flush;
                
                // [Modified: 传入 actual_iter]
                success = solveStep(dt, step_oil, step_water, step_gas, actual_iter);

                if(success) {
                    std::cout << "Converged in " << actual_iter << " iters." << std::endl;
                    // 更新时间
                    current_time += dt;
                    states_prev = states; // 归档历史状态
                    
                    // 累加产量
                    tot_oil += step_oil;
                    tot_water += step_water;
                    tot_gas += step_gas;

                    // 写入日志
                    double avgP = 0;
                    for(int i=0; i<n_matrix; ++i) avgP += states[i].P;
                    avgP /= n_matrix;
                    file << current_time << "," << tot_oil << "," << tot_water << "," << tot_gas << "," << avgP << "," << dt << "\n";
                    file.flush();

                    // [Modified: 基于迭代目标的平滑步长调整逻辑]
                    double fac = std::pow((double)target_iter / (double)std::max(1, actual_iter), 0.5);
                    fac = std::max(0.5, std::min(1.5, fac)); // 限制单步缩放比例
                    dt = std::min(dt_max, dt * fac);
                } else {
                    std::cout << "Failed. Cutting timestep." << std::endl;
                    // 失败，减小步长重试
                    dt *= 0.25; // [Modified: 失败时更果断地切步长]
                }
            }
        }
        
        file.close();
        
        // 输出场
        std::ofstream field("final_field.csv");
        field << "cell_id,x,y,z,P,Sw,Sg\n";
        for(int i=0; i<n_matrix; ++i) {
            field << i << ","<< cells[i].center.x << "," << cells[i].center.y << "," << cells[i].center.z << ","
                  << states[i].P << "," << states[i].Sw << "," << states[i].Sg << "\n";
        }
        field.close();
    }
};

int main() {
    Simulator sim;
    
    // 1. 初始化网格
    std::cout << "Initializing Grid..." << std::endl;
    sim.initGrid();
    
    // 2. 生成裂缝
    std::cout << "Generating Fractures..." << std::endl;
    sim.generateFractures();
    sim.generateHydraulicFractures();
    
    // 3. 计算几何相交 (EDFM)
    std::cout << "Processing Geometry..." << std::endl;
    sim.processGeometry();
    
    // 4. 建立连接关系 (Transmissibility)
    std::cout << "Building Connections..." << std::endl;
    sim.buildConnections();
    
    // 5. 建立井
    std::cout << "Setting up Wells..." << std::endl;
    sim.setupWells();
    sim.exportWells();
    
    // 6. 初始化状态
    sim.initState();
    sim.buildJacobianPattern();

    // --- 【在这里添加调用】 ---
    std::cout << "Exporting Geometry for Visualization..." << std::endl;
    sim.exportStaticGeometry(); 
    // -----------------------
    
    // 7. 运行模拟
    // 100 天，步长 10 天
    std::cout << "Starting Simulation..." << std::endl;
    sim.run(100.0); 
    
    std::cout << "Done. Results saved to CSV." << std::endl;
    }

    // ---------------------------------------------------------------------
    // Python Interface Methods
    // ---------------------------------------------------------------------
    void setGridParameters(int nx, int ny, int nz, double lx, double ly, double lz) {
        Nx = nx; Ny = ny; Nz = nz;
        Lx = lx; Ly = ly; Lz = lz;
        dx = Lx / Nx; dy = Ly / Ny; dz = Lz / Nz;
    }
    
    void setFractureParameters(int num_fractures, double min_length, double max_length,
                               double max_dip, double min_strike, double max_strike,
                               double aperture, double permeability) {
        num_fractures_ = num_fractures;
        min_length_ = min_length;
        max_length_ = max_length;
        max_dip_ = max_dip;
        min_strike_ = min_strike;
        max_strike_ = max_strike;
        aperture_ = aperture;
        perm_f_ = permeability;
    }
    
    void setWellParameters(double well_x, double well_y, double well_z,
                          double well_radius, double well_pressure) {
        well_x_ = well_x;
        well_y_ = well_y;
        well_z_ = well_z;
        well_radius_ = well_radius;
        well_pressure_ = well_pressure;
    }
    
    void setSimulationParameters(double simulation_time, double time_step,
                                double porosity, double perm_x, double perm_y, double perm_z) {
        simulation_time_ = simulation_time;
        time_step_ = time_step;
        porosity_ = porosity;
        perm_x_ = perm_x;
        perm_y_ = perm_y;
        perm_z_ = perm_z;
    }
    
    py::array_t<double> getPressureData() {
        py::array_t<double> result({n_matrix, 4});
        auto r = result.mutable_unchecked<2>();
        for (int i = 0; i < n_matrix; ++i) {
            r(i, 0) = cells[i].center.x;
            r(i, 1) = cells[i].center.y;
            r(i, 2) = cells[i].center.z;
            r(i, 3) = states[i].P;
        }
        return result;
    }
    
    py::array_t<double> getFractureVertices() {
        int num_verts = fractures.size() * 4;
        py::array_t<double> result({num_verts, 4});
        auto r = result.mutable_unchecked<2>();
        int row = 0;
        for (const auto& f : fractures) {
            for (int i = 0; i < 4; ++i) {
                r(row, 0) = f.vertices[i].x;
                r(row, 1) = f.vertices[i].y;
                r(row, 2) = f.vertices[i].z;
                r(row, 3) = (double)f.id;
                row++;
            }
        }
        return result;
    }

    py::array_t<double> getGridLines() {
        int num_lines = n_matrix * 12;
        py::array_t<double> result({num_lines, 6});
        auto r = result.mutable_unchecked<2>();
        int row = 0;
        for (int i = 0; i < n_matrix; ++i) {
            const auto& cell = cells[i];
            // 使用 corners 而非 dx/dy/dz，因为这是角点网格
            // 0-1, 1-2, 2-3, 3-0 (zmin)
            // 4-5, 5-6, 6-7, 7-4 (zmax)
            // 0-4, 1-5, 2-6, 3-7
            auto add_line = [&](const Point3& a, const Point3& b) {
                r(row, 0) = a.x; r(row, 1) = a.y; r(row, 2) = a.z;
                r(row, 3) = b.x; r(row, 4) = b.y; r(row, 5) = b.z;
                row++;
            };
            add_line(cell.corners[0], cell.corners[1]);
            add_line(cell.corners[1], cell.corners[2]);
            add_line(cell.corners[2], cell.corners[3]);
            add_line(cell.corners[3], cell.corners[0]);
            add_line(cell.corners[4], cell.corners[5]);
            add_line(cell.corners[5], cell.corners[6]);
            add_line(cell.corners[6], cell.corners[7]);
            add_line(cell.corners[7], cell.corners[4]);
            add_line(cell.corners[0], cell.corners[4]);
            add_line(cell.corners[1], cell.corners[5]);
            add_line(cell.corners[2], cell.corners[6]);
            add_line(cell.corners[3], cell.corners[7]);
        }
        return result;
    }

    SimulationResult runSimulation() {
        SimulationResult result;
        std::cout << "=== Corner Point Grid Simulation (EDFM) ===" << std::endl;
        std::cout << "Grid: " << Nx << "x" << Ny << "x" << Nz << ", Domain: " << Lx << "x" << Ly << "x" << Lz << std::endl;
        
        std::cout << "Preprocessing..." << std::endl;
        initGrid();
        generateFractures(num_fractures_, min_length_, max_length_, max_dip_, min_strike_, max_strike_, aperture_, perm_f_);
        generateHydraulicFractures();
        processGeometry();
        buildConnections();
        setupWells();
        initState();
        buildJacobianPattern();
        
        std::cout << "Running simulation..." << std::endl;
        run(simulation_time_);
        
        std::cout << "Simulation complete." << std::endl;
        
        result.pressure_field = getPressureData();
        result.fracture_vertices = getFractureVertices();
        
        return result;
    }

private:
    int num_fractures_{60};
    double min_length_{30.0}, max_length_{80.0};
    double max_dip_{PI/3.0};
    double min_strike_{0.0}, max_strike_{PI};
    double aperture_{0.01};
    double perm_f_{1000.0};
    double well_x_{1500.0}, well_y_{150.0}, well_z_{20.0};
    double well_radius_{0.1};
    double well_pressure_{50.0};
    double simulation_time_{100.0};
    double time_step_{1.0};
    double porosity_{0.04};
    double perm_x_{0.005}, perm_y_{0.005}, perm_z_{0.005};
};

// =============================================================================
// Python Module
// =============================================================================

PYBIND11_MODULE(edfm_core_corner, m) {
    py::class_<SimulationResult>(m, "SimulationResult")
        .def_readwrite("pressure_field", &SimulationResult::pressure_field)
        .def_readwrite("temperature_field", &SimulationResult::temperature_field)
        .def_readwrite("stress_field", &SimulationResult::stress_field)
        .def_readwrite("fracture_vertices", &SimulationResult::fracture_vertices)
        .def_readwrite("fracture_cells", &SimulationResult::fracture_cells);
    
    py::class_<Simulator>(m, "EDFMSimulator")
        .def(py::init<>())
        .def("setGridParameters", &Simulator::setGridParameters)
        .def("setFractureParameters", &Simulator::setFractureParameters)
        .def("setWellParameters", &Simulator::setWellParameters)
        .def("setSimulationParameters", &Simulator::setSimulationParameters)
        .def("runSimulation", &Simulator::runSimulation)
        .def("getPressureData", &Simulator::getPressureData)
        .def("getFractureVertices", &Simulator::getFractureVertices)
        .def("getGridLines", &Simulator::getGridLines);
}

int main() {
    Simulator sim;
    
    // 1. 初始化网格
    std::cout << "Initializing Grid..." << std::endl;
    sim.initGrid();
    
    // 2. 生成裂缝
    std::cout << "Generating Fractures..." << std::endl;
    sim.generateFractures();
    sim.generateHydraulicFractures();
    
    // 3. 计算几何相交 (EDFM)
    std::cout << "Processing Geometry..." << std::endl;
    sim.processGeometry();
    
    // 4. 建立连接关系 (Transmissibility)
    std::cout << "Building Connections..." << std::endl;
    sim.buildConnections();
    
    // 5. 建立井
    std::cout << "Setting up Wells..." << std::endl;
    sim.setupWells();
    sim.exportWells();
    
    // 6. 初始化状态
    sim.initState();
    sim.buildJacobianPattern();

    // --- 【在这里添加调用】 ---
    std::cout << "Exporting Geometry for Visualization..." << std::endl;
    sim.exportStaticGeometry(); 
    // -----------------------
    
    // 7. 运行模拟
    // 100 天，步长 10 天
    std::cout << "Starting Simulation..." << std::endl;
    sim.run(100.0); 
    
    std::cout << "Done. Results saved to CSV." << std::endl;
    return 0;
}