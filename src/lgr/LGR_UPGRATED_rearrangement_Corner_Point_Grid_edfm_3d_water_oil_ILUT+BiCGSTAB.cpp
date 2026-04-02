

#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/SparseLU>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/AutoDiff>

using namespace std;
using namespace Eigen;
namespace py = pybind11;

static constexpr double EPS = 1e-12;
static constexpr double PI  = 3.14159265358979323846;

enum FaceID {
    XM = 0, XP = 1,
    YM = 2, YP = 3,
    ZM = 4, ZP = 5
};

struct Point3 {
    double x{}, y{}, z{};
    Point3 operator+(const Point3& o) const { return {x+o.x, y+o.y, z+o.z}; }
    Point3 operator-(const Point3& o) const { return {x-o.x, y-o.y, z-o.z}; }
    Point3 operator*(double s) const { return {x*s, y*s, z*s}; }
    Point3 operator/(double s) const { return {x/s, y/s, z/s}; }
    double dot(const Point3& o) const { return x*o.x + y*o.y + z*o.z; }
    Point3 cross(const Point3& o) const {
        return {y*o.z - z*o.y, z*o.x - x*o.z, x*o.y - y*o.x};
    }
    double norm() const { return std::sqrt(x*x + y*y + z*z); }
};

static inline double clamp01(double v) { return std::max(0.0, std::min(1.0, v)); }
static inline double clampd(double v, double a, double b) { return std::max(a, std::min(b, v)); }

struct AABB {
    Point3 mn, mx;
};

struct Pillar {
    Point3 top{0,0,0};
    Point3 bot{0,0,0};
    bool has_top{false};
    bool has_bot{false};
};

struct CoordRow {
    int i{0};
    int j{0};
    bool is_top{false};
    Point3 p{0,0,0};
};

struct ZCornRow {
    int i{0};
    int j{0};
    int k{0};
    std::array<double, 8> z{{0,0,0,0,0,0,0,0}};
};

static std::string stripUTF8BOM(const std::string& s) {
    if (s.size() >= 3 &&
        (unsigned char)s[0] == 0xEF &&
        (unsigned char)s[1] == 0xBB &&
        (unsigned char)s[2] == 0xBF) {
        return s.substr(3);
    }
    return s;
}

static std::string trim(const std::string& s) {
    std::string t = stripUTF8BOM(s);
    size_t b = 0, e = t.size();
    while (b < e && std::isspace((unsigned char)t[b])) ++b;
    while (e > b && std::isspace((unsigned char)t[e-1])) --e;
    return t.substr(b, e - b);
}

static std::vector<std::string> splitCSVSimple(const std::string& line) {
    std::vector<std::string> cols;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ',')) cols.push_back(trim(item));
    if (!line.empty() && line.back() == ',') cols.push_back("");
    return cols;
}

static bool parseIntStrict(const std::string& s, int& v) {
    try {
        std::string t = trim(s);
        size_t pos = 0;
        v = std::stoi(t, &pos);
        return pos == t.size();
    } catch (...) {
        return false;
    }
}

static bool parseDoubleStrict(const std::string& s, double& v) {
    try {
        std::string t = trim(s);
        size_t pos = 0;
        v = std::stod(t, &pos);
        return pos == t.size();
    } catch (...) {
        return false;
    }
}

static bool isTopMarker(const std::string& s) {
    std::string t = trim(s);
    if (t == "\xE9\xA1\xB6") return true; // UTF-8 for "顶"
    std::string low = t;
    for (char& ch : low) ch = (char)std::tolower((unsigned char)ch);
    return low == "top";
}

static bool isBottomMarker(const std::string& s) {
    std::string t = trim(s);
    if (t == "\xE5\xBA\x95") return true; // UTF-8 for "底"
    std::string low = t;
    for (char& ch : low) ch = (char)std::tolower((unsigned char)ch);
    return low == "bottom";
}

static int flatPillarIndex(int i, int j, int npx) {
    return j * npx + i;
}

static int flatCellIndex(int i, int j, int k, int Nx, int Ny) {
    return k * Nx * Ny + j * Nx + i;
}

struct FaceGeom {
    int id{-1};
    int owner{-1};
    int neighbor{-1};
    int local_owner_face{-1};
    int local_neighbor_face{-1};
    std::array<Point3, 4> vertices{};
    Point3 center{0,0,0};
    Point3 normal{0,0,0};
    double area{0.0};
    Point3 bbox_min{0,0,0};
    Point3 bbox_max{0,0,0};
};

struct Fracture {
    int id{-1};
    Point3 vertices[4];
    double aperture{0.01};
    double perm{10000.0};
    bool is_hydraulic{false};
};

struct ParentCell {
    int parent_id{-1};
    int ix{}, iy{}, iz{};
    Point3 center{0,0,0};
    double dx{}, dy{}, dz{}, vol{0.0};
    double phi{0.0};
    double K[3]{};
    double depth{0.0};
    std::array<Point3, 8> corners{};
    std::array<int, 6> face_ids{{-1,-1,-1,-1,-1,-1}};
    Point3 bbox_min{0,0,0};
    Point3 bbox_max{0,0,0};
    bool refined{false};
    int leaf_base{-1};
    uint16_t Nrx{1}, Nry{1}, Nrz{1};
};

struct LeafCell {
    int leaf_id{-1};
    int parent_id{-1};
    uint16_t lix{0}, liy{0}, liz{0};
    Point3 center{0,0,0};
    double dx{}, dy{}, dz{}, vol{0.0};
    double phi{0.0};
    double K[3]{};
    double depth{0.0};
    std::array<Point3, 8> corners{};
    std::array<int, 6> face_ids{{-1,-1,-1,-1,-1,-1}};
    std::array<std::vector<int>, 6> subface_ids{};
    Point3 bbox_min{0,0,0};
    Point3 bbox_max{0,0,0};
};

struct Segment {
    int id{};
    int frac_id{};
    int matrix_leaf_id{};
    int parent_id{-1};
    double area{0.0};
    Point3 center{0,0,0};
    Point3 normal{0,0,0};
    double aperture{0.01};
    double perm{10000.0};
    double T_mf{0.0};
    double face_trace_len[6]{0,0,0,0,0,0};
    double face_center_dist[6]{0,0,0,0,0,0};
    std::unordered_map<int, double> subface_trace_len;
    std::unordered_map<int, double> subface_center_dist;
    std::vector<Point3> poly;
};

struct Connection {
    int u{}, v{};
    double T{0.0};
    int type{0};
};

struct ParentFacePatch {
    int leaf_id{-1};
    int leaf_local_face{-1};
    double s0{0.0}, s1{1.0};
    double t0{0.0}, t1{1.0};
    std::array<Point3, 4> quad{};
};

struct ParentFaceCoverage {
    std::array<std::vector<ParentFacePatch>, 6> patches;
};

static inline AABB makeAABBFromCenterSize(const Point3& c, double dx, double dy, double dz) {
    return {{c.x - dx*0.5, c.y - dy*0.5, c.z - dz*0.5}, {c.x + dx*0.5, c.y + dy*0.5, c.z + dz*0.5}};
}

static inline AABB expandAABB(const AABB& b, double r) {
    return {{b.mn.x - r, b.mn.y - r, b.mn.z - r}, {b.mx.x + r, b.mx.y + r, b.mx.z + r}};
}

static inline bool aabbIntersect(const AABB& a, const AABB& b) {
    return !(a.mx.x < b.mn.x || a.mn.x > b.mx.x ||
             a.mx.y < b.mn.y || a.mn.y > b.mx.y ||
             a.mx.z < b.mn.z || a.mn.z > b.mx.z);
}

static inline double distAABB_AABB(const AABB& a, const AABB& b) {
    double dx = 0.0;
    if (a.mx.x < b.mn.x) dx = b.mn.x - a.mx.x;
    else if (b.mx.x < a.mn.x) dx = a.mn.x - b.mx.x;
    double dy = 0.0;
    if (a.mx.y < b.mn.y) dy = b.mn.y - a.mx.y;
    else if (b.mx.y < a.mn.y) dy = a.mn.y - b.mx.y;
    double dz = 0.0;
    if (a.mx.z < b.mn.z) dz = b.mn.z - a.mx.z;
    else if (b.mx.z < a.mn.z) dz = a.mn.z - b.mx.z;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

static inline bool isInsidePlane(const Point3& p, const Point3& n, double d) {
    return (n.dot(p) + d) >= -1e-10;
}

static inline Point3 intersectPlaneSeg(const Point3& p1, const Point3& p2, const Point3& n, double d) {
    double d1 = n.dot(p1) + d;
    double d2 = n.dot(p2) + d;
    double t = d1 / (d1 - d2);
    return p1 + (p2 - p1) * t;
}

static std::vector<Point3> clipPolyPlane(const std::vector<Point3>& in, const Point3& n, double d) {
    std::vector<Point3> out;
    if (in.empty()) return out;
    for (size_t i = 0; i < in.size(); ++i) {
        const Point3& cur = in[i];
        const Point3& prev = in[(i + in.size() - 1) % in.size()];
        bool curIn  = isInsidePlane(cur,  n, d);
        bool prevIn = isInsidePlane(prev, n, d);
        if (curIn) {
            if (!prevIn) out.push_back(intersectPlaneSeg(prev, cur, n, d));
            out.push_back(cur);
        } else if (prevIn) {
            out.push_back(intersectPlaneSeg(prev, cur, n, d));
        }
    }
    return out;
}

static double polygonArea(const std::vector<Point3>& poly) {
    if (poly.size() < 3) return 0.0;
    Point3 total{0,0,0};
    Point3 v0 = poly[0];
    for (size_t i = 1; i + 1 < poly.size(); ++i) {
        total = total + (poly[i] - v0).cross(poly[i+1] - v0);
    }
    return 0.5 * total.norm();
}

static Point3 polygonCenter(const std::vector<Point3>& poly) {
    Point3 c{0,0,0};
    if (poly.empty()) return c;
    if (poly.size() < 3) {
        for (const auto& p : poly) c = c + p;
        return c * (1.0 / (double)poly.size());
    }
    Point3 v0 = poly[0];
    double A_total = 0.0;
    Point3 C_total{0,0,0};
    for (size_t i = 1; i + 1 < poly.size(); ++i) {
        Point3 v1 = poly[i];
        Point3 v2 = poly[i+1];
        double Ai = 0.5 * ((v1 - v0).cross(v2 - v0)).norm();
        if (Ai < EPS) continue;
        Point3 triC = (v0 + v1 + v2) * (1.0 / 3.0);
        C_total = C_total + triC * Ai;
        A_total += Ai;
    }
    if (A_total < EPS) {
        for (const auto& p : poly) c = c + p;
        return c * (1.0 / (double)poly.size());
    }
    return C_total * (1.0 / A_total);
}

static inline double distPointSegment(const Point3& p, const Point3& a, const Point3& b) {
    Point3 ab = b - a;
    double t = (p - a).dot(ab) / std::max(EPS, ab.dot(ab));
    t = clampd(t, 0.0, 1.0);
    Point3 q = a + ab * t;
    return (p - q).norm();
}

static double distPointTriangle(const Point3& p, const Point3& a, const Point3& b, const Point3& c) {
    Point3 ab = b - a;
    Point3 ac = c - a;
    Point3 ap = p - a;
    double d1 = ab.dot(ap);
    double d2 = ac.dot(ap);
    if (d1 <= 0.0 && d2 <= 0.0) return (p - a).norm();
    Point3 bp = p - b;
    double d3 = ab.dot(bp);
    double d4 = ac.dot(bp);
    if (d3 >= 0.0 && d4 <= d3) return (p - b).norm();
    double vc = d1*d4 - d3*d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        double v = d1 / std::max(EPS, (d1 - d3));
        Point3 q = a + ab * v;
        return (p - q).norm();
    }
    Point3 cp = p - c;
    double d5 = ab.dot(cp);
    double d6 = ac.dot(cp);
    if (d6 >= 0.0 && d5 <= d6) return (p - c).norm();
    double vb = d5*d2 - d1*d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        double w = d2 / std::max(EPS, (d2 - d6));
        Point3 q = a + ac * w;
        return (p - q).norm();
    }
    double va = d3*d6 - d5*d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        double w = (d4 - d3) / std::max(EPS, ((d4 - d3) + (d5 - d6)));
        Point3 q = b + (c - b) * w;
        return (p - q).norm();
    }
    Point3 n = ab.cross(ac);
    double nlen = std::max(EPS, n.norm());
    n = n * (1.0 / nlen);
    return std::abs((p - a).dot(n));
}

static inline double distPointQuad(const Point3& p, const Fracture& f) {
    const Point3& v0 = f.vertices[0];
    const Point3& v1 = f.vertices[1];
    const Point3& v2 = f.vertices[2];
    const Point3& v3 = f.vertices[3];
    double d1 = distPointTriangle(p, v0, v1, v2);
    double d2 = distPointTriangle(p, v0, v2, v3);
    return std::min(d1, d2);
}

static inline Point3 pointMin(const Point3& a, const Point3& b) {
    return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)};
}

static inline Point3 pointMax(const Point3& a, const Point3& b) {
    return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)};
}

static inline Point3 averagePoints(const std::array<Point3, 4>& pts) {
    Point3 c{0,0,0};
    for (const auto& p : pts) c = c + p;
    return c / 4.0;
}

static inline Point3 averagePoints(const std::array<Point3, 8>& pts) {
    Point3 c{0,0,0};
    for (const auto& p : pts) c = c + p;
    return c / 8.0;
}

static inline double triangleArea3D(const Point3& a, const Point3& b, const Point3& c) {
    return 0.5 * ((b - a).cross(c - a)).norm();
}

static inline double quadArea3D(const std::array<Point3, 4>& q) {
    return triangleArea3D(q[0], q[1], q[2]) + triangleArea3D(q[0], q[2], q[3]);
}

static inline Point3 quadUnitNormal(const std::array<Point3, 4>& q) {
    Point3 n{0,0,0};
    for (int i = 0; i < 4; ++i) {
        const Point3& p = q[i];
        const Point3& r = q[(i + 1) % 4];
        n.x += (p.y - r.y) * (p.z + r.z);
        n.y += (p.z - r.z) * (p.x + r.x);
        n.z += (p.x - r.x) * (p.y + r.y);
    }
    double nn = n.norm();
    if (nn < EPS) return {0,0,0};
    return n / nn;
}

static inline void orientQuadOutward(std::array<Point3, 4>& q, const Point3& cell_center) {
    Point3 fc = averagePoints(q);
    Point3 n = quadUnitNormal(q);
    if (n.norm() < EPS) return;
    if (n.dot(fc - cell_center) < 0.0) std::reverse(q.begin(), q.end());
}

static inline double signedTetVolumeFromOrigin(const Point3& a, const Point3& b, const Point3& c) {
    return a.dot(b.cross(c)) / 6.0;
}

static inline double hexaVolumeFromOrientedFaces(const std::array<std::array<Point3, 4>, 6>& face_quads) {
    double v = 0.0;
    for (const auto& q : face_quads) {
        v += signedTetVolumeFromOrigin(q[0], q[1], q[2]);
        v += signedTetVolumeFromOrigin(q[0], q[2], q[3]);
    }
    return std::abs(v);
}

static inline std::array<int, 4> getLocalFaceCornerIds(int face_id) {
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

template<typename HexCell>
static inline std::array<Point3, 4> getCellFaceVertices(const HexCell& c, int face_id) {
    auto ids = getLocalFaceCornerIds(face_id);
    return {c.corners[ids[0]], c.corners[ids[1]], c.corners[ids[2]], c.corners[ids[3]]};
}

template<typename HexCell>
static void computeCellDerivedGeometry(HexCell& c) {
    c.center = averagePoints(c.corners);
    c.bbox_min = c.corners[0];
    c.bbox_max = c.corners[0];
    for (int n = 1; n < 8; ++n) {
        c.bbox_min = pointMin(c.bbox_min, c.corners[n]);
        c.bbox_max = pointMax(c.bbox_max, c.corners[n]);
    }
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

template<typename HexCell>
static Point3 mapHexTrilinear(const HexCell& cell, double u, double v, double w) {
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

static void computeFaceDerivedGeometry(FaceGeom& f, const Point3& owner_center) {
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

static std::vector<Point3> cleanupPolygon3D(const std::vector<Point3>& poly, double tol) {
    std::vector<Point3> out;
    if (poly.empty()) return out;
    for (const auto& p : poly) {
        if (out.empty() || (p - out.back()).norm() > tol) out.push_back(p);
    }
    if (out.size() >= 2 && (out.front() - out.back()).norm() <= tol) out.pop_back();
    return out;
}

template<typename HexCell>
static bool getInwardPlaneForCellLocalFace(const HexCell& cell, int local_face_id, Point3& n_in, double& d) {
    auto q = getCellFaceVertices(cell, local_face_id);
    orientQuadOutward(q, cell.center);
    Point3 n_out = quadUnitNormal(q);
    if (n_out.norm() < EPS) return false;
    n_in = n_out * (-1.0);
    const Point3& p0 = q[0];
    d = -n_in.dot(p0);
    return true;
}

template<typename HexCell>
static std::vector<Point3> clipPolygonByHexCell(const std::vector<Point3>& inputPoly, const HexCell& cell) {
    std::vector<Point3> poly = inputPoly;
    double scale = std::max(1.0, std::max(cell.dx, std::max(cell.dy, cell.dz)));
    double tol = 1e-10 * scale;
    for (int lf = 0; lf < 6; ++lf) {
        Point3 n_in;
        double d = 0.0;
        if (!getInwardPlaneForCellLocalFace(cell, lf, n_in, d)) {
            poly.clear();
            return poly;
        }
        poly = clipPolyPlane(poly, n_in, d);
        poly = cleanupPolygon3D(poly, tol);
        if (poly.size() < 3) {
            poly.clear();
            return poly;
        }
    }
    return poly;
}

template<typename HexCell>
static std::vector<Point3> clipFractureCell(const Fracture& frac, const HexCell& cell) {
    std::vector<Point3> poly;
    poly.reserve(4);
    for (int i = 0; i < 4; ++i) poly.push_back(frac.vertices[i]);
    double scale = std::max(1.0, std::max(cell.dx, std::max(cell.dy, cell.dz)));
    double tol = 1e-10 * scale;
    poly = cleanupPolygon3D(poly, tol);
    if (poly.size() < 3) return {};
    poly = clipPolygonByHexCell(poly, cell);
    poly = cleanupPolygon3D(poly, tol);
    if (poly.size() < 3) return {};
    return poly;
}

template<typename HexCell>
static double dbarHexQuad(const HexCell& cell, const Fracture& f, int nsu, int nsv, int nsw) {
    double sum = 0.0;
    int cnt = 0;
    for (int k = 0; k < nsw; ++k) {
        for (int j = 0; j < nsv; ++j) {
            for (int i = 0; i < nsu; ++i) {
                double u = (i + 0.5) / (double)nsu;
                double v = (j + 0.5) / (double)nsv;
                double w = (k + 0.5) / (double)nsw;
                Point3 p = mapHexTrilinear(cell, u, v, w);
                sum += distPointQuad(p, f);
                cnt++;
            }
        }
    }
    return (cnt > 0) ? (sum / cnt) : 1e30;
}

template<typename HexCell>
static double normalProjectedPerm(const HexCell& c, const Point3& n_unit) {
    return
        n_unit.x * n_unit.x * c.K[0] +
        n_unit.y * n_unit.y * c.K[1] +
        n_unit.z * n_unit.z * c.K[2];
}

template<typename HexCell>
static double centerToFaceNormalDistance(const HexCell& c, const FaceGeom& f) {
    double d = std::abs((f.center - c.center).dot(f.normal));
    return std::max(d, 1e-10);
}

template<typename HexCell>
static double computeMMTransmissibilityTPFA(const HexCell& cu, const HexCell& cv, const FaceGeom& f) {
    double Af = f.area;
    if (Af <= EPS) return 0.0;
    double kn_u = normalProjectedPerm(cu, f.normal);
    double kn_v = normalProjectedPerm(cv, f.normal);
    if (kn_u <= EPS || kn_v <= EPS) return 0.0;
    double du = centerToFaceNormalDistance(cu, f);
    double dv = centerToFaceNormalDistance(cv, f);
    double tau_u = kn_u * Af / du;
    double tau_v = kn_v * Af / dv;
    if (tau_u <= EPS || tau_v <= EPS) return 0.0;
    return (tau_u * tau_v) / std::max(EPS, tau_u + tau_v);
}

template<typename HexCell>
static double averageDistanceCellToPlane(const HexCell& cell, const Point3& planePoint, const Point3& unitNormal,
                                         int nxs = 2, int nys = 2, int nzs = 2) {
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
    return (count > 0) ? (sum / (double)count) : 0.0;
}

template<typename HexCell>
static double computeMatrixFractureTransmissibility(const HexCell& cell, const Segment& seg, int nxs = 2, int nys = 2, int nzs = 2) {
    if (seg.area <= EPS) return 0.0;
    Point3 n = seg.normal;
    double nn = n.norm();
    if (nn < EPS) return 0.0;
    n = n * (1.0 / nn);
    double Kn = normalProjectedPerm(cell, n);
    if (Kn <= EPS) return 0.0;
    double d_avg = averageDistanceCellToPlane(cell, seg.center, n, nxs, nys, nzs);
    double scale = std::max(1.0, std::max(cell.dx, std::max(cell.dy, cell.dz)));
    d_avg = std::max(d_avg, 1e-10 * scale);
    double Tmf = 2.0 * seg.area * (Kn / d_avg);
    if (!std::isfinite(Tmf) || Tmf <= EPS) return 0.0;
    return Tmf;
}

static void pushUniquePoint(std::vector<Point3>& pts, const Point3& p, double tol) {
    for (const auto& q : pts) if ((p - q).norm() <= tol) return;
    pts.push_back(p);
}

static bool pointInTriangle3D(const Point3& p, const Point3& a, const Point3& b, const Point3& c, double tol) {
    Point3 v0 = b - a;
    Point3 v1 = c - a;
    Point3 v2 = p - a;
    double d00 = v0.dot(v0);
    double d01 = v0.dot(v1);
    double d11 = v1.dot(v1);
    double d20 = v2.dot(v0);
    double d21 = v2.dot(v1);
    double denom = d00 * d11 - d01 * d01;
    if (std::abs(denom) < EPS) return false;
    double v = (d11 * d20 - d01 * d21) / denom;
    double w = (d00 * d21 - d01 * d20) / denom;
    double u = 1.0 - v - w;
    return (u >= -tol && v >= -tol && w >= -tol);
}

static bool pointInConvexQuad3D(const Point3& p, const std::array<Point3, 4>& q, double tol) {
    return pointInTriangle3D(p, q[0], q[1], q[2], tol) || pointInTriangle3D(p, q[0], q[2], q[3], tol);
}

static bool pointOnFaceGeom(const Point3& p, const FaceGeom& fg, double tol) {
    if (fg.area <= EPS) return false;
    const Point3& p0 = fg.vertices[0];
    double dist_to_plane = std::abs((p - p0).dot(fg.normal));
    if (dist_to_plane > tol) return false;
    return pointInConvexQuad3D(p, fg.vertices, tol);
}

static bool extractTraceOnFaceGeom(const std::vector<Point3>& poly, const FaceGeom& fg,
                                   Point3& a, Point3& b, double& len, double tol) {
    len = 0.0;
    if (poly.size() < 2 || fg.area <= EPS) return false;
    const Point3& p0 = fg.vertices[0];
    const Point3& n = fg.normal;
    double planeD = -n.dot(p0);
    std::vector<Point3> pts;
    auto try_add_point = [&](const Point3& q) {
        if (pointOnFaceGeom(q, fg, tol)) pushUniquePoint(pts, q, tol);
    };
    for (const auto& p : poly) try_add_point(p);
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
        if ((d1 > tol && d2 < -tol) || (d1 < -tol && d2 > tol)) {
            Point3 q = intersectPlaneSeg(p1, p2, n, planeD);
            try_add_point(q);
        }
    }
    if (pts.size() < 2) return false;
    double best = -1.0;
    Point3 pa{0,0,0}, pb{0,0,0};
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
    if (best <= EPS) return false;
    a = pa; b = pb; len = best;
    return true;
}

static double pointToLineDistance3D(const Point3& p, const Point3& a, const Point3& b) {
    Point3 ab = b - a;
    double lab = ab.norm();
    if (lab < EPS) return 0.0;
    return ((p - a).cross(ab)).norm() / lab;
}

static void fillSegmentFaceGeom(Segment& seg, const LeafCell& leaf, const std::vector<FaceGeom>& mm_faces) {
    seg.subface_trace_len.clear();
    seg.subface_center_dist.clear();
    for (int lf = 0; lf < 6; ++lf) {
        seg.face_trace_len[lf] = 0.0;
        seg.face_center_dist[lf] = 0.0;
        double best_len_local = 0.0;
        double best_dist_local = 0.0;
        for (int sfid : leaf.subface_ids[lf]) {
            if (sfid < 0 || sfid >= (int)mm_faces.size()) continue;
            const FaceGeom& fg = mm_faces[sfid];
            double scale = std::max(1.0, std::max(leaf.dx, std::max(leaf.dy, leaf.dz)));
            double tol = 1e-8 * scale;
            Point3 a, b;
            double len = 0.0;
            if (!extractTraceOnFaceGeom(seg.poly, fg, a, b, len, tol)) continue;
            double dist = pointToLineDistance3D(seg.center, a, b);
            dist = std::max(dist, 1e-10);
            seg.subface_trace_len[sfid] = len;
            seg.subface_center_dist[sfid] = dist;
            if ((int)leaf.subface_ids[lf].size() == 1) {
                seg.face_trace_len[lf] = len;
                seg.face_center_dist[lf] = dist;
            } else if (len > best_len_local) {
                best_len_local = len;
                best_dist_local = dist;
            }
        }
        if ((int)leaf.subface_ids[lf].size() > 1 && best_len_local > 0.0) {
            seg.face_trace_len[lf] = best_len_local;
            seg.face_center_dist[lf] = best_dist_local;
        }
    }
}

static bool intersectTwoPlanes(const Point3& n1, const Point3& p1, const Point3& n2, const Point3& p2,
                               Point3& linePoint, Point3& lineDir) {
    Point3 dir = n1.cross(n2);
    double dir2 = dir.dot(dir);
    if (dir2 < 1e-14) return false;
    double d1 = n1.dot(p1);
    double d2 = n2.dot(p2);
    Point3 term1 = n2.cross(dir) * d1;
    Point3 term2 = dir.cross(n1) * d2;
    linePoint = (term1 + term2) * (1.0 / dir2);
    lineDir = dir * (1.0 / std::sqrt(dir2));
    return true;
}

static bool clipLineByConvexPolygon(const std::vector<Point3>& poly, const Point3& normal,
                                    const Point3& linePoint, const Point3& lineDir,
                                    double& tmin, double& tmax) {
    if (poly.size() < 3) return false;
    Point3 centroid = polygonCenter(poly);
    tmin = -1e100;
    tmax =  1e100;
    for (size_t i = 0; i < poly.size(); ++i) {
        const Point3& vi = poly[i];
        const Point3& vj = poly[(i + 1) % poly.size()];
        Point3 e = vj - vi;
        Point3 m = normal.cross(e);
        if (m.norm() < EPS) continue;
        if (m.dot(centroid - vi) < 0.0) m = m * (-1.0);
        double c = m.dot(linePoint - vi);
        double den = m.dot(lineDir);
        const double tol = 1e-12;
        if (std::abs(den) < tol) {
            if (c < -tol) return false;
            continue;
        }
        double tbound = -c / den;
        if (den > 0.0) tmin = std::max(tmin, tbound);
        else tmax = std::min(tmax, tbound);
        if (tmin > tmax) return false;
    }
    return true;
}

static bool computeCrossIntersectionSegment(const Segment& seg1, const Segment& seg2,
                                            Point3& a, Point3& b, double& ell_int) {
    ell_int = 0.0;
    Point3 linePoint, lineDir;
    if (!intersectTwoPlanes(seg1.normal, seg1.center, seg2.normal, seg2.center, linePoint, lineDir)) return false;
    double t1min, t1max;
    if (!clipLineByConvexPolygon(seg1.poly, seg1.normal, linePoint, lineDir, t1min, t1max)) return false;
    double t2min, t2max;
    if (!clipLineByConvexPolygon(seg2.poly, seg2.normal, linePoint, lineDir, t2min, t2max)) return false;
    double ta = std::max(t1min, t2min);
    double tb = std::min(t1max, t2max);
    if (tb - ta <= 1e-10) return false;
    a = linePoint + lineDir * ta;
    b = linePoint + lineDir * tb;
    ell_int = (b - a).norm();
    return (ell_int > EPS);
}

static bool computeCrossGeom(const Segment& seg1, const Segment& seg2, double& ell_int, double& L1, double& L2) {
    Point3 a, b;
    if (!computeCrossIntersectionSegment(seg1, seg2, a, b, ell_int)) return false;
    L1 = std::max(distPointSegment(seg1.center, a, b), 1e-10);
    L2 = std::max(distPointSegment(seg2.center, a, b), 1e-10);
    return true;
}

static Point3 fractureCenter(const Fracture& f) {
    Point3 c{0,0,0};
    for (int i = 0; i < 4; ++i) c = c + f.vertices[i];
    return c / 4.0;
}

static void buildPlaneBasisFromNormal(const Point3& normal, Point3& e1, Point3& e2) {
    Point3 n = normal;
    double nn = n.norm();
    if (nn < EPS) {
        e1 = {1,0,0};
        e2 = {0,1,0};
        return;
    }
    n = n * (1.0 / nn);
    Point3 ref;
    if (std::abs(n.x) <= std::abs(n.y) && std::abs(n.x) <= std::abs(n.z)) ref = {1,0,0};
    else if (std::abs(n.y) <= std::abs(n.x) && std::abs(n.y) <= std::abs(n.z)) ref = {0,1,0};
    else ref = {0,0,1};
    e1 = n.cross(ref);
    double ne1 = e1.norm();
    if (ne1 < EPS) {
        ref = {0,1,0};
        e1 = n.cross(ref);
        ne1 = e1.norm();
        if (ne1 < EPS) {
            e1 = {1,0,0};
            e2 = {0,1,0};
            return;
        }
    }
    e1 = e1 * (1.0 / ne1);
    e2 = n.cross(e1);
    double ne2 = e2.norm();
    if (ne2 < EPS) e2 = {0,1,0};
    else e2 = e2 * (1.0 / ne2);
}

static bool computeSegmentLocalPlaneExtents(const Segment& seg, double& Lu, double& Lv) {
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
    if (Lu <= EPS && Lv > EPS && seg.area > EPS) Lu = seg.area / Lv;
    if (Lv <= EPS && Lu > EPS && seg.area > EPS) Lv = seg.area / Lu;
    return (Lu > EPS && Lv > EPS);
}

static double computeSegmentEquivalentRadius(const Segment& seg, double rw) {
    double Lu = 0.0, Lv = 0.0;
    bool ok = computeSegmentLocalPlaneExtents(seg, Lu, Lv);
    double re = 0.0;
    if (ok) re = 0.14 * std::sqrt(Lu * Lu + Lv * Lv);
    else if (seg.area > EPS) {
        double Leq = std::sqrt(seg.area);
        re = 0.14 * std::sqrt(2.0) * Leq;
    } else re = 1.1 * rw;
    re = std::max(re, 1.1 * rw);
    return re;
}

typedef Eigen::Matrix<double, 3, 1> Deriv3;
typedef Eigen::AutoDiffScalar<Deriv3> AD3;

struct FluidProps {
    double mu_w = 1.0;
    double mu_o = 5.0;
    double mu_g = 0.2;
    double cw = 1e-8;
    double co = 1e-5;
    double cg = 1e-3;
    double P_ref = 100.0;
    double Swi = 0.05;
    double Sor = 0.01;
    double Sgc = 0.05;
};

static FluidProps g_props;

template <typename T>
static void calcPVT(const T& P, T& Bw, T& Bo, T& Bg, T& dBw_dP, T& dBo_dP, T& dBg_dP) {
    T dP = P - g_props.P_ref;
    using std::exp;
    Bw = exp(-g_props.cw * dP);
    Bo = exp(-g_props.co * dP);
    Bg = exp(-g_props.cg * dP);
}

template <typename T>
static void calcRelPerm(const T& Sw, const T& Sg, T& krw, T& kro, T& krg) {
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

template <typename T>
struct StateT {
    T P{800.0};
    T Sw{0.05};
    T Sg{0.9};
};

template <typename T>
struct PropertiesT {
    T Bw, Bo, Bg;
    T krw, kro, krg;
    T lw, lo, lg;
};

typedef StateT<double> State;
typedef StateT<AD3> StateAD3;

template <typename T>
PropertiesT<T> getProps(const StateT<T>& s) {
    PropertiesT<T> p;
    T dummy;
    calcPVT(s.P, p.Bw, p.Bo, p.Bg, dummy, dummy, dummy);
    calcRelPerm(s.Sw, s.Sg, p.krw, p.kro, p.krg);
    p.lw = p.krw / (g_props.mu_w * p.Bw);
    p.lo = p.kro / (g_props.mu_o * p.Bo);
    p.lg = p.krg / (g_props.mu_g * p.Bg);
    return p;
}

struct SimulationResult {
    py::array_t<double> pressure_field;
    py::array_t<double> temperature_field;
    py::array_t<double> stress_field;
    py::array_t<double> fracture_vertices;
    py::array_t<double> fracture_cells;
};

class SimulatorLGR {
public:
    int Nx{150}, Ny{15}, Nz{2};
    double Lx{3000}, Ly{300}, Lz{40};
    double dx{}, dy{}, dz{};

    bool enable_lgr{true};
    double d_threshold{5.0};
    uint16_t lgr_Nrx{2}, lgr_Nry{2}, lgr_Nrz{2};
    int dbar_nsx{4}, dbar_nsy{4}, dbar_nsz{4};
    double eps_area_factor{1e-8};
    double d_avg_factor{0.5};

    std::vector<ParentCell> parents;
    std::vector<LeafCell> leaves;
    std::vector<Fracture> fractures;
    std::vector<Segment> segments;
    std::vector<FaceGeom> mm_faces;
    std::vector<ParentFaceCoverage> parent_face_cov;

    std::vector<std::vector<int>> leaf_neighbors;
    std::vector<std::vector<int>> leaf_to_segs;

    std::vector<Connection> connections;
    struct Neighbor { int v; double T; int type; };
    std::vector<std::vector<Neighbor>> adj;

    int n_leaf{0};
    int n_seg{0};
    int n_total{0};

    std::vector<State> states, states_prev;

    struct Well { int target_node_idx; double WI; double P_bhp; };
    std::vector<Well> wells;
    std::unordered_map<int,int> well_map;

    SparseMatrix<double> J;
    struct CellOffsets { int diag[3][3]; };
    struct ConnOffsets { int off_uv[3][3]; int off_vu[3][3]; };
    std::vector<CellOffsets> cell_J_idx;
    std::vector<ConnOffsets> conn_J_idx;

    std::string coord_file_path{"COORD.csv"};
    std::string zcorn_file_path{"ZCORN.csv"};
    int natural_frac_count{30};
    double natural_min_length{30.0};
    double natural_max_length{80.0};
    double natural_max_dip{PI / 3.0};
    double natural_min_strike{0.0};
    double natural_max_strike{PI};
    double natural_aperture{0.01};
    double natural_perm{1000.0};
    bool use_region_fractures{false};
    double region_x_min{0.0};
    double region_x_max{-1.0};
    double region_y_min{0.0};
    double region_y_max{-1.0};
    double region_z_min{0.0};
    double region_z_max{-1.0};
    int hydraulic_frac_count{20};
    double hydraulic_well_length{2000.0};
    double hydraulic_half_length{120.0};
    double hydraulic_height{30.0};
    double hydraulic_aperture{0.1};
    double hydraulic_perm{1000.0};
    double hydraulic_center_x{-1.0};
    double hydraulic_center_y{-1.0};
    double hydraulic_center_z{-1.0};
    double well_radius{0.05};
    double well_pressure{50.0};
    double initial_pressure{800.0};
    double initial_sw{0.05};
    double initial_sg{0.9};
    double simulation_total_days{100.0};

    SimulatorLGR() {
        dx = Lx / Nx;
        dy = Ly / Ny;
        dz = Lz / Nz;
    }

    void setCornerPointFiles(const std::string& coord_file, const std::string& zcorn_file) {
        coord_file_path = coord_file;
        zcorn_file_path = zcorn_file;
    }

    void setFractureParameters(int total_fracs,
                               double min_L,
                               double max_L,
                               double max_dip,
                               double min_strike,
                               double max_strike,
                               double aperture_val,
                               double perm_val) {
        natural_frac_count = total_fracs;
        natural_min_length = min_L;
        natural_max_length = max_L;
        natural_max_dip = max_dip;
        natural_min_strike = min_strike;
        natural_max_strike = max_strike;
        natural_aperture = aperture_val;
        natural_perm = perm_val;
        use_region_fractures = false;
        region_x_min = 0.0;
        region_x_max = -1.0;
        region_y_min = 0.0;
        region_y_max = -1.0;
        region_z_min = 0.0;
        region_z_max = -1.0;
    }

    void setRegionFractureParameters(int total_fracs,
                                     double x_min,
                                     double x_max,
                                     double y_min,
                                     double y_max,
                                     double z_min,
                                     double z_max) {
        natural_frac_count = total_fracs;
        use_region_fractures = true;
        region_x_min = x_min;
        region_x_max = x_max;
        region_y_min = y_min;
        region_y_max = y_max;
        region_z_min = z_min;
        region_z_max = z_max;
    }

    void setHydraulicFractureParameters(int total_fracs,
                                        double well_length,
                                        double hf_len,
                                        double hf_height,
                                        double aperture_val,
                                        double perm_val,
                                        double x_center = -1.0,
                                        double y_center = -1.0,
                                        double z_center = -1.0) {
        hydraulic_frac_count = total_fracs;
        hydraulic_well_length = well_length;
        hydraulic_half_length = hf_len;
        hydraulic_height = hf_height;
        hydraulic_aperture = aperture_val;
        hydraulic_perm = perm_val;
        hydraulic_center_x = x_center;
        hydraulic_center_y = y_center;
        hydraulic_center_z = z_center;
    }

    void setWellParameters(double rw, double pressure) {
        well_radius = rw;
        well_pressure = pressure;
    }

    void setInitialStateParameters(double pressure, double sw, double sg) {
        initial_pressure = pressure;
        initial_sw = sw;
        initial_sg = sg;
    }

    void setSimulationParameters(double total_days) {
        simulation_total_days = total_days;
    }

    void setLGRParameters(bool enabled,
                          double threshold,
                          int nrx,
                          int nry,
                          int nrz) {
        enable_lgr = enabled;
        d_threshold = threshold;
        lgr_Nrx = static_cast<uint16_t>(std::max(1, nrx));
        lgr_Nry = static_cast<uint16_t>(std::max(1, nry));
        lgr_Nrz = static_cast<uint16_t>(std::max(1, nrz));
    }

    int getNextFractureId() const {
        int max_id = -1;
        for (const auto& f : fractures) max_id = std::max(max_id, f.id);
        return max_id + 1;
    }

    bool loadCOORD(const std::string& filename, std::vector<Pillar>& pillars,
                   int& coord_max_i, int& coord_max_j) const {
        std::ifstream fin(filename);
        if (!fin.is_open()) {
            std::cerr << "Error: cannot open COORD file: " << filename << std::endl;
            return false;
        }
        std::string line;
        int line_no = 0;
        if (!std::getline(fin, line)) {
            std::cerr << "Error: COORD file is empty: " << filename << std::endl;
            return false;
        }
        line_no++;

        std::vector<CoordRow> rows;
        coord_max_i = 0;
        coord_max_j = 0;

        while (std::getline(fin, line)) {
            line_no++;
            line = trim(line);
            if (line.empty()) continue;

            auto cols = splitCSVSimple(line);
            if (cols.size() < 6) {
                std::cerr << "Error: " << filename << " line " << line_no
                          << " has fewer than 6 columns." << std::endl;
                return false;
            }

            int i1 = 0, j1 = 0;
            double x = 0.0, y = 0.0, z = 0.0;
            if (!parseIntStrict(cols[0], i1) || !parseIntStrict(cols[1], j1)) {
                std::cerr << "Error: " << filename << " line " << line_no
                          << " failed to parse X(I)/Y(J)." << std::endl;
                return false;
            }
            if (i1 <= 0 || j1 <= 0) {
                std::cerr << "Error: " << filename << " line " << line_no
                          << " has non-positive pillar index." << std::endl;
                return false;
            }

            bool is_top = false;
            bool is_bot = false;
            if (isTopMarker(cols[2])) is_top = true;
            else if (isBottomMarker(cols[2])) is_bot = true;
            else {
                std::cerr << "Error: " << filename << " line " << line_no
                          << " has invalid Z(K) marker: " << cols[2]
                          << " , expected 顶/底 or top/bottom." << std::endl;
                return false;
            }

            if (!parseDoubleStrict(cols[3], x) ||
                !parseDoubleStrict(cols[4], y) ||
                !parseDoubleStrict(cols[5], z)) {
                std::cerr << "Error: " << filename << " line " << line_no
                          << " failed to parse 坐标X/坐标Y/坐标Z." << std::endl;
                return false;
            }

            CoordRow r;
            r.i = i1 - 1;
            r.j = j1 - 1;
            r.is_top = is_top;
            r.p = {x, y, z};
            rows.push_back(r);

            coord_max_i = std::max(coord_max_i, i1);
            coord_max_j = std::max(coord_max_j, j1);
        }

        if (rows.empty()) {
            std::cerr << "Error: COORD file contains no data rows: " << filename << std::endl;
            return false;
        }

        pillars.assign(coord_max_i * coord_max_j, Pillar{});

        for (const auto& r : rows) {
            int idx = flatPillarIndex(r.i, r.j, coord_max_i);
            Pillar& p = pillars[idx];
            if (r.is_top) {
                if (p.has_top) {
                    std::cerr << "Error: duplicate TOP record in " << filename
                              << " for pillar(" << (r.i+1) << "," << (r.j+1) << ")." << std::endl;
                    return false;
                }
                p.top = r.p;
                p.has_top = true;
            } else {
                if (p.has_bot) {
                    std::cerr << "Error: duplicate BOTTOM record in " << filename
                              << " for pillar(" << (r.i+1) << "," << (r.j+1) << ")." << std::endl;
                    return false;
                }
                p.bot = r.p;
                p.has_bot = true;
            }
        }

        for (int j = 0; j < coord_max_j; ++j) {
            for (int i = 0; i < coord_max_i; ++i) {
                const Pillar& p = pillars[flatPillarIndex(i, j, coord_max_i)];
                if (!p.has_top || !p.has_bot) {
                    std::cerr << "Error: " << filename << " pillar(" << (i+1) << "," << (j+1)
                              << ") missing "
                              << ((!p.has_top && !p.has_bot) ? "top and bottom" :
                                  (!p.has_top ? "top" : "bottom"))
                              << " record." << std::endl;
                    return false;
                }
                if (std::abs(p.bot.z - p.top.z) < EPS) {
                    std::cerr << "Error: " << filename << " pillar(" << (i+1) << "," << (j+1)
                              << ") has Z_top == Z_bottom." << std::endl;
                    return false;
                }
                if (!(p.top.z > p.bot.z)) {
                    std::cerr << "Error: " << filename << " pillar(" << (i+1) << "," << (j+1)
                              << ") violates height convention: Z_top must be > Z_bottom." << std::endl;
                    return false;
                }
            }
        }
        return true;
    }

    bool loadZCORN(const std::string& filename, std::vector<std::array<double,8>>& zcorn_cells,
                   int& zcorn_max_i, int& zcorn_max_j, int& zcorn_max_k) const {
        std::ifstream fin(filename);
        if (!fin.is_open()) {
            std::cerr << "Error: cannot open ZCORN file: " << filename << std::endl;
            return false;
        }
        std::string line;
        int line_no = 0;
        if (!std::getline(fin, line)) {
            std::cerr << "Error: ZCORN file is empty: " << filename << std::endl;
            return false;
        }
        line_no++;

        std::vector<ZCornRow> rows;
        zcorn_max_i = zcorn_max_j = zcorn_max_k = 0;

        while (std::getline(fin, line)) {
            line_no++;
            line = trim(line);
            if (line.empty()) continue;

            auto cols = splitCSVSimple(line);
            if (cols.size() < 11) {
                std::cerr << "Error: " << filename << " line " << line_no
                          << " has fewer than 11 columns." << std::endl;
                return false;
            }

            int i1 = 0, j1 = 0, k1 = 0;
            if (!parseIntStrict(cols[0], i1) ||
                !parseIntStrict(cols[1], j1) ||
                !parseIntStrict(cols[2], k1)) {
                std::cerr << "Error: " << filename << " line " << line_no
                          << " failed to parse X(I)/Y(J)/Z(K)." << std::endl;
                return false;
            }
            if (i1 <= 0 || j1 <= 0 || k1 <= 0) {
                std::cerr << "Error: " << filename << " line " << line_no
                          << " has non-positive cell index." << std::endl;
                return false;
            }

            ZCornRow zr;
            zr.i = i1 - 1;
            zr.j = j1 - 1;
            zr.k = k1 - 1;

            for (int t = 0; t < 8; ++t) {
                if (!parseDoubleStrict(cols[3 + t], zr.z[t])) {
                    std::cerr << "Error: " << filename << " line " << line_no
                              << " failed to parse Z" << (t + 1) << "." << std::endl;
                    return false;
                }
            }

            if (!(zr.z[0] > zr.z[4] &&
                  zr.z[1] > zr.z[5] &&
                  zr.z[2] > zr.z[6] &&
                  zr.z[3] > zr.z[7])) {
                std::cerr << "Error: " << filename << " line " << line_no
                          << " violates height convention: top Z must be greater than bottom Z."
                          << std::endl;
                return false;
            }

            rows.push_back(zr);
            zcorn_max_i = std::max(zcorn_max_i, i1);
            zcorn_max_j = std::max(zcorn_max_j, j1);
            zcorn_max_k = std::max(zcorn_max_k, k1);
        }

        if (rows.empty()) {
            std::cerr << "Error: ZCORN file contains no data rows: " << filename << std::endl;
            return false;
        }

        int total_cells = zcorn_max_i * zcorn_max_j * zcorn_max_k;
        zcorn_cells.assign(total_cells, std::array<double,8>{{0,0,0,0,0,0,0,0}});
        std::vector<char> seen(total_cells, 0);

        for (const auto& zr : rows) {
            int idx = flatCellIndex(zr.i, zr.j, zr.k, zcorn_max_i, zcorn_max_j);
            if (seen[idx]) {
                std::cerr << "Error: duplicate ZCORN record in " << filename
                          << " for cell(" << (zr.i+1) << "," << (zr.j+1) << "," << (zr.k+1)
                          << ")." << std::endl;
                return false;
            }
            seen[idx] = 1;
            zcorn_cells[idx] = zr.z;
        }

        for (int k = 0; k < zcorn_max_k; ++k) {
            for (int j = 0; j < zcorn_max_j; ++j) {
                for (int i = 0; i < zcorn_max_i; ++i) {
                    int idx = flatCellIndex(i, j, k, zcorn_max_i, zcorn_max_j);
                    if (!seen[idx]) {
                        std::cerr << "Error: missing ZCORN record in " << filename
                                  << " for cell(" << (i+1) << "," << (j+1) << "," << (k+1)
                                  << ")." << std::endl;
                        return false;
                    }
                }
            }
        }
        return true;
    }

    Point3 interpolateOnPillar(const Pillar& p, double zc) const {
        if (!p.has_top || !p.has_bot) {
            throw std::runtime_error("pillar missing top or bottom endpoint.");
        }
        double zt = p.top.z;
        double zb = p.bot.z;
        double denom = zb - zt;
        if (std::abs(denom) < EPS) {
            throw std::runtime_error("pillar has Z_top == Z_bottom, cannot interpolate.");
        }

        double zmax = std::max(zt, zb);
        double zmin = std::min(zt, zb);
        double tol = 1e-9 * std::max(1.0, zmax - zmin);
        if (zc > zmax + tol || zc < zmin - tol) {
            std::ostringstream oss;
            oss << "interpolation z = " << zc
                << " is outside pillar range [" << zmin << ", " << zmax << "]";
            throw std::runtime_error(oss.str());
        }

        double lambda = (zc - zt) / denom;
        Point3 pc = p.top + (p.bot - p.top) * lambda;
        pc.z = zc;
        return pc;
    }

    bool buildParentGridFromCornerPointCSV(const std::string& coordFile, const std::string& zcornFile) {
        parents.clear();

        std::vector<Pillar> pillars;
        std::vector<std::array<double, 8>> zcorn_cells;
        int coord_max_i = 0, coord_max_j = 0;
        int zcorn_max_i = 0, zcorn_max_j = 0, zcorn_max_k = 0;

        if (!loadCOORD(coordFile, pillars, coord_max_i, coord_max_j)) {
            std::cerr << "Error: failed to load COORD file." << std::endl;
            return false;
        }
        if (!loadZCORN(zcornFile, zcorn_cells, zcorn_max_i, zcorn_max_j, zcorn_max_k)) {
            std::cerr << "Error: failed to load ZCORN file." << std::endl;
            return false;
        }

        if (coord_max_i != zcorn_max_i + 1) {
            std::cerr << "Error: inconsistent grid size in I direction: COORD max I = "
                      << coord_max_i << ", but ZCORN max I = " << zcorn_max_i
                      << " , expected COORD max I = ZCORN max I + 1." << std::endl;
            return false;
        }
        if (coord_max_j != zcorn_max_j + 1) {
            std::cerr << "Error: inconsistent grid size in J direction: COORD max J = "
                      << coord_max_j << ", but ZCORN max J = " << zcorn_max_j
                      << " , expected COORD max J = ZCORN max J + 1." << std::endl;
            return false;
        }

        Nx = zcorn_max_i;
        Ny = zcorn_max_j;
        Nz = zcorn_max_k;

        int n_parent = Nx * Ny * Nz;
        parents.resize(n_parent);

        auto getPillar = [&](int i, int j) -> const Pillar& {
            if (i < 0 || i >= coord_max_i || j < 0 || j >= coord_max_j) {
                std::ostringstream oss;
                oss << "pillar index out of range: (" << (i + 1) << "," << (j + 1) << ")";
                throw std::runtime_error(oss.str());
            }
            return pillars[flatPillarIndex(i, j, coord_max_i)];
        };

        try {
            for (int k = 0; k < Nz; ++k) {
                for (int j = 0; j < Ny; ++j) {
                    for (int i = 0; i < Nx; ++i) {
                        int id = flatCellIndex(i, j, k, Nx, Ny);
                        ParentCell pc;
                        pc.parent_id = id;
                        pc.ix = i;
                        pc.iy = j;
                        pc.iz = k;

                        const Pillar& p_ll = getPillar(i,     j    );
                        const Pillar& p_lr = getPillar(i + 1, j    );
                        const Pillar& p_ul = getPillar(i,     j + 1);
                        const Pillar& p_ur = getPillar(i + 1, j + 1);

                        const auto& z = zcorn_cells[id];

                        pc.corners[0] = interpolateOnPillar(p_ll, z[4]); // Z5
                        pc.corners[1] = interpolateOnPillar(p_lr, z[5]); // Z6
                        pc.corners[2] = interpolateOnPillar(p_ur, z[7]); // Z8
                        pc.corners[3] = interpolateOnPillar(p_ul, z[6]); // Z7
                        pc.corners[4] = interpolateOnPillar(p_ll, z[0]); // Z1
                        pc.corners[5] = interpolateOnPillar(p_lr, z[1]); // Z2
                        pc.corners[6] = interpolateOnPillar(p_ur, z[3]); // Z4
                        pc.corners[7] = interpolateOnPillar(p_ul, z[2]); // Z3

                        pc.phi = 0.04;
                        pc.K[0] = 0.005;
                        pc.K[1] = 0.005;
                        pc.K[2] = 0.005;
                        pc.face_ids = {{-1,-1,-1,-1,-1,-1}};
                        pc.refined = false;
                        pc.leaf_base = -1;
                        pc.Nrx = lgr_Nrx;
                        pc.Nry = lgr_Nry;
                        pc.Nrz = lgr_Nrz;

                        computeCellDerivedGeometry(pc);
                        parents[id] = pc;
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error while constructing parent corner-point grid: "
                      << e.what() << std::endl;
            return false;
        }

        if (!parents.empty()) {
            Point3 gmin = parents[0].bbox_min;
            Point3 gmax = parents[0].bbox_max;
            for (const auto& p : parents) {
                gmin = pointMin(gmin, p.bbox_min);
                gmax = pointMax(gmax, p.bbox_max);
            }
            Lx = gmax.x - gmin.x;
            Ly = gmax.y - gmin.y;
            Lz = gmax.z - gmin.z;
            dx = (Nx > 0) ? (Lx / Nx) : 0.0;
            dy = (Ny > 0) ? (Ly / Ny) : 0.0;
            dz = (Nz > 0) ? (Lz / Nz) : 0.0;
        } else {
            Lx = Ly = Lz = 0.0;
            dx = dy = dz = 0.0;
        }

        std::cout << "Parent corner-point grid initialized from CSV successfully." << std::endl;
        std::cout << "Nx = " << Nx << ", Ny = " << Ny << ", Nz = " << Nz << std::endl;
        std::cout << "Pillars = " << coord_max_i << " x " << coord_max_j << std::endl;
        std::cout << "Parents = " << n_parent << std::endl;
        return true;
    }

    std::array<Point3, 4> buildParentFaceSubQuad(const ParentCell& p, int face_id,
                                                 double s0, double s1, double t0, double t1) const {
        auto mapFaceST = [&](double s, double t) -> Point3 {
            double u = 0.0, v = 0.0, w = 0.0;
            switch (face_id) {
                case XM: u = 0.0; v = s;   w = t;   break;
                case XP: u = 1.0; v = s;   w = t;   break;
                case YM: u = s;   v = 0.0; w = t;   break;
                case YP: u = s;   v = 1.0; w = t;   break;
                case ZM: u = s;   v = t;   w = 0.0; break;
                case ZP: u = s;   v = t;   w = 1.0; break;
                default: break;
            }
            return mapHexTrilinear(p, u, v, w);
        };
        return {
            mapFaceST(s0, t0),
            mapFaceST(s1, t0),
            mapFaceST(s1, t1),
            mapFaceST(s0, t1)
        };
    }

    void buildParentFaceCoverage() {
        parent_face_cov.clear();
        parent_face_cov.resize(parents.size());
        for (const auto& p : parents) {
            ParentFaceCoverage cov;
            if (!p.refined) {
                int lid = p.leaf_base;
                for (int f = 0; f < 6; ++f) {
                    ParentFacePatch patch;
                    patch.leaf_id = lid;
                    patch.leaf_local_face = f;
                    patch.s0 = 0.0; patch.s1 = 1.0;
                    patch.t0 = 0.0; patch.t1 = 1.0;
                    patch.quad = buildParentFaceSubQuad(p, f, 0.0, 1.0, 0.0, 1.0);
                    cov.patches[f].push_back(patch);
                }
            } else {
                for (uint16_t lk = 0; lk < p.Nrz; ++lk) {
                    for (uint16_t lj = 0; lj < p.Nry; ++lj) {
                        double s0 = (double)lj / (double)p.Nry;
                        double s1 = (double)(lj + 1) / (double)p.Nry;
                        double t0 = (double)lk / (double)p.Nrz;
                        double t1 = (double)(lk + 1) / (double)p.Nrz;
                        int lid_xm = p.leaf_base + ((int)lk * (int)p.Nry + (int)lj) * (int)p.Nrx + 0;
                        int lid_xp = p.leaf_base + ((int)lk * (int)p.Nry + (int)lj) * (int)p.Nrx + ((int)p.Nrx - 1);

                        ParentFacePatch pxm;
                        pxm.leaf_id = lid_xm; pxm.leaf_local_face = XM;
                        pxm.s0 = s0; pxm.s1 = s1; pxm.t0 = t0; pxm.t1 = t1;
                        pxm.quad = buildParentFaceSubQuad(p, XM, s0, s1, t0, t1);
                        cov.patches[XM].push_back(pxm);

                        ParentFacePatch pxp;
                        pxp.leaf_id = lid_xp; pxp.leaf_local_face = XP;
                        pxp.s0 = s0; pxp.s1 = s1; pxp.t0 = t0; pxp.t1 = t1;
                        pxp.quad = buildParentFaceSubQuad(p, XP, s0, s1, t0, t1);
                        cov.patches[XP].push_back(pxp);
                    }
                }

                for (uint16_t lk = 0; lk < p.Nrz; ++lk) {
                    for (uint16_t li = 0; li < p.Nrx; ++li) {
                        double s0 = (double)li / (double)p.Nrx;
                        double s1 = (double)(li + 1) / (double)p.Nrx;
                        double t0 = (double)lk / (double)p.Nrz;
                        double t1 = (double)(lk + 1) / (double)p.Nrz;
                        int lid_ym = p.leaf_base + ((int)lk * (int)p.Nry + 0) * (int)p.Nrx + (int)li;
                        int lid_yp = p.leaf_base + ((int)lk * (int)p.Nry + ((int)p.Nry - 1)) * (int)p.Nrx + (int)li;

                        ParentFacePatch pym;
                        pym.leaf_id = lid_ym; pym.leaf_local_face = YM;
                        pym.s0 = s0; pym.s1 = s1; pym.t0 = t0; pym.t1 = t1;
                        pym.quad = buildParentFaceSubQuad(p, YM, s0, s1, t0, t1);
                        cov.patches[YM].push_back(pym);

                        ParentFacePatch pyp;
                        pyp.leaf_id = lid_yp; pyp.leaf_local_face = YP;
                        pyp.s0 = s0; pyp.s1 = s1; pyp.t0 = t0; pyp.t1 = t1;
                        pyp.quad = buildParentFaceSubQuad(p, YP, s0, s1, t0, t1);
                        cov.patches[YP].push_back(pyp);
                    }
                }

                for (uint16_t lj = 0; lj < p.Nry; ++lj) {
                    for (uint16_t li = 0; li < p.Nrx; ++li) {
                        double s0 = (double)li / (double)p.Nrx;
                        double s1 = (double)(li + 1) / (double)p.Nrx;
                        double t0 = (double)lj / (double)p.Nry;
                        double t1 = (double)(lj + 1) / (double)p.Nry;
                        int lid_zm = p.leaf_base + (0 * (int)p.Nry + (int)lj) * (int)p.Nrx + (int)li;
                        int lid_zp = p.leaf_base + (((int)p.Nrz - 1) * (int)p.Nry + (int)lj) * (int)p.Nrx + (int)li;

                        ParentFacePatch pzm;
                        pzm.leaf_id = lid_zm; pzm.leaf_local_face = ZM;
                        pzm.s0 = s0; pzm.s1 = s1; pzm.t0 = t0; pzm.t1 = t1;
                        pzm.quad = buildParentFaceSubQuad(p, ZM, s0, s1, t0, t1);
                        cov.patches[ZM].push_back(pzm);

                        ParentFacePatch pzp;
                        pzp.leaf_id = lid_zp; pzp.leaf_local_face = ZP;
                        pzp.s0 = s0; pzp.s1 = s1; pzp.t0 = t0; pzp.t1 = t1;
                        pzp.quad = buildParentFaceSubQuad(p, ZP, s0, s1, t0, t1);
                        cov.patches[ZP].push_back(pzp);
                    }
                }
            }
            parent_face_cov[p.parent_id] = std::move(cov);
        }
        std::cout << "Parent face coverage built." << std::endl;
    }

    void addMMFace(int owner, int owner_local_face, int neighbor, int neighbor_local_face,
                   const std::array<Point3,4>& quad) {
        FaceGeom f;
        f.id = (int)mm_faces.size();
        f.owner = owner;
        f.neighbor = neighbor;
        f.local_owner_face = owner_local_face;
        f.local_neighbor_face = neighbor_local_face;
        f.vertices = quad;
        computeFaceDerivedGeometry(f, leaves[owner].center);
        mm_faces.push_back(f);
        leaves[owner].subface_ids[owner_local_face].push_back(f.id);
        if (neighbor >= 0) leaves[neighbor].subface_ids[neighbor_local_face].push_back(f.id);
    }

    void finalizeLeafConvenienceFaceIds() {
        for (auto& c : leaves) {
            for (int f = 0; f < 6; ++f) {
                if (c.subface_ids[f].size() == 1) c.face_ids[f] = c.subface_ids[f][0];
                else c.face_ids[f] = -1;
            }
        }
    }

    void buildLeafInterfaceFaces() {
        mm_faces.clear();
        for (auto& c : leaves) {
            c.face_ids = {{-1,-1,-1,-1,-1,-1}};
            for (int f = 0; f < 6; ++f) c.subface_ids[f].clear();
        }

        for (const auto& p : parents) {
            if (!p.refined) continue;
            int Nrx = p.Nrx, Nry = p.Nry, Nrz = p.Nrz;
            int base = p.leaf_base;
            for (int lk = 0; lk < Nrz; ++lk) {
                for (int lj = 0; lj < Nry; ++lj) {
                    for (int li = 0; li < Nrx; ++li) {
                        int lid = base + (lk * Nry + lj) * Nrx + li;
                        if (li + 1 < Nrx) {
                            int rid = lid + 1;
                            auto quad = getCellFaceVertices(leaves[lid], XP);
                            addMMFace(lid, XP, rid, XM, quad);
                        }
                        if (lj + 1 < Nry) {
                            int uid = base + (lk * Nry + (lj + 1)) * Nrx + li;
                            auto quad = getCellFaceVertices(leaves[lid], YP);
                            addMMFace(lid, YP, uid, YM, quad);
                        }
                        if (lk + 1 < Nrz) {
                            int wid = base + ((lk + 1) * Nry + lj) * Nrx + li;
                            auto quad = getCellFaceVertices(leaves[lid], ZP);
                            addMMFace(lid, ZP, wid, ZM, quad);
                        }
                    }
                }
            }
        }

        auto parentAt = [&](int i,int j,int k)->int {
            return k * Nx * Ny + j * Nx + i;
        };
        auto overlapPositive = [](double a0, double a1, double b0, double b1, double tol = 1e-12) -> bool {
            return std::min(a1, b1) - std::max(a0, b0) > tol;
        };

        auto buildInterfaceFromParentPair =
            [&](const ParentCell& pA, int faceA, const ParentCell& pB, int faceB) {
                const auto& patchesA = parent_face_cov[pA.parent_id].patches[faceA];
                const auto& patchesB = parent_face_cov[pB.parent_id].patches[faceB];
                for (const auto& pa : patchesA) {
                    for (const auto& pb : patchesB) {
                        if (!overlapPositive(pa.s0, pa.s1, pb.s0, pb.s1)) continue;
                        if (!overlapPositive(pa.t0, pa.t1, pb.t0, pb.t1)) continue;
                        double s0 = std::max(pa.s0, pb.s0);
                        double s1 = std::min(pa.s1, pb.s1);
                        double t0 = std::max(pa.t0, pb.t0);
                        double t1 = std::min(pa.t1, pb.t1);
                        if (s1 - s0 <= 1e-12 || t1 - t0 <= 1e-12) continue;
                        auto quad = buildParentFaceSubQuad(pA, faceA, s0, s1, t0, t1);
                        addMMFace(pa.leaf_id, pa.leaf_local_face, pb.leaf_id, pb.leaf_local_face, quad);
                    }
                }
            };

        for (int k = 0; k < Nz; ++k) {
            for (int j = 0; j < Ny; ++j) {
                for (int i = 0; i < Nx; ++i) {
                    int A = parentAt(i, j, k);
                    const ParentCell& pA = parents[A];
                    if (i + 1 < Nx) {
                        int B = parentAt(i + 1, j, k);
                        buildInterfaceFromParentPair(pA, XP, parents[B], XM);
                    }
                    if (j + 1 < Ny) {
                        int B = parentAt(i, j + 1, k);
                        buildInterfaceFromParentPair(pA, YP, parents[B], YM);
                    }
                    if (k + 1 < Nz) {
                        int B = parentAt(i, j, k + 1);
                        buildInterfaceFromParentPair(pA, ZP, parents[B], ZM);
                    }
                }
            }
        }

        auto addBoundaryPatches = [&](const ParentCell& p, int face_id) {
            const auto& patches = parent_face_cov[p.parent_id].patches[face_id];
            for (const auto& patch : patches) {
                auto quad = buildParentFaceSubQuad(p, face_id, patch.s0, patch.s1, patch.t0, patch.t1);
                addMMFace(patch.leaf_id, patch.leaf_local_face, -1, -1, quad);
            }
        };

        for (int k = 0; k < Nz; ++k) {
            for (int j = 0; j < Ny; ++j) {
                for (int i = 0; i < Nx; ++i) {
                    const ParentCell& p = parents[parentAt(i,j,k)];
                    if (i == 0) addBoundaryPatches(p, XM);
                    if (i == Nx - 1) addBoundaryPatches(p, XP);
                    if (j == 0) addBoundaryPatches(p, YM);
                    if (j == Ny - 1) addBoundaryPatches(p, YP);
                    if (k == 0) addBoundaryPatches(p, ZM);
                    if (k == Nz - 1) addBoundaryPatches(p, ZP);
                }
            }
        }

        finalizeLeafConvenienceFaceIds();
        std::cout << "Leaf interface faces built: n_mm_faces = " << mm_faces.size() << std::endl;
    }

    LeafCell buildLeafFromParentRefSubcell(const ParentCell& p, uint16_t li, uint16_t lj, uint16_t lk) const {
        LeafCell lc;
        lc.parent_id = p.parent_id;
        lc.lix = li;
        lc.liy = lj;
        lc.liz = lk;
        lc.phi = p.phi;
        lc.K[0] = p.K[0];
        lc.K[1] = p.K[1];
        lc.K[2] = p.K[2];

        double u0 = (double)li / (double)p.Nrx;
        double u1 = (double)(li + 1) / (double)p.Nrx;
        double v0 = (double)lj / (double)p.Nry;
        double v1 = (double)(lj + 1) / (double)p.Nry;
        double w0 = (double)lk / (double)p.Nrz;
        double w1 = (double)(lk + 1) / (double)p.Nrz;

        lc.corners[0] = mapHexTrilinear(p, u0, v0, w0);
        lc.corners[1] = mapHexTrilinear(p, u1, v0, w0);
        lc.corners[2] = mapHexTrilinear(p, u1, v1, w0);
        lc.corners[3] = mapHexTrilinear(p, u0, v1, w0);
        lc.corners[4] = mapHexTrilinear(p, u0, v0, w1);
        lc.corners[5] = mapHexTrilinear(p, u1, v0, w1);
        lc.corners[6] = mapHexTrilinear(p, u1, v1, w1);
        lc.corners[7] = mapHexTrilinear(p, u0, v1, w1);

        lc.face_ids = {{-1,-1,-1,-1,-1,-1}};
        computeCellDerivedGeometry(lc);
        return lc;
    }

    void buildParentGrid() {
        dx = Lx / Nx; dy = Ly / Ny; dz = Lz / Nz;
        int n_parent = Nx * Ny * Nz;
        parents.resize(n_parent);
        for (int k = 0; k < Nz; ++k) {
            for (int j = 0; j < Ny; ++j) {
                for (int i = 0; i < Nx; ++i) {
                    int pid = k * Nx * Ny + j * Nx + i;
                    ParentCell pc;
                    pc.parent_id = pid;
                    pc.ix = i; pc.iy = j; pc.iz = k;
                    pc.phi = 0.04;
                    pc.K[0] = 0.005; pc.K[1] = 0.005; pc.K[2] = 0.005;
                    pc.refined = false;
                    pc.leaf_base = -1;
                    pc.Nrx = lgr_Nrx; pc.Nry = lgr_Nry; pc.Nrz = lgr_Nrz;
                    double x0 = i * dx, x1 = (i + 1) * dx;
                    double y0 = j * dy, y1 = (j + 1) * dy;
                    double z0 = k * dz, z1 = (k + 1) * dz;
                    pc.corners[0] = {x0, y0, z0};
                    pc.corners[1] = {x1, y0, z0};
                    pc.corners[2] = {x1, y1, z0};
                    pc.corners[3] = {x0, y1, z0};
                    pc.corners[4] = {x0, y0, z1};
                    pc.corners[5] = {x1, y0, z1};
                    pc.corners[6] = {x1, y1, z1};
                    pc.corners[7] = {x0, y1, z1};
                    computeCellDerivedGeometry(pc);
                    parents[pid] = pc;
                }
            }
        }
        std::cout << "Parent corner-point grid built: n_parent = " << n_parent << std::endl;
    }

    void generateFractures(int total_fracs = 10,
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
        if (parents.empty()) {
            std::cerr << "No parent cells available for fracture generation." << std::endl;
            return;
        }
        std::vector<int> candidate_cells;
        candidate_cells.reserve(parents.size());
        for (int pid = 0; pid < (int)parents.size(); ++pid) {
            const ParentCell& pc = parents[pid];
            if (pc.bbox_min.x >= use_max_x || pc.bbox_max.x <= range_x_min) continue;
            if (pc.bbox_min.y >= use_max_y || pc.bbox_max.y <= range_y_min) continue;
            if (pc.bbox_min.z >= use_max_z || pc.bbox_max.z <= range_z_min) continue;
            candidate_cells.push_back(pid);
        }
        if (candidate_cells.empty()) {
            std::cerr << "No cells found in the specified range for fracture generation." << std::endl;
            return;
        }
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> distCell(0, (int)candidate_cells.size() - 1);
        std::uniform_real_distribution<double> dist01(0.0, 1.0);
        std::uniform_real_distribution<double> distAngle(min_strike, max_strike);
        std::uniform_real_distribution<double> distDip(0, max_dip);
        for (int i = 0; i < total_fracs; ++i) {
            int cell_idx = distCell(rng);
            int pid = candidate_cells[cell_idx];
            const ParentCell& pc = parents[pid];
            double cell_dx = pc.bbox_max.x - pc.bbox_min.x;
            double cell_dy = pc.bbox_max.y - pc.bbox_min.y;
            double cell_dz = pc.bbox_max.z - pc.bbox_min.z;
            double max_len_in_cell = std::min({cell_dx, cell_dy, cell_dz}) * 0.9;
            double actual_max_L = std::min(max_L, max_len_in_cell);
            double actual_min_L = std::min(min_L, actual_max_L);
            std::uniform_real_distribution<double> distL(actual_min_L, actual_max_L);
            double len = distL(rng);
            double height = len * 0.5;
            double strike = distAngle(rng);
            double dip = distDip(rng);
            double margin_x = len * 0.6;
            double margin_y = len * 0.6;
            double margin_z = height * 0.6;
            double cx_min = pc.bbox_min.x + margin_x;
            double cx_max = pc.bbox_max.x - margin_x;
            double cy_min = pc.bbox_min.y + margin_y;
            double cy_max = pc.bbox_max.y - margin_y;
            double cz_min = pc.bbox_min.z + margin_z;
            double cz_max = pc.bbox_max.z - margin_z;
            if (cx_max < cx_min) { cx_min = cx_max = (pc.bbox_min.x + pc.bbox_max.x) / 2.0; }
            if (cy_max < cy_min) { cy_min = cy_max = (pc.bbox_min.y + pc.bbox_max.y) / 2.0; }
            if (cz_max < cz_min) { cz_min = cz_max = (pc.bbox_min.z + pc.bbox_max.z) / 2.0; }
            double cx = cx_min + dist01(rng) * (cx_max - cx_min);
            double cy = cy_min + dist01(rng) * (cy_max - cy_min);
            double cz = cz_min + dist01(rng) * (cz_max - cz_min);
            Point3 center = {cx, cy, cz};
            Point3 u = {cos(strike), sin(strike), 0};
            Point3 n_horiz = {-sin(strike), cos(strike), 0};
            Point3 v = {n_horiz.x * cos(dip), n_horiz.y * cos(dip), -sin(dip)};
            Fracture f;
            f.id = i;
            f.aperture = aperture_val;
            f.perm = perm_val;
            f.is_hydraulic = false;
            f.vertices[0] = center - u*(len/2) - v*(height/2);
            f.vertices[1] = center + u*(len/2) - v*(height/2);
            f.vertices[2] = center + u*(len/2) + v*(height/2);
            f.vertices[3] = center - u*(len/2) + v*(height/2);
            fractures.push_back(f);
        }
        std::cout << "Generated " << total_fracs << " natural fractures inside grid cells." << std::endl;
    }

    void generateHydraulicFractures(int total_fracs = 20,
                                    double well_length = 600,
                                    double hf_len = 120.0,
                                    double hf_height = 30.0,
                                    double aperture_val = 0.1,
                                    double perm_val = 1000.0,
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
        auto findParentCell = [&](const Point3& p) -> int {
            if (parents.empty()) return -1;
            static Point3 gmin = {0,0,0}, gmax = {0,0,0};
            static bool initialized = false;
            if (!initialized) {
                gmin = parents[0].bbox_min;
                gmax = parents[0].bbox_max;
                for (const auto& pc : parents) {
                    gmin = pointMin(gmin, pc.bbox_min);
                    gmax = pointMax(gmax, pc.bbox_max);
                }
                initialized = true;
            }
            if (p.x < gmin.x || p.x > gmax.x ||
                p.y < gmin.y || p.y > gmax.y ||
                p.z < gmin.z || p.z > gmax.z) {
                return -1;
            }
            int ci = static_cast<int>((p.x - gmin.x) / (gmax.x - gmin.x) * Nx);
            int cj = static_cast<int>((p.y - gmin.y) / (gmax.y - gmin.y) * Ny);
            int ck = static_cast<int>((p.z - gmin.z) / (gmax.z - gmin.z) * Nz);
            ci = std::max(0, std::min(Nx - 1, ci));
            cj = std::max(0, std::min(Ny - 1, cj));
            ck = std::max(0, std::min(Nz - 1, ck));
            const int di_range = 2;
            const int dj_range = 2;
            const int dk_range = 1;
            for (int dk = -dk_range; dk <= dk_range; ++dk) {
                int kk = ck + dk;
                if (kk < 0 || kk >= Nz) continue;
                for (int dj = -dj_range; dj <= dj_range; ++dj) {
                    int jj = cj + dj;
                    if (jj < 0 || jj >= Ny) continue;
                    for (int di = -di_range; di <= di_range; ++di) {
                        int ii = ci + di;
                        if (ii < 0 || ii >= Nx) continue;
                        int pid = kk * Nx * Ny + jj * Nx + ii;
                        const ParentCell& pc = parents[pid];
                        if (p.x >= pc.bbox_min.x && p.x <= pc.bbox_max.x &&
                            p.y >= pc.bbox_min.y && p.y <= pc.bbox_max.y &&
                            p.z >= pc.bbox_min.z && p.z <= pc.bbox_max.z) {
                            return pid;
                        }
                    }
                }
            }
            return -1;
        };
        auto pointInGrid = [&](const Point3& p) -> bool {
            return findParentCell(p) >= 0;
        };
        auto fracVerticesInGrid = [&](const Fracture& f) -> bool {
            return pointInGrid(f.vertices[0]) && pointInGrid(f.vertices[1]) &&
                   pointInGrid(f.vertices[2]) && pointInGrid(f.vertices[3]);
        };
        int base_id = (start_id >= 0) ? start_id : getNextFractureId();
        double spacing = 0.0;
        double x_start = xc;
        if (total_fracs > 1) {
            double eps_x=0.1;
            x_start = xc - well_length / 2.0 + eps_x;
            double x_end = xc + well_length / 2.0 - eps_x;
            spacing = (x_end - x_start) / (total_fracs - 1);
        }
        int valid_count = 0;
        for (int k = 0; k < total_fracs; ++k) {
            Fracture f;
            f.id = base_id + valid_count;
            f.aperture = aperture_val;
            f.perm = perm_val;
            f.is_hydraulic = true;
            double x_curr = (total_fracs == 1) ? xc : (x_start + k * spacing);
            f.vertices[0] = {x_curr, yc - hf_len/2.0, zc - hf_height/2.0};
            f.vertices[1] = {x_curr, yc + hf_len/2.0, zc - hf_height/2.0};
            f.vertices[2] = {x_curr, yc + hf_len/2.0, zc + hf_height/2.0};
            f.vertices[3] = {x_curr, yc - hf_len/2.0, zc + hf_height/2.0};
            if (fracVerticesInGrid(f)) {
                fractures.push_back(f);
                valid_count++;
            } else {
                std::cerr << "Warning: Hydraulic fracture " << k << " at x=" << x_curr
                          << " exceeds actual grid bounds, skipped." << std::endl;
            }
        }
        std::cout << "Generated " << valid_count << " hydraulic fractures ("
                  << total_fracs << " requested). "
                  << "ID range: [" << base_id << ", " << (base_id + valid_count - 1) << "]" << std::endl;
    }

    static AABB fractureAABB(const Fracture& f) {
        AABB b;
        b.mn = { 1e30,  1e30,  1e30};
        b.mx = {-1e30, -1e30, -1e30};
        for (int i=0;i<4;++i) {
            b.mn.x = std::min(b.mn.x, f.vertices[i].x);
            b.mn.y = std::min(b.mn.y, f.vertices[i].y);
            b.mn.z = std::min(b.mn.z, f.vertices[i].z);
            b.mx.x = std::max(b.mx.x, f.vertices[i].x);
            b.mx.y = std::max(b.mx.y, f.vertices[i].y);
            b.mx.z = std::max(b.mx.z, f.vertices[i].z);
        }
        return b;
    }

    double epsAreaForBox(double dx_, double dy_, double dz_) const {
        double a1 = dx_*dy_, a2 = dx_*dz_, a3 = dy_*dz_;
        double minA = std::min(a1, std::min(a2, a3));
        return eps_area_factor * minA;
    }

    void markRefinement() {
        if (!enable_lgr) {
            for (auto& p : parents) p.refined = false;
            return;
        }
        for (auto& p : parents) p.refined = false;
        for (const auto& f : fractures) {
            AABB fb = fractureAABB(f);
            AABB fb_exp = expandAABB(fb, d_threshold);
            int i0 = std::max(0, (int)std::floor(fb_exp.mn.x / std::max(dx, 1e-12)));
            int i1 = std::min(Nx - 1, (int)std::floor(fb_exp.mx.x / std::max(dx, 1e-12)));
            int j0 = std::max(0, (int)std::floor(fb_exp.mn.y / std::max(dy, 1e-12)));
            int j1 = std::min(Ny - 1, (int)std::floor(fb_exp.mx.y / std::max(dy, 1e-12)));
            int k0 = std::max(0, (int)std::floor(fb_exp.mn.z / std::max(dz, 1e-12)));
            int k1 = std::min(Nz - 1, (int)std::floor(fb_exp.mx.z / std::max(dz, 1e-12)));
            for (int k = k0; k <= k1; ++k) {
                for (int j = j0; j <= j1; ++j) {
                    for (int i = i0; i <= i1; ++i) {
                        int pid = k * Nx * Ny + j * Nx + i;
                        ParentCell& pc = parents[pid];
                        if (pc.refined) continue;
                        if (pc.bbox_max.x < fb.mn.x - d_threshold || fb.mx.x < pc.bbox_min.x - d_threshold) continue;
                        if (pc.bbox_max.y < fb.mn.y - d_threshold || fb.mx.y < pc.bbox_min.y - d_threshold) continue;
                        if (pc.bbox_max.z < fb.mn.z - d_threshold || fb.mx.z < pc.bbox_min.z - d_threshold) continue;
                        double epsA = epsAreaForBox(std::max(pc.dx, 1e-12), std::max(pc.dy, 1e-12), std::max(pc.dz, 1e-12));
                        auto poly = clipFractureCell(f, pc);
                        double area = polygonArea(poly);
                        if (area > epsA) {
                            pc.refined = true;
                            continue;
                        }
                        double dbar = dbarHexQuad(pc, f, dbar_nsx, dbar_nsy, dbar_nsz);
                        if (dbar < d_threshold) pc.refined = true;
                    }
                }
            }
        }
        int cnt = 0;
        for (const auto& p : parents) if (p.refined) cnt++;
        std::cout << "Refinement marked (real hex): refined parents = "
                  << cnt << " / " << (int)parents.size() << std::endl;
    }

    void buildLeafGrid() {
        n_leaf = 0;
        for (auto& p : parents) {
            p.leaf_base = n_leaf;
            if (p.refined) n_leaf += (int)p.Nrx * (int)p.Nry * (int)p.Nrz;
            else n_leaf += 1;
        }
        leaves.clear();
        leaves.resize(n_leaf);
        for (const auto& p : parents) {
            if (!p.refined) {
                int lid = p.leaf_base;
                LeafCell lc;
                lc.leaf_id = lid;
                lc.parent_id = p.parent_id;
                lc.lix = 0; lc.liy = 0; lc.liz = 0;
                lc.phi = p.phi;
                lc.K[0] = p.K[0]; lc.K[1] = p.K[1]; lc.K[2] = p.K[2];
                lc.corners = p.corners;
                lc.face_ids = {{-1,-1,-1,-1,-1,-1}};
                computeCellDerivedGeometry(lc);
                leaves[lid] = lc;
            } else {
                for (uint16_t lk = 0; lk < p.Nrz; ++lk) {
                    for (uint16_t lj = 0; lj < p.Nry; ++lj) {
                        for (uint16_t li = 0; li < p.Nrx; ++li) {
                            int lid = p.leaf_base + ((int)lk * (int)p.Nry + (int)lj) * (int)p.Nrx + (int)li;
                            LeafCell lc = buildLeafFromParentRefSubcell(p, li, lj, lk);
                            lc.leaf_id = lid;
                            leaves[lid] = lc;
                        }
                    }
                }
            }
        }
        std::cout << "Leaf corner-point grid built: n_leaf = " << n_leaf << std::endl;
    }

    void buildMMConnections(std::vector<Connection>& mm_out) {
        mm_out.clear();
        leaf_neighbors.clear();
        leaf_neighbors.resize(n_leaf);
        mm_out.reserve(mm_faces.size());
        for (const auto& f : mm_faces) {
            if (f.owner < 0 || f.neighbor < 0) continue;
            int u = f.owner, v = f.neighbor;
            double T = computeMMTransmissibilityTPFA(leaves[u], leaves[v], f);
            if (T <= EPS) continue;
            int a = std::min(u, v);
            int b = std::max(u, v);
            mm_out.push_back({a, b, T, 0});
            leaf_neighbors[u].push_back(v);
            leaf_neighbors[v].push_back(u);
        }
        for (auto& nb : leaf_neighbors) {
            std::sort(nb.begin(), nb.end());
            nb.erase(std::unique(nb.begin(), nb.end()), nb.end());
        }
        std::cout << "MM edges built from mm_faces: " << mm_out.size() << std::endl;
    }

    void buildSegmentsAndMF(std::vector<Connection>& mf_out) {
        segments.clear();
        mf_out.clear();
        leaf_to_segs.clear();
        leaf_to_segs.resize(n_leaf);
        int seg_id_counter = 0;

        for (const auto& frac : fractures) {
            AABB fb = fractureAABB(frac);
            int i0 = std::max(0, (int)std::floor(fb.mn.x / std::max(dx, 1e-12)));
            int i1 = std::min(Nx - 1, (int)std::floor(fb.mx.x / std::max(dx, 1e-12)));
            int j0 = std::max(0, (int)std::floor(fb.mn.y / std::max(dy, 1e-12)));
            int j1 = std::min(Ny - 1, (int)std::floor(fb.mx.y / std::max(dy, 1e-12)));
            int k0 = std::max(0, (int)std::floor(fb.mn.z / std::max(dz, 1e-12)));
            int k1 = std::min(Nz - 1, (int)std::floor(fb.mx.z / std::max(dz, 1e-12)));

            Point3 vec1 = frac.vertices[1] - frac.vertices[0];
            Point3 vec2 = frac.vertices[3] - frac.vertices[0];
            Point3 normal = vec1.cross(vec2);
            double nlen = normal.norm();
            if (nlen < EPS) continue;
            normal = normal * (1.0 / nlen);

            for (int k = k0; k <= k1; ++k) {
                for (int j = j0; j <= j1; ++j) {
                    for (int i = i0; i <= i1; ++i) {
                        int pid = k * Nx * Ny + j * Nx + i;
                        const ParentCell& p = parents[pid];
                        if (p.bbox_max.x < fb.mn.x || fb.mx.x < p.bbox_min.x) continue;
                        if (p.bbox_max.y < fb.mn.y || fb.mx.y < p.bbox_min.y) continue;
                        if (p.bbox_max.z < fb.mn.z || fb.mx.z < p.bbox_min.z) continue;

                        auto try_build_seg_on_leaf = [&](int lid) {
                            const LeafCell& lc = leaves[lid];
                            if (lc.bbox_max.x < fb.mn.x || fb.mx.x < lc.bbox_min.x) return;
                            if (lc.bbox_max.y < fb.mn.y || fb.mx.y < lc.bbox_min.y) return;
                            if (lc.bbox_max.z < fb.mn.z || fb.mx.z < lc.bbox_min.z) return;
                            auto poly = clipFractureCell(frac, lc);
                            if (poly.size() < 3) return;
                            double area = polygonArea(poly);
                            double epsA = epsAreaForBox(std::max(lc.dx, 1e-12), std::max(lc.dy, 1e-12), std::max(lc.dz, 1e-12));
                            if (area <= epsA) return;

                            Segment seg;
                            seg.id = seg_id_counter++;
                            seg.frac_id = frac.id;
                            seg.matrix_leaf_id = lid;
                            seg.parent_id = pid;
                            seg.area = area;
                            seg.center = polygonCenter(poly);
                            seg.normal = normal;
                            seg.aperture = frac.aperture;
                            seg.perm = frac.perm;
                            seg.poly = poly;
                            seg.T_mf = computeMatrixFractureTransmissibility(lc, seg, 2, 2, 2);
                            fillSegmentFaceGeom(seg, lc, mm_faces);

                            segments.push_back(seg);
                            leaf_to_segs[lid].push_back((int)segments.size() - 1);
                        };

                        if (!p.refined) {
                            try_build_seg_on_leaf(p.leaf_base);
                        } else {
                            int nsub = (int)p.Nrx * (int)p.Nry * (int)p.Nrz;
                            for (int off = 0; off < nsub; ++off) try_build_seg_on_leaf(p.leaf_base + off);
                        }
                    }
                }
            }
        }

        n_seg = (int)segments.size();
        std::cout << "Segments built on real leaf hexes: n_seg = " << n_seg << std::endl;
        mf_out.reserve(n_seg);
        for (int s = 0; s < n_seg; ++s) {
            int u = segments[s].matrix_leaf_id;
            int v = n_leaf + s;
            if (segments[s].T_mf > EPS) mf_out.push_back({std::min(u, v), std::max(u, v), segments[s].T_mf, 1});
        }
        std::cout << "MF edges built: " << mf_out.size() << std::endl;
    }

    void buildFFConnections(std::vector<Connection>& ff_out) {
        ff_out.clear();
        for (const auto& fg : mm_faces) {
            if (fg.owner < 0 || fg.neighbor < 0) continue;
            int leaf_u = fg.owner;
            int leaf_v = fg.neighbor;
            int sfid = fg.id;
            const auto& segs_u = leaf_to_segs[leaf_u];
            const auto& segs_v = leaf_to_segs[leaf_v];
            if (segs_u.empty() || segs_v.empty()) continue;

            for (int s1 : segs_u) {
                auto it_len1 = segments[s1].subface_trace_len.find(sfid);
                auto it_dist1 = segments[s1].subface_center_dist.find(sfid);
                if (it_len1 == segments[s1].subface_trace_len.end()) continue;
                if (it_dist1 == segments[s1].subface_center_dist.end()) continue;

                for (int s2 : segs_v) {
                    if (segments[s1].frac_id != segments[s2].frac_id) continue;
                    auto it_len2 = segments[s2].subface_trace_len.find(sfid);
                    auto it_dist2 = segments[s2].subface_center_dist.find(sfid);
                    if (it_len2 == segments[s2].subface_trace_len.end()) continue;
                    if (it_dist2 == segments[s2].subface_center_dist.end()) continue;

                    double ell1 = it_len1->second;
                    double ell2 = it_len2->second;
                    if (ell1 <= EPS || ell2 <= EPS) continue;

                    double ell = std::min(ell1, ell2);
                    double L1 = std::max(it_dist1->second, 1e-10);
                    double L2 = std::max(it_dist2->second, 1e-10);
                    double Af1 = segments[s1].aperture * ell;
                    double Af2 = segments[s2].aperture * ell;
                    if (Af1 <= EPS || Af2 <= EPS) continue;

                    double tau1 = segments[s1].perm * Af1 / L1;
                    double tau2 = segments[s2].perm * Af2 / L2;
                    if (tau1 <= EPS || tau2 <= EPS) continue;

                    double Tff = (tau1 * tau2) / std::max(EPS, tau1 + tau2);
                    if (Tff <= EPS) continue;

                    int u = n_leaf + s1;
                    int v = n_leaf + s2;
                    ff_out.push_back({std::min(u, v), std::max(u, v), Tff, 2});
                }
            }
        }

        for (int leaf = 0; leaf < n_leaf; ++leaf) {
            const auto& segs = leaf_to_segs[leaf];
            if (segs.size() < 2) continue;
            for (size_t i = 0; i < segs.size(); ++i) {
                for (size_t j = i + 1; j < segs.size(); ++j) {
                    int s1 = segs[i];
                    int s2 = segs[j];
                    if (segments[s1].frac_id == segments[s2].frac_id) continue;
                    double ell_int = 0.0, L1 = 0.0, L2 = 0.0;
                    if (!computeCrossGeom(segments[s1], segments[s2], ell_int, L1, L2)) continue;
                    if (ell_int <= EPS) continue;
                    double A1 = segments[s1].aperture * ell_int;
                    double A2 = segments[s2].aperture * ell_int;
                    if (A1 <= EPS || A2 <= EPS) continue;
                    double tau1 = segments[s1].perm * A1 / std::max(L1, 1e-10);
                    double tau2 = segments[s2].perm * A2 / std::max(L2, 1e-10);
                    if (tau1 <= EPS || tau2 <= EPS) continue;
                    double T_cross = (tau1 * tau2) / std::max(EPS, tau1 + tau2);
                    if (T_cross <= EPS) continue;
                    int u = n_leaf + s1;
                    int v = n_leaf + s2;
                    ff_out.push_back({std::min(u, v), std::max(u, v), T_cross, 2});
                }
            }
        }

        std::cout << "FF edges built (real-geometry): " << ff_out.size() << std::endl;
    }

    struct ConnKey {
        int type;
        int u;
        int v;
        bool operator==(const ConnKey& o) const { return type==o.type && u==o.u && v==o.v; }
    };
    struct ConnKeyHash {
        size_t operator()(const ConnKey& k) const {
            uint64_t x = (uint64_t)(k.type & 0xFF);
            x = (x << 28) ^ (uint64_t)k.u;
            x = (x << 28) ^ (uint64_t)k.v;
            x += 0x9e3779b97f4a7c15ULL;
            x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
            x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
            x = x ^ (x >> 31);
            return (size_t)x;
        }
    };

    void buildAllConnections(const std::vector<Connection>& mm,
                             const std::vector<Connection>& mf,
                             const std::vector<Connection>& ff) {
        n_total = n_leaf + n_seg;
        connections.clear();
        connections.reserve(mm.size() + mf.size() + ff.size());
        std::unordered_set<ConnKey, ConnKeyHash> seen;
        seen.reserve((mm.size() + mf.size() + ff.size()) * 2 + 16);

        auto push_unique = [&](const Connection& c) {
            int u = std::min(c.u, c.v);
            int v = std::max(c.u, c.v);
            ConnKey key{c.type, u, v};
            if (seen.insert(key).second) connections.push_back({u, v, c.T, c.type});
        };
        for (const auto& c : mm) push_unique(c);
        for (const auto& c : mf) push_unique(c);
        for (const auto& c : ff) push_unique(c);

        adj.assign(n_total, {});
        for (const auto& c : connections) {
            adj[c.u].push_back({c.v, c.T, c.type});
            adj[c.v].push_back({c.u, c.T, c.type});
        }
        std::cout << "Connections built (unique): " << connections.size() << std::endl;
    }

    void setupWells() {
        wells.clear();
        well_map.clear();
        std::vector<int> target_fracs;
        std::unordered_map<int, Point3> frac_center_map;
        frac_center_map.reserve(fractures.size() * 2 + 1);

        for (const auto& f : fractures) {
            if (f.is_hydraulic) {
                target_fracs.push_back(f.id);
                frac_center_map[f.id] = fractureCenter(f);
            }
        }
        std::sort(target_fracs.begin(), target_fracs.end());

        for (int fid : target_fracs) {
            std::vector<int> cands;
            for (int s = 0; s < n_seg; ++s) if (segments[s].frac_id == fid) cands.push_back(s);
            if (cands.empty()) continue;

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

            if (best_s >= 0) {
                double rw = well_radius;
                double re = computeSegmentEquivalentRadius(segments[best_s], rw);
                double kf = segments[best_s].perm;
                double b  = segments[best_s].aperture;
                double denom = std::log(re / rw);
                if (denom <= EPS) continue;

                Well w;
                w.target_node_idx = n_leaf + best_s;
                w.WI = 2.0 * PI * kf * b / denom;
                w.P_bhp = well_pressure;
                if (!std::isfinite(w.WI) || w.WI <= EPS) continue;

                int idx = (int)wells.size();
                wells.push_back(w);
                well_map[w.target_node_idx] = idx;
            }
        }
        std::cout << "Setup wells (segment-based WI) = " << wells.size() << std::endl;
    }

    void initState() {
        states.assign(n_total, {});
        states_prev = states;
        for (int i=0; i<n_total; ++i) {
            states[i].P = initial_pressure;
            states[i].Sw = initial_sw;
            states[i].Sg = initial_sg;
        }
        states_prev = states;
    }

    py::array_t<double> getPressureData() const {
        py::array_t<double> result(std::vector<py::ssize_t>{static_cast<py::ssize_t>(n_leaf), 4});
        auto r = result.mutable_unchecked<2>();
        for (int i = 0; i < n_leaf; ++i) {
            r(i, 0) = leaves[i].center.x;
            r(i, 1) = leaves[i].center.y;
            r(i, 2) = leaves[i].center.z;
            r(i, 3) = states[i].P;
        }
        return result;
    }

    py::array_t<double> getFractureVertices() const {
        py::array_t<double> result(std::vector<py::ssize_t>{static_cast<py::ssize_t>(fractures.size()) * 4, 4});
        auto r = result.mutable_unchecked<2>();
        py::ssize_t row = 0;
        for (const auto& f : fractures) {
            for (int i = 0; i < 4; ++i) {
                r(row, 0) = f.vertices[i].x;
                r(row, 1) = f.vertices[i].y;
                r(row, 2) = f.vertices[i].z;
                r(row, 3) = static_cast<double>(f.id);
                ++row;
            }
        }
        return result;
    }

    py::array_t<double> getCellGeometryWithPressure() const {
        py::array_t<double> result(std::vector<py::ssize_t>{static_cast<py::ssize_t>(n_leaf), 29});
        auto r = result.mutable_unchecked<2>();
        for (int i = 0; i < n_leaf; ++i) {
            const auto& c = leaves[i];
            r(i, 0) = static_cast<double>(c.leaf_id);
            r(i, 1) = static_cast<double>(c.parent_id);
            r(i, 2) = static_cast<double>(c.lix);
            r(i, 3) = static_cast<double>(c.liy);
            for (int j = 0; j < 8; ++j) {
                r(i, 4 + j*3 + 0) = c.corners[j].x;
                r(i, 4 + j*3 + 1) = c.corners[j].y;
                r(i, 4 + j*3 + 2) = c.corners[j].z;
            }
            r(i, 28) = states[i].P;
        }
        return result;
    }

    // 获取LGR加密网格的几何信息（用于可视化加密网格）
    py::array_t<double> getLGRGridGeometry() const {
        // 返回每个leaf cell的8个角点坐标，格式: [n_leaf, 24] (8个点 * 3个坐标)
        py::array_t<double> result(std::vector<py::ssize_t>{static_cast<py::ssize_t>(n_leaf), 24});
        auto r = result.mutable_unchecked<2>();
        for (int i = 0; i < n_leaf; ++i) {
            const auto& c = leaves[i];
            for (int j = 0; j < 8; ++j) {
                r(i, j*3 + 0) = c.corners[j].x;
                r(i, j*3 + 1) = c.corners[j].y;
                r(i, j*3 + 2) = c.corners[j].z;
            }
        }
        return result;
    }

    // 获取父网格的几何信息（未被加密的父单元格）
    py::array_t<double> getParentGridGeometry() const {
        // 只返回未被加密的父单元格
        int n_parent_cells = 0;
        for (const auto& p : parents) {
            if (!p.refined) n_parent_cells++;
        }

        py::array_t<double> result(std::vector<py::ssize_t>{n_parent_cells, 24});
        auto r = result.mutable_unchecked<2>();
        int idx = 0;
        for (const auto& p : parents) {
            if (p.refined) continue;
            // 获取父单元格的8个角点
            for (int j = 0; j < 8; ++j) {
                r(idx, j*3 + 0) = p.corners[j].x;
                r(idx, j*3 + 1) = p.corners[j].y;
                r(idx, j*3 + 2) = p.corners[j].z;
            }
            idx++;
        }
        return result;
    }

    // 获取加密后的子网格几何信息（被加密的父单元格的子单元格）
    py::array_t<double> getRefinedGridGeometry() const {
        // 只返回被加密的父单元格的子单元格
        int n_refined_cells = 0;
        for (const auto& p : parents) {
            if (p.refined) {
                n_refined_cells += p.Nrx * p.Nry * p.Nrz;
            }
        }

        py::array_t<double> result(std::vector<py::ssize_t>{n_refined_cells, 24});
        auto r = result.mutable_unchecked<2>();
        int idx = 0;
        for (const auto& p : parents) {
            if (!p.refined) continue;
            // 获取该父单元格的所有子单元格
            int nsub = p.Nrx * p.Nry * p.Nrz;
            for (int off = 0; off < nsub; ++off) {
                int lid = p.leaf_base + off;
                const auto& c = leaves[lid];
                for (int j = 0; j < 8; ++j) {
                    r(idx, j*3 + 0) = c.corners[j].x;
                    r(idx, j*3 + 1) = c.corners[j].y;
                    r(idx, j*3 + 2) = c.corners[j].z;
                }
                idx++;
            }
        }
        return result;
    }

    SimulationResult runSimulation() {
        if (coord_file_path.empty() || zcorn_file_path.empty()) {
            throw std::runtime_error("Corner-point input files are not set. Call setCornerPointFiles(coord, zcorn) first.");
        }

        std::cout << "Preprocessing (corner-point CSV parent grid + LGR + EDFM)..." << std::endl;
        if (!preprocess()) {
            throw std::runtime_error("Preprocess failed.");
        }

        std::cout << "Running simulation..." << std::endl;
        run(simulation_total_days);

        SimulationResult result;
        result.pressure_field = getPressureData();
        result.fracture_vertices = getFractureVertices();
        return result;
    }

    void buildJacobianPattern() {
        J.resize(3*n_total, 3*n_total);
        std::vector<Triplet<double>> trips;
        trips.reserve(n_total * 9 + connections.size() * 18);
        for(int i=0; i<n_total; ++i) {
            for(int eq=0; eq<3; ++eq) {
                for(int var=0; var<3; ++var) trips.emplace_back(3*i+eq, 3*i+var, 0.0);
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
            for (int k = col_start; k < col_end; ++k) if (J.innerIndexPtr()[k] == r) return k;
            return -1;
        };

        cell_J_idx.resize(n_total);
        for(int i=0; i<n_total; ++i) {
            for(int eq=0; eq<3; ++eq) {
                for(int var=0; var<3; ++var) cell_J_idx[i].diag[eq][var] = get_val_idx(3*i+eq, 3*i+var);
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

    Eigen::Matrix<AD3, 3, 1> computeAccumulation_AD(double dt, const State& s_old_val, const StateAD3& s_new,
                                                    const PropertiesT<AD3>& p_new, double vol, double phi) const {
        StateAD3 s_old;
        s_old.P.value() = s_old_val.P;   s_old.P.derivatives().setZero();
        s_old.Sw.value() = s_old_val.Sw; s_old.Sw.derivatives().setZero();
        s_old.Sg.value() = s_old_val.Sg; s_old.Sg.derivatives().setZero();

        PropertiesT<AD3> p_old = getProps(s_old);
        AD3 accum = vol * phi / dt;
        Eigen::Matrix<AD3, 3, 1> R;
        R(0) = accum * (s_new.Sw / p_new.Bw - s_old.Sw / p_old.Bw);
        R(1) = accum * ((AD3(1.0) - s_new.Sw - s_new.Sg) / p_new.Bo -
                        (AD3(1.0) - s_old.Sw - s_old.Sg) / p_old.Bo);
        R(2) = accum * (s_new.Sg / p_new.Bg - s_old.Sg / p_old.Bg);
        return R;
    }

    Eigen::Matrix<AD3, 3, 1> computeWell_AD(const Well& w, const StateAD3& s_new,
                                            const PropertiesT<AD3>& pu) const {
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

    FluxAD computeFlux_FastAD(double T_trans, const StateAD3& su, const StateAD3& sv,
                              const PropertiesT<AD3>& pu, const PropertiesT<AD3>& pv) const {
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

    bool solveStep(double dt, double& step_oil, double& step_water, double& step_gas, int& actual_iter) {
        const int max_iter = 15;
        const double tol = 1e-3;
        std::vector<State> backup = states;
        std::vector<StateAD3> states_ad(n_total);
        std::vector<PropertiesT<AD3>> props_ad(n_total);

        for (int iter=0; iter<max_iter; ++iter) {
            actual_iter = iter + 1;
            VectorXd Rg(3*n_total);
            Rg.setZero();
            std::fill(J.valuePtr(), J.valuePtr() + J.nonZeros(), 0.0);

            for (int i = 0; i < n_total; ++i) {
                states_ad[i].P.value() = states[i].P;    states_ad[i].P.derivatives()  = Eigen::Vector3d::Unit(0);
                states_ad[i].Sw.value() = states[i].Sw;  states_ad[i].Sw.derivatives() = Eigen::Vector3d::Unit(1);
                states_ad[i].Sg.value() = states[i].Sg;  states_ad[i].Sg.derivatives() = Eigen::Vector3d::Unit(2);
                props_ad[i] = getProps(states_ad[i]);
            }

            for (int i=0; i<n_total; ++i) {
                double vol = (i < n_leaf) ? leaves[i].vol : (segments[i - n_leaf].area * segments[i - n_leaf].aperture);
                double phi = (i < n_leaf) ? leaves[i].phi : 1.0;
                auto R_acc = computeAccumulation_AD(dt, states_prev[i], states_ad[i], props_ad[i], vol, phi);

                auto it = well_map.find(i);
                if (it != well_map.end()) {
                    auto R_well = computeWell_AD(wells[it->second], states_ad[i], props_ad[i]);
                    R_acc(0) += R_well(0);
                    R_acc(1) += R_well(1);
                    R_acc(2) += R_well(2);
                }

                for (int eq = 0; eq < 3; ++eq) {
                    Rg(3*i + eq) += R_acc(eq).value();
                    Deriv3 derivs = R_acc(eq).derivatives();
                    for (int var = 0; var < 3; ++var) {
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

            double max_res = Rg.lpNorm<Infinity>();
            if (max_res < tol) {
                for (const auto& w : wells) {
                    int u = w.target_node_idx;
                    double dP = states[u].P - w.P_bhp;
                    if (dP > 0.0) {
                        PropertiesT<double> pu = getProps(states[u]);
                        step_water += w.WI * pu.lw * dP * dt;
                        step_oil   += w.WI * pu.lo * dP * dt;
                        step_gas   += w.WI * pu.lg * dP * dt;
                    }
                }
                return true;
            }

            if (iter == 0) {
                std::ofstream jac_file("jacobian_sparsity.csv");
                jac_file << "row,col,val\n";
                for (int k = 0; k < J.outerSize(); ++k) {
                    for (SparseMatrix<double>::InnerIterator it(J, k); it; ++it) {
                        if (std::abs(it.value()) > 1e-12) jac_file << it.row() << "," << it.col() << "," << it.value() << "\n";
                    }
                }
                jac_file.close();
            }

            std::cout << "\n      [Newton Iter " << iter+1 << "] 计算ILU预条件子..." << std::flush;
            BiCGSTAB<SparseMatrix<double>, IncompleteLUT<double>> solver;
            solver.preconditioner().setDroptol(1e-5);
            solver.preconditioner().setFillfactor(40);
            solver.setTolerance(1e-5);
            solver.setMaxIterations(500);
            solver.compute(J);
            if (solver.info() != Success) {
                std::cout << " ILU分解失败!" << std::endl;
                states = backup;
                return false;
            }

            std::cout << "完成! 求解线性方程组..." << std::flush;
            VectorXd delta = solver.solve(-Rg);
            if (solver.info() != Success) {
                std::cout << " 求解失败! (迭代次数: " << solver.iterations() << ")" << std::endl;
                states = backup;
                return false;
            } else {
                std::cout << "成功! (迭代次数: " << solver.iterations() << ", 误差: " << solver.error() << ")" << std::endl;
            }

            std::vector<State> states_before_ls = states;
            double alpha = 1.0;
            for (int i=0; i<n_total; ++i) {
                double dP_new  = delta(3*i+0) * alpha;
                double dSw_new = delta(3*i+1) * alpha;
                double dSg_new = delta(3*i+2) * alpha;

                double lw = props_ad[i].lw.value();
                double lo = props_ad[i].lo.value();
                double lg = props_ad[i].lg.value();
                double lt = std::max(1e-20, lw + lo + lg);

                double dlw_dSw = props_ad[i].lw.derivatives()(1);
                double dlo_dSw = props_ad[i].lo.derivatives()(1);
                double dlg_dSw = props_ad[i].lg.derivatives()(1);
                double dlt_dSw = dlw_dSw + dlo_dSw + dlg_dSw;

                double dlw_dSg = props_ad[i].lw.derivatives()(2);
                double dlo_dSg = props_ad[i].lo.derivatives()(2);
                double dlg_dSg = props_ad[i].lg.derivatives()(2);
                double dlt_dSg = dlw_dSg + dlo_dSg + dlg_dSg;

                double dfw_dSw = (dlw_dSw * lt - lw * dlt_dSw) / (lt * lt);
                double dfg_dSg = (dlg_dSg * lt - lg * dlt_dSg) / (lt * lt);

                const double F_tol = 0.15;
                double omega_w = 1.0;
                double omega_g = 1.0;
                if (std::abs(dfw_dSw * dSw_new) > F_tol) omega_w = F_tol / std::max(1e-12, std::abs(dfw_dSw * dSw_new));
                if (std::abs(dfg_dSg * dSg_new) > F_tol) omega_g = F_tol / std::max(1e-12, std::abs(dfg_dSg * dSg_new));
                double omega_appleyard = std::min(omega_w, omega_g);

                const double MAX_DP = 50.0;
                const double MAX_DS = 0.20;
                double omega_P = (std::abs(dP_new) > MAX_DP) ? (MAX_DP / std::abs(dP_new)) : 1.0;

                double dSw_chopped = dSw_new * omega_appleyard;
                double dSg_chopped = dSg_new * omega_appleyard;
                double omega_S = 1.0;
                if (std::abs(dSw_chopped) > MAX_DS) omega_S = std::min(omega_S, MAX_DS / std::abs(dSw_chopped));
                if (std::abs(dSg_chopped) > MAX_DS) omega_S = std::min(omega_S, MAX_DS / std::abs(dSg_chopped));
                double omega_final = std::min({omega_appleyard, omega_P, omega_S});

                states[i].P  = states_before_ls[i].P  + dP_new * omega_final;
                states[i].Sw = states_before_ls[i].Sw + dSw_new * omega_final;
                states[i].Sg = states_before_ls[i].Sg + dSg_new * omega_final;

                states[i].P = std::max(14.7, states[i].P);
                states[i].Sw = clamp01(states[i].Sw);
                states[i].Sg = clamp01(states[i].Sg);

                if (states[i].Sw + states[i].Sg > 1.0 - 1e-6) {
                    double ssum = states[i].Sw + states[i].Sg;
                    states[i].Sw /= ssum;
                    states[i].Sg /= ssum;
                }
            }
        }

        states = backup;
        return false;
    }

    void checkGeometryConsistency() {
        double max_rel_err = 0.0;
        int worst_pid = -1;
        for (const auto& p : parents) {
            double v_sum = 0.0;
            if (!p.refined) v_sum = leaves[p.leaf_base].vol;
            else {
                int nsub = (int)p.Nrx * (int)p.Nry * (int)p.Nrz;
                for (int off = 0; off < nsub; ++off) v_sum += leaves[p.leaf_base + off].vol;
            }
            double rel = std::abs(v_sum - p.vol) / std::max(1e-12, std::abs(p.vol));
            if (rel > max_rel_err) {
                max_rel_err = rel;
                worst_pid = p.parent_id;
            }
        }
        std::cout << "[Geometry Check] max parent-volume relative error = "
                  << max_rel_err << " @ parent " << worst_pid << std::endl;

        double sum_int_area = 0.0;
        int n_internal = 0, n_boundary = 0;
        for (const auto& f : mm_faces) {
            if (f.owner >= 0 && f.neighbor >= 0) { sum_int_area += f.area; n_internal++; }
            else n_boundary++;
        }
        std::cout << "[Geometry Check] mm_faces: internal = " << n_internal
                  << ", boundary = " << n_boundary
                  << ", sum_internal_area = " << sum_int_area << std::endl;

        long long n_trace = 0;
        for (const auto& s : segments) n_trace += (long long)s.subface_trace_len.size();
        std::cout << "[Geometry Check] total segment-subface traces = " << n_trace << std::endl;
    }

    void exportStaticGeometry() {
        std::ofstream gridFile("grid_info_lgr.csv");
        gridFile << Nx << "," << Ny << "," << Nz << ","
                 << Lx << "," << Ly << "," << Lz << ","
                 << dx << "," << dy << "," << dz << "\n";
        gridFile.close();

        std::ofstream pf("parent_cells_lgr.csv");
        pf << "parent_id,ix,iy,iz,refined,leaf_base,Nrx,Nry,Nrz,"
           << "cx,cy,cz,vol,"
           << "bbox_min_x,bbox_min_y,bbox_min_z,"
           << "bbox_max_x,bbox_max_y,bbox_max_z,";
        for (int n = 0; n < 8; ++n) pf << "c" << n << "_x,c" << n << "_y,c" << n << "_z,";
        pf << "face0,face1,face2,face3,face4,face5\n";
        for (const auto& p : parents) {
            pf << p.parent_id << "," << p.ix << "," << p.iy << "," << p.iz << ","
               << (p.refined ? 1 : 0) << "," << p.leaf_base << ","
               << p.Nrx << "," << p.Nry << "," << p.Nrz << ","
               << p.center.x << "," << p.center.y << "," << p.center.z << ","
               << p.vol << ","
               << p.bbox_min.x << "," << p.bbox_min.y << "," << p.bbox_min.z << ","
               << p.bbox_max.x << "," << p.bbox_max.y << "," << p.bbox_max.z << ",";
            for (int n = 0; n < 8; ++n) pf << p.corners[n].x << "," << p.corners[n].y << "," << p.corners[n].z << ",";
            pf << p.face_ids[0] << "," << p.face_ids[1] << "," << p.face_ids[2] << ","
               << p.face_ids[3] << "," << p.face_ids[4] << "," << p.face_ids[5] << "\n";
        }
        pf.close();

        std::ofstream lf("leaf_cells_lgr.csv");
        lf << "leaf_id,parent_id,lix,liy,liz,"
           << "cx,cy,cz,vol,"
           << "bbox_min_x,bbox_min_y,bbox_min_z,"
           << "bbox_max_x,bbox_max_y,bbox_max_z,";
        for (int n = 0; n < 8; ++n) lf << "c" << n << "_x,c" << n << "_y,c" << n << "_z,";
        lf << "face0,face1,face2,face3,face4,face5\n";
        for (const auto& c : leaves) {
            lf << c.leaf_id << "," << c.parent_id << ","
               << c.lix << "," << c.liy << "," << c.liz << ","
               << c.center.x << "," << c.center.y << "," << c.center.z << ","
               << c.vol << ","
               << c.bbox_min.x << "," << c.bbox_min.y << "," << c.bbox_min.z << ","
               << c.bbox_max.x << "," << c.bbox_max.y << "," << c.bbox_max.z << ",";
            for (int n = 0; n < 8; ++n) lf << c.corners[n].x << "," << c.corners[n].y << "," << c.corners[n].z << ",";
            lf << c.face_ids[0] << "," << c.face_ids[1] << "," << c.face_ids[2] << ","
               << c.face_ids[3] << "," << c.face_ids[4] << "," << c.face_ids[5] << "\n";
        }
        lf.close();

        std::ofstream ff("mm_faces_lgr.csv");
        ff << "face_id,owner,neighbor,local_owner_face,local_neighbor_face,"
           << "cx,cy,cz,nx,ny,nz,area,"
           << "bbox_min_x,bbox_min_y,bbox_min_z,"
           << "bbox_max_x,bbox_max_y,bbox_max_z,"
           << "v0_x,v0_y,v0_z,v1_x,v1_y,v1_z,v2_x,v2_y,v2_z,v3_x,v3_y,v3_z\n";
        for (const auto& f : mm_faces) {
            ff << f.id << "," << f.owner << "," << f.neighbor << ","
               << f.local_owner_face << "," << f.local_neighbor_face << ","
               << f.center.x << "," << f.center.y << "," << f.center.z << ","
               << f.normal.x << "," << f.normal.y << "," << f.normal.z << ","
               << f.area << ","
               << f.bbox_min.x << "," << f.bbox_min.y << "," << f.bbox_min.z << ","
               << f.bbox_max.x << "," << f.bbox_max.y << "," << f.bbox_max.z << ",";
            for (int i = 0; i < 4; ++i) {
                ff << f.vertices[i].x << "," << f.vertices[i].y << "," << f.vertices[i].z;
                if (i < 3) ff << ",";
            }
            ff << "\n";
        }
        ff.close();

        std::ofstream fracFile("fracture_geometry_lgr.csv");
        fracFile << "id,x0,y0,z0,x1,y1,z1,x2,y2,z2,x3,y3,z3\n";
        for (const auto& f : fractures) {
            fracFile << f.id;
            for (int i = 0; i < 4; ++i) {
                fracFile << "," << f.vertices[i].x << "," << f.vertices[i].y << "," << f.vertices[i].z;
            }
            fracFile << "\n";
        }
        fracFile.close();

        std::cout << "Geometry exported: grid_info_lgr.csv, parent_cells_lgr.csv, leaf_cells_lgr.csv, mm_faces_lgr.csv, fracture_geometry_lgr.csv" << std::endl;
    }

    void exportWells() {
        std::ofstream wellFile("well_info_lgr.csv");
        wellFile << "well_id,node_idx,type,x,y,z,WI,P_bhp\n";
        for (size_t i = 0; i < wells.size(); ++i) {
            int u = wells[i].target_node_idx;
            double x, y, z;
            std::string type;
            if (u < n_leaf) {
                x = leaves[u].center.x; y = leaves[u].center.y; z = leaves[u].center.z;
                type = "Leaf";
            } else {
                int seg_idx = u - n_leaf;
                x = segments[seg_idx].center.x;
                y = segments[seg_idx].center.y;
                z = segments[seg_idx].center.z;
                type = "FractureSegment";
            }
            wellFile << i << "," << u << "," << type << ","
                     << x << "," << y << "," << z << ","
                     << wells[i].WI << "," << wells[i].P_bhp << "\n";
        }
        wellFile.close();
        std::cout << "Wells exported: well_info_lgr.csv" << std::endl;
    }

    void exportSegmentDebugGeometry() {
        std::ofstream segf("segments_lgr.csv");
        segf << "seg_id,frac_id,leaf_id,parent_id,area,cx,cy,cz,nx,ny,nz,aperture,perm,Tmf,poly_n\n";
        for (const auto& s : segments) {
            segf << s.id << "," << s.frac_id << "," << s.matrix_leaf_id << "," << s.parent_id << ","
                 << s.area << "," << s.center.x << "," << s.center.y << "," << s.center.z << ","
                 << s.normal.x << "," << s.normal.y << "," << s.normal.z << ","
                 << s.aperture << "," << s.perm << "," << s.T_mf << "," << s.poly.size() << "\n";
        }
        segf.close();

        std::ofstream polyf("segment_poly_lgr.csv");
        polyf << "seg_id,vid,x,y,z\n";
        for (const auto& s : segments) {
            for (size_t i = 0; i < s.poly.size(); ++i) {
                polyf << s.id << "," << i << ","
                      << s.poly[i].x << "," << s.poly[i].y << "," << s.poly[i].z << "\n";
            }
        }
        polyf.close();

        std::ofstream trf("segment_subface_trace_lgr.csv");
        trf << "seg_id,leaf_id,subface_id,trace_len,center_dist\n";
        for (const auto& s : segments) {
            for (const auto& kv : s.subface_trace_len) {
                int sfid = kv.first;
                double len = kv.second;
                double dist = 0.0;
                auto it = s.subface_center_dist.find(sfid);
                if (it != s.subface_center_dist.end()) dist = it->second;
                trf << s.id << "," << s.matrix_leaf_id << "," << sfid << ","
                    << len << "," << dist << "\n";
            }
        }
        trf.close();

        std::cout << "Segment debug geometry exported: segments_lgr.csv, segment_poly_lgr.csv, segment_subface_trace_lgr.csv" << std::endl;
    }

    void run(double total_days) {
        std::ofstream file("output_sim_lgr.csv");
        file << "Time,CumOil,CumWater,CumGas,AvgPressure,DT,nLeaf,nSeg,Qo,Qw,Qg\n";
        double t = 0.0;
        const double dt0 = 1e-5;
        const double dt_min = 1e-8;
        const double dt_max = 5.0;
        int target_iter = 6;
        double dt_try = dt0;
        double tot_o = 0.0, tot_w = 0.0, tot_g = 0.0;
        int step = 0;

        while (t < total_days - 1e-12) {
            step++;
            dt_try = std::min(dt_try, total_days - t);
            double so = 0.0, sw = 0.0, sg = 0.0;
            bool ok = false;
            int actual_iter = 0;

            while (!ok) {
                if (dt_try < dt_min) {
                    std::cerr << "dt too small, abort." << std::endl;
                    return;
                }
                std::cout << "Step " << step << " t=" << t << " dt=" << dt_try << " ... " << std::flush;
                ok = solveStep(dt_try, so, sw, sg, actual_iter);
                if (!ok) {
                    std::cout << "fail -> dt_try*=0.25" << std::endl;
                    dt_try *= 0.25;
                }
            }
            std::cout << "ok" << std::endl;

            t += dt_try;
            states_prev = states;
            tot_o += so; tot_w += sw; tot_g += sg;
            double avgP = 0.0;
            for (int i = 0; i < n_leaf; ++i) avgP += states[i].P;
            avgP /= std::max(1, n_leaf);

            double qo = so / std::max(dt_try, 1e-30);
            double qw = sw / std::max(dt_try, 1e-30);
            double qg = sg / std::max(dt_try, 1e-30);

            file << t << "," << tot_o << "," << tot_w << "," << tot_g << "," << avgP << "," << dt_try
                 << "," << n_leaf << "," << n_seg << "," << qo << "," << qw << "," << qg << "\n";
            file.flush();

            double fac = std::pow((double)target_iter / (double)std::max(1, actual_iter), 0.5);
            fac = std::max(0.5, std::min(1.5, fac));
            dt_try = std::min(dt_max, std::max(dt0, dt_try) * fac);
        }

        file.close();

        std::ofstream field("final_field_lgr.csv");
        field << "leaf_id,parent_id,x,y,z,P,Sw,Sg\n";
        for (int i=0;i<n_leaf;++i) {
            field << i << "," << leaves[i].parent_id << ","
                  << leaves[i].center.x << "," << leaves[i].center.y << "," << leaves[i].center.z << ","
                  << states[i].P << "," << states[i].Sw << "," << states[i].Sg << "\n";
        }
        field.close();
    }

    bool preprocess() {
        if (!buildParentGridFromCornerPointCSV(coord_file_path, zcorn_file_path)) return false;
        generateFractures(
            natural_frac_count,
            natural_min_length,
            natural_max_length,
            natural_max_dip,
            natural_min_strike,
            natural_max_strike,
            natural_aperture,
            natural_perm,
            use_region_fractures ? region_x_min : 0.0,
            use_region_fractures ? region_x_max : -1.0,
            use_region_fractures ? region_y_min : 0.0,
            use_region_fractures ? region_y_max : -1.0,
            use_region_fractures ? region_z_min : 0.0,
            use_region_fractures ? region_z_max : -1.0
        );
        generateHydraulicFractures(
            hydraulic_frac_count,
            hydraulic_well_length,
            hydraulic_half_length,
            hydraulic_height,
            hydraulic_aperture,
            hydraulic_perm,
            hydraulic_center_x,
            hydraulic_center_y,
            hydraulic_center_z
        );
        markRefinement();
        buildLeafGrid();
        buildParentFaceCoverage();
        buildLeafInterfaceFaces();

        std::vector<Connection> mm, mf, ff;
        buildMMConnections(mm);
        buildSegmentsAndMF(mf);
        buildFFConnections(ff);
        buildAllConnections(mm, mf, ff);

        setupWells();
        initState();
        buildJacobianPattern();
        checkGeometryConsistency();
        return true;
    }
};

#ifndef EDFM_CORE_CORNER_LGR_PYBIND
int main() {
    SimulatorLGR sim;
    sim.setCornerPointFiles("COORD.csv", "ZCORN.csv");
    std::cout << "Preprocessing (corner-point CSV parent grid + LGR + EDFM)..." << std::endl;
    if (!sim.preprocess()) {
        std::cerr << "Preprocess failed." << std::endl;
        return 1;
    }

    std::cout << "Exporting geometry/debug files..." << std::endl;
    sim.exportStaticGeometry();
    sim.exportSegmentDebugGeometry();
    sim.exportWells();

    std::cout << "Running simulation..." << std::endl;
    sim.run(100.0);

    std::cout << "Done. Outputs saved." << std::endl;
    return 0;
}
#endif

PYBIND11_MODULE(edfm_core_corner_lgr, m) {
    py::class_<SimulationResult>(m, "SimulationResult")
        .def_readwrite("pressure_field", &SimulationResult::pressure_field)
        .def_readwrite("temperature_field", &SimulationResult::temperature_field)
        .def_readwrite("stress_field", &SimulationResult::stress_field)
        .def_readwrite("fracture_vertices", &SimulationResult::fracture_vertices)
        .def_readwrite("fracture_cells", &SimulationResult::fracture_cells);

    py::class_<SimulatorLGR>(m, "EDFMSimulator")
        .def(py::init<>())
        .def("setCornerPointFiles", &SimulatorLGR::setCornerPointFiles)
        .def("setFractureParameters", &SimulatorLGR::setFractureParameters)
        .def("setRegionFractureParameters", &SimulatorLGR::setRegionFractureParameters)
        .def("setHydraulicFractureParameters", &SimulatorLGR::setHydraulicFractureParameters,
             py::arg("total_fracs"),
             py::arg("well_length"),
             py::arg("hf_len"),
             py::arg("hf_height"),
             py::arg("aperture_val"),
             py::arg("perm_val"),
             py::arg("x_center") = -1.0,
             py::arg("y_center") = -1.0,
             py::arg("z_center") = -1.0)
        .def("setWellParameters", &SimulatorLGR::setWellParameters)
        .def("setInitialStateParameters", &SimulatorLGR::setInitialStateParameters)
        .def("setSimulationParameters", &SimulatorLGR::setSimulationParameters)
        .def("setLGRParameters", &SimulatorLGR::setLGRParameters)
        .def("runSimulation", &SimulatorLGR::runSimulation)
        .def("getPressureData", &SimulatorLGR::getPressureData)
        .def("getFractureVertices", &SimulatorLGR::getFractureVertices)
        .def("getCellGeometryWithPressure", &SimulatorLGR::getCellGeometryWithPressure)
        .def("getLGRGridGeometry", &SimulatorLGR::getLGRGridGeometry)
        .def("getParentGridGeometry", &SimulatorLGR::getParentGridGeometry)
        .def("getRefinedGridGeometry", &SimulatorLGR::getRefinedGridGeometry);
}

