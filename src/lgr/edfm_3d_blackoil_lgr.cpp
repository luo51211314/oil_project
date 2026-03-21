// =============================================================================
// File: edfm_3d_water_oil_lgr.cpp
// Desc: 3D (W/O/G) fully-implicit EDFM + Static one-level LGR around fractures
// Build: g++ -O2 -std=c++17 edfm_3d_water_oil_lgr.cpp -o edfm_lgr -I /path/to/eigen
// =============================================================================
//
// "First runnable" LGR implementation
// -----------------------------------
// - Parent structured grid is built first; then parents are marked refined if:
//     (a) fracture intersects parent cell (clip area > eps_area), OR
//     (b) volume-averaged distance d_bar < d_threshold.
// - Each refined parent is uniformly subdivided into Nrx*Nry*Nrz leaf cells.
// - Only leaf cells participate in the matrix unknown vector.
// - EDFM fracture segments are generated at leaf scale (clip quad ∩ leaf AABB).
// - Matrix-Matrix (MM) connections are rebuilt, including coarse-fine interfaces.
// - Matrix-Fracture (MF) and Fracture-Fracture (FF) connections are built with
//   leaf bucketing + neighbor search to avoid O(m^2).
//
// Important practical defaults
// ----------------------------
// - Your original grid (50x20x5) plus default LGR (10x10x5) can explode in
//   unknown count and make SparseLU / numerical Jacobian infeasible.
// - Therefore this file sets a SAFE default grid and refinement ratio so the
//   code can run end-to-end on a laptop.
// - You can switch back to your original Nx/Ny/Nz and 10x10x5 after verifying
//   correctness, and then you will very likely need: (i) analytic Jacobian,
//   (ii) iterative linear solver + preconditioner, and (iii) stricter pruning.
// =============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <random>
#include <tuple>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <cstdint>

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/SparseLU>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/AutoDiff>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using namespace std;
using namespace Eigen;

// =============================================================================
// 1. Constants & basic geometry
// =============================================================================

static constexpr double EPS = 1e-12;
static constexpr double PI  = 3.14159265358979323846;

struct Point3 {
    double x{}, y{}, z{};
    Point3 operator+(const Point3& o) const { return {x+o.x, y+o.y, z+o.z}; }
    Point3 operator-(const Point3& o) const { return {x-o.x, y-o.y, z-o.z}; }
    Point3 operator*(double s) const { return {x*s, y*s, z*s}; }
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
    // minimum Euclidean distance between two AABBs (0 if intersect)
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

// =============================================================================
// 2. Fracture & grid entities
// =============================================================================

struct Fracture {
    int id{};
    Point3 vertices[4];
    double aperture{0.001};
    double perm{10000.0};
};

struct ParentCell {
    int parent_id{};
    int ix{}, iy{}, iz{};
    Point3 center{};
    double dx{}, dy{}, dz{};
    double vol{};
    double phi{};
    double K[3]{};
    double depth{};
    AABB box{};
    bool refined{false};
    int leaf_base{-1};
    uint16_t Nrx{1}, Nry{1}, Nrz{1};
};

struct LeafCell {
    int leaf_id{};
    int parent_id{};
    uint16_t lix{0}, liy{0}, liz{0};
    Point3 center{};
    double dx{}, dy{}, dz{};
    double vol{};
    double phi{};
    double K[3]{};
    double depth{};
    AABB box{};
};

struct Segment {
    int id{};
    int frac_id{};
    int matrix_leaf_id{}; // IMPORTANT: leaf id
    int parent_id{-1};    // optional debug
    double area{0.0};
    Point3 center{};
    Point3 normal{};
    double aperture{0.001};
    double perm{10000.0};
    double T_mf{0.0};
};

struct Connection {
    int u{}, v{}; // global node index (leaf or segment)
    double T{0.0};
    int type{0};  // 0=MM,1=MF,2=FF
};

// =============================================================================
// 3. Polygon clipping: quad ∩ AABB (Sutherland-Hodgman)
// =============================================================================

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
    for (const auto& p : poly) c = c + p;
    return c * (1.0 / (double)poly.size());
}

static std::vector<Point3> clipFractureBox(const Fracture& frac, const AABB& box) {
    std::vector<Point3> poly;
    poly.reserve(4);
    for (int i = 0; i < 4; ++i) poly.push_back(frac.vertices[i]);
    const double x_min = box.mn.x, x_max = box.mx.x;
    const double y_min = box.mn.y, y_max = box.mx.y;
    const double z_min = box.mn.z, z_max = box.mx.z;
    // 6 planes with inward normals
    poly = clipPolyPlane(poly, { 1, 0, 0}, -x_min);
    poly = clipPolyPlane(poly, {-1, 0, 0},  x_max);
    poly = clipPolyPlane(poly, { 0, 1, 0}, -y_min);
    poly = clipPolyPlane(poly, { 0,-1, 0},  y_max);
    poly = clipPolyPlane(poly, { 0, 0, 1}, -z_min);
    poly = clipPolyPlane(poly, { 0, 0,-1},  z_max);
    return poly;
}

// =============================================================================
// 4. Distance: point-to-triangle (Ericson) and quad distance
// =============================================================================

static inline double distPointSegment(const Point3& p, const Point3& a, const Point3& b) {
    Point3 ab = b - a;
    double t = (p - a).dot(ab) / std::max(EPS, ab.dot(ab));
    t = clampd(t, 0.0, 1.0);
    Point3 q = a + ab * t;
    return (p - q).norm();
}

static double distPointTriangle(const Point3& p, const Point3& a, const Point3& b, const Point3& c) {
    // Real-Time Collision Detection (Christer Ericson)
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

    // inside face region
    Point3 n = ab.cross(ac);
    double nlen = std::max(EPS, n.norm());
    n = n * (1.0 / nlen);
    double dist = std::abs((p - a).dot(n));
    return dist;
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

static double dbarCellQuad(const AABB& cell, const Fracture& f, int nsx, int nsy, int nsz) {
    // uniform sampling of cell volume
    double sx = (cell.mx.x - cell.mn.x) / nsx;
    double sy = (cell.mx.y - cell.mn.y) / nsy;
    double sz = (cell.mx.z - cell.mn.z) / nsz;
    double sum = 0.0;
    int cnt = 0;
    for (int k = 0; k < nsz; ++k) {
        for (int j = 0; j < nsy; ++j) {
            for (int i = 0; i < nsx; ++i) {
                Point3 p{cell.mn.x + (i + 0.5)*sx,
                         cell.mn.y + (j + 0.5)*sy,
                         cell.mn.z + (k + 0.5)*sz};
                sum += distPointQuad(p, f);
                cnt++;
            }
        }
    }
    return (cnt > 0) ? (sum / cnt) : 1e30;
}

// =============================================================================
// 5. Fluid props (same style as your original code)
// =============================================================================

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
    double Swi = 0.2;
    double Sor = 0.2;
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
    T P{200.0};
    T Sw{0.2};
    T Sg{0.05};
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

// =============================================================================
// Python Result Structure
// =============================================================================

struct SimulationResult {
    std::vector<std::tuple<double,double,double,double>> pressure_field;
    std::vector<std::tuple<double,double,double,double>> temperature_field;
    std::vector<std::tuple<double,double,double,double>> stress_field;
    std::vector<std::tuple<double,double,double,double>> fracture_vertices;
    std::vector<std::tuple<int,int,int,int>> fracture_cells;
};

// =============================================================================
// 6. Simulator with LGR
// =============================================================================

class SimulatorLGR {
public:
    // Domain
    int Nx{100}, Ny{50}, Nz{10};           // SAFE default for "runnable" LGR
    double Lx{1000}, Ly{500}, Lz{100};
    double dx{}, dy{}, dz{};

    // LGR config
    bool enable_lgr{true};
    double d_threshold{30.0};
    uint16_t lgr_Nrx{2}, lgr_Nry{2}, lgr_Nrz{2}; // SAFE default; switch to 10/10/5 later
    int dbar_nsx{4}, dbar_nsy{4}, dbar_nsz{4};
    double eps_area_factor{1e-8}; // eps_area = eps_area_factor * min_face_area
    double d_avg_factor{0.5};     // d_avg = d_avg_factor * min(dx,dy,dz)

    // Geometry storage
    std::vector<ParentCell> parents;
    std::vector<LeafCell> leaves;
    std::vector<Fracture> fractures;
    std::vector<Segment> segments;

    // Maps / helpers
    std::vector<std::vector<int>> parent_face_leaves[6]; // [face][pid] -> list of leaf ids
    std::vector<std::vector<int>> leaf_neighbors;        // MM neighbors
    std::vector<std::vector<int>> leaf_to_segs;          // bucket

    // Graph
    std::vector<Connection> connections;
    struct Neighbor { int v; double T; int type; };
    std::vector<std::vector<Neighbor>> adj; // for residual

    // unknown counts
    int n_leaf{0};
    int n_seg{0};
    int n_total{0};

    // states
    std::vector<State> states, states_prev;

    // Well
    struct Well { int target_node_idx; double WI; double P_bhp; };
    std::vector<Well> wells;
    std::unordered_map<int,int> well_map;

    SparseMatrix<double> J;
    struct CellOffsets { int diag[3][3]; };
    struct ConnOffsets { int off_uv[3][3]; int off_vu[3][3]; };
    std::vector<CellOffsets> cell_J_idx;
    std::vector<ConnOffsets> conn_J_idx;

    SimulatorLGR() {
        dx = Lx / Nx; dy = Ly / Ny; dz = Lz / Nz;
    }

    // ---------------------------------------------------------------------
    // 6.1 Build parent grid
    // ---------------------------------------------------------------------
    void buildParentGrid() {
        dx = Lx / Nx; dy = Ly / Ny; dz = Lz / Nz;
        int n_parent = Nx*Ny*Nz;
        parents.resize(n_parent);
        for (int k = 0; k < Nz; ++k) {
            for (int j = 0; j < Ny; ++j) {
                for (int i = 0; i < Nx; ++i) {
                    int pid = k*Nx*Ny + j*Nx + i;
                    ParentCell pc;
                    pc.parent_id = pid;
                    pc.ix=i; pc.iy=j; pc.iz=k;
                    pc.dx=dx; pc.dy=dy; pc.dz=dz;
                    pc.center = {(i+0.5)*dx, (j+0.5)*dy, (k+0.5)*dz};
                    pc.vol = dx*dy*dz;
                    pc.phi = porosity_;
                    pc.K[0]=perm_x_; pc.K[1]=perm_y_; pc.K[2]=perm_z_;
                    pc.depth = pc.center.z;
                    pc.box = makeAABBFromCenterSize(pc.center, dx, dy, dz);
                    pc.refined = false;
                    pc.Nrx = lgr_Nrx; pc.Nry = lgr_Nry; pc.Nrz = lgr_Nrz;
                    parents[pid] = pc;
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // 6.2 Generate fractures (same as your original implementation)
    // ---------------------------------------------------------------------
    void generateFractures(int total_fracs = 100, 
                           double min_L = 30.0, double max_L = 80.0, 
                           double max_dip = PI/3.0, 
                           double min_strike = 0.0, double max_strike = PI,
                           double aperture_val = 0.001, double perm_val = 10000.0,
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

        for(int i=0; i<total_fracs; ++i) { 
            Fracture f; f.id = i; f.aperture = aperture_val; f.perm = perm_val;

            int tries = 0;
            while (true) {
                if (++tries > 200000) {
                    std::cerr << "Failed to place natural fracture " << i
                            << " fully inside domain. "
                            << "Consider reducing distL/distDip or enlarging domain.\n";
                    return;
                }

                Point3 center = {distX(rng), distY(rng), distZ(rng)};
                double len = distL(rng); double height = distL(rng) * 0.5;
                double strike = distAngle(rng); double dip = distDip(rng);
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

        double xc=Lx/2.0, yc=Ly/2.0, zc=Lz/2.0;
        double offsets[3] = {-100.1, 0.1, 100.1};
        double hf_len=200.0, hf_height=40.0;
        for(int k=0; k<3; ++k) {
            Fracture f; f.id = 100 + k; f.aperture = aperture_val; f.perm = perm_val;
            double x_curr = xc + offsets[k];
            f.vertices[0] = {x_curr, yc - hf_len/2, zc - hf_height/2};
            f.vertices[1] = {x_curr, yc + hf_len/2, zc - hf_height/2};
            f.vertices[2] = {x_curr, yc + hf_len/2, zc + hf_height/2};
            f.vertices[3] = {x_curr, yc - hf_len/2, zc + hf_height/2};
            fractures.push_back(f);
        }
    }

    void generateHydraulicFractures(int total_fracs = 0,
                                    double strike_val = PI / 2.0, double dip_val = PI / 2.0,
                                    double length_val = 200.0, double height_val = 40.0,
                                    double aperture_val = 0.001, double perm_val = 10000.0,
                                    double range_x_min = 0.0, double range_x_max = -1.0,
                                    double range_y_min = 0.0, double range_y_max = -1.0,
                                    double range_z_min = 0.0, double range_z_max = -1.0,
                                    int start_id = 200) {
        
        double use_max_x = (range_x_max < 0) ? Lx : range_x_max;
        double use_max_y = (range_y_max < 0) ? Ly : range_y_max;
        double use_max_z = (range_z_max < 0) ? Lz : range_z_max;

        std::mt19937 rng(42); 
        std::uniform_real_distribution<double> distX(range_x_min, use_max_x);
        std::uniform_real_distribution<double> distY(range_y_min, use_max_y);
        std::uniform_real_distribution<double> distZ(range_z_min, use_max_z);

        Point3 u = {cos(strike_val), sin(strike_val), 0};
        Point3 n_horiz = {-sin(strike_val), cos(strike_val), 0};
        Point3 v = {n_horiz.x * cos(dip_val), n_horiz.y * cos(dip_val), -sin(dip_val)};

        auto inBox = [&](const Point3& p) -> bool {
            return (p.x >= 0.0 && p.x <= Lx &&
                    p.y >= 0.0 && p.y <= Ly &&
                    p.z >= 0.0 && p.z <= Lz);
        };
        auto fracVerticesInBox = [&](const Fracture& f) -> bool {
            return inBox(f.vertices[0]) && inBox(f.vertices[1]) &&
                inBox(f.vertices[2]) && inBox(f.vertices[3]);
        };

        for(int i=0; i<total_fracs; ++i) {
            Fracture f; f.id = start_id + i; f.aperture = aperture_val; f.perm = perm_val;
            
            int tries = 0;
            while(true) {
                if (++tries > 200000) {
                    std::cerr << "Failed to place hydraulic fracture " << i << ".\n";
                    return;
                }

                Point3 center = {distX(rng), distY(rng), distZ(rng)};
                
                f.vertices[0] = center - u*(length_val/2) - v*(height_val/2);
                f.vertices[1] = center + u*(length_val/2) - v*(height_val/2);
                f.vertices[2] = center + u*(length_val/2) + v*(height_val/2);
                f.vertices[3] = center - u*(length_val/2) + v*(height_val/2);

                if (fracVerticesInBox(f)) {
                    fractures.push_back(f);
                    break;
                }
            }
        }
    }


    // ---------------------------------------------------------------------
    // 6.3 Mark refinement
    // ---------------------------------------------------------------------
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
        // reset
        for (auto& p : parents) p.refined = false;

        // For each fracture, scan candidate parent range based on expanded AABB
        for (const auto& f : fractures) {
            AABB fb = fractureAABB(f);
            AABB fb_exp = expandAABB(fb, d_threshold);

            int i0 = std::max(0, (int)std::floor(fb_exp.mn.x / dx));
            int i1 = std::min(Nx-1, (int)std::floor(fb_exp.mx.x / dx));
            int j0 = std::max(0, (int)std::floor(fb_exp.mn.y / dy));
            int j1 = std::min(Ny-1, (int)std::floor(fb_exp.mx.y / dy));
            int k0 = std::max(0, (int)std::floor(fb_exp.mn.z / dz));
            int k1 = std::min(Nz-1, (int)std::floor(fb_exp.mx.z / dz));

            for (int k=k0; k<=k1; ++k) {
                for (int j=j0; j<=j1; ++j) {
                    for (int i=i0; i<=i1; ++i) {
                        int pid = k*Nx*Ny + j*Nx + i;
                        ParentCell& pc = parents[pid];
                        if (pc.refined) continue; // already refined by other fracture

                        // L0.5: AABB distance pruning
                        if (distAABB_AABB(pc.box, fb) > d_threshold) continue;

                        // L1: intersection via clipping
                        double epsA = epsAreaForBox(pc.dx, pc.dy, pc.dz);
                        auto poly = clipFractureBox(f, pc.box);
                        double area = polygonArea(poly);
                        if (area > epsA) {
                            pc.refined = true;
                            continue;
                        }

                        // L2: d_bar sampling
                        double dbar = dbarCellQuad(pc.box, f, dbar_nsx, dbar_nsy, dbar_nsz);
                        if (dbar < d_threshold) {
                            pc.refined = true;
                        }
                    }
                }
            }
        }

        int cnt = 0;
        for (const auto& p : parents) if (p.refined) cnt++;
        std::cout << "Refinement marked: refined parents = " << cnt << " / " << (int)parents.size() << std::endl;
    }

    // ---------------------------------------------------------------------
    // 6.4 Build leaf grid + mapping
    // ---------------------------------------------------------------------
    void buildLeafGrid() {
        // leaf count
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
                lc.lix = lc.liy = lc.liz = 0;
                lc.dx = p.dx; lc.dy = p.dy; lc.dz = p.dz;
                lc.center = p.center;
                lc.vol = p.vol;
                lc.phi = p.phi;
                lc.K[0]=p.K[0]; lc.K[1]=p.K[1]; lc.K[2]=p.K[2];
                lc.depth = p.depth;
                lc.box = p.box;
                leaves[lid] = lc;
            } else {
                double dxs = p.dx / p.Nrx;
                double dys = p.dy / p.Nry;
                double dzs = p.dz / p.Nrz;
                for (uint16_t lk=0; lk<p.Nrz; ++lk) {
                    for (uint16_t lj=0; lj<p.Nry; ++lj) {
                        for (uint16_t li=0; li<p.Nrx; ++li) {
                            int lid = p.leaf_base + ((int)lk*(int)p.Nry + (int)lj)*(int)p.Nrx + (int)li;
                            Point3 c{
                                p.box.mn.x + (li + 0.5)*dxs,
                                p.box.mn.y + (lj + 0.5)*dys,
                                p.box.mn.z + (lk + 0.5)*dzs
                            };
                            LeafCell lc;
                            lc.leaf_id = lid;
                            lc.parent_id = p.parent_id;
                            lc.lix = li; lc.liy = lj; lc.liz = lk;
                            lc.dx = dxs; lc.dy = dys; lc.dz = dzs;
                            lc.center = c;
                            lc.vol = dxs*dys*dzs;
                            lc.phi = p.phi;
                            lc.K[0]=p.K[0]; lc.K[1]=p.K[1]; lc.K[2]=p.K[2];
                            lc.depth = c.z;
                            lc.box = makeAABBFromCenterSize(c, dxs, dys, dzs);
                            leaves[lid] = lc;
                        }
                    }
                }
            }
        }

        std::cout << "Leaf grid built: n_leaf = " << n_leaf << std::endl;
    }

    // ---------------------------------------------------------------------
    // 6.5 Build parent face leaf cover (nonconforming interface table)
    // ---------------------------------------------------------------------
    void buildParentFaceLeaves() {
        int n_parent = (int)parents.size();
        for (int f=0; f<6; ++f) {
            parent_face_leaves[f].clear();
            parent_face_leaves[f].resize(n_parent);
        }
        for (const auto& p : parents) {
            int pid = p.parent_id;
            if (!p.refined) {
                for (int f=0; f<6; ++f) parent_face_leaves[f][pid] = {p.leaf_base};
                continue;
            }
            // ordering: lk outer, lj inner (consistent across neighbors)
            // -X (li=0) / +X (li=Nrx-1)
            {
                std::vector<int> negx, posx;
                negx.reserve((int)p.Nry*(int)p.Nrz);
                posx.reserve((int)p.Nry*(int)p.Nrz);
                for (uint16_t lk=0; lk<p.Nrz; ++lk) {
                    for (uint16_t lj=0; lj<p.Nry; ++lj) {
                        int lid0 = p.leaf_base + ((int)lk*(int)p.Nry + (int)lj)*(int)p.Nrx + 0;
                        int lid1 = p.leaf_base + ((int)lk*(int)p.Nry + (int)lj)*(int)p.Nrx + ((int)p.Nrx - 1);
                        negx.push_back(lid0);
                        posx.push_back(lid1);
                    }
                }
                parent_face_leaves[0][pid] = std::move(negx);
                parent_face_leaves[1][pid] = std::move(posx);
            }
            // -Y / +Y
            {
                std::vector<int> negy, posy;
                negy.reserve((int)p.Nrx*(int)p.Nrz);
                posy.reserve((int)p.Nrx*(int)p.Nrz);
                for (uint16_t lk=0; lk<p.Nrz; ++lk) {
                    for (uint16_t li=0; li<p.Nrx; ++li) {
                        int lid0 = p.leaf_base + ((int)lk*(int)p.Nry + 0)*(int)p.Nrx + (int)li;
                        int lid1 = p.leaf_base + ((int)lk*(int)p.Nry + ((int)p.Nry - 1))*(int)p.Nrx + (int)li;
                        negy.push_back(lid0);
                        posy.push_back(lid1);
                    }
                }
                parent_face_leaves[2][pid] = std::move(negy);
                parent_face_leaves[3][pid] = std::move(posy);
            }
            // -Z / +Z
            {
                std::vector<int> negz, posz;
                negz.reserve((int)p.Nrx*(int)p.Nry);
                posz.reserve((int)p.Nrx*(int)p.Nry);
                for (uint16_t lj=0; lj<p.Nry; ++lj) {
                    for (uint16_t li=0; li<p.Nrx; ++li) {
                        int lid0 = p.leaf_base + (0*(int)p.Nry + (int)lj)*(int)p.Nrx + (int)li;
                        int lid1 = p.leaf_base + (((int)p.Nrz - 1)*(int)p.Nry + (int)lj)*(int)p.Nrx + (int)li;
                        negz.push_back(lid0);
                        posz.push_back(lid1);
                    }
                }
                parent_face_leaves[4][pid] = std::move(negz);
                parent_face_leaves[5][pid] = std::move(posz);
            }
        }
    }

    // ---------------------------------------------------------------------
    // 6.6 Build MM connections (leaf graph only) + leaf_neighbors
    // ---------------------------------------------------------------------
    static inline double halfResT(double area, double dL, double kL, double dR, double kR) {
        // T = A / (dL/kL + dR/kR)
        return area / (dL / std::max(EPS, kL) + dR / std::max(EPS, kR));
    }

    void buildMMConnections(std::vector<Connection>& mm_out) {
        mm_out.clear();
        mm_out.reserve(n_leaf * 4);
        leaf_neighbors.clear();
        leaf_neighbors.resize(n_leaf);

        auto add_mm = [&](int a, int b, double T) {
            if (a == b) return;
            int u = std::min(a,b);
            int v = std::max(a,b);
            mm_out.push_back({u,v,T,0});
            leaf_neighbors[u].push_back(v);
            leaf_neighbors[v].push_back(u);
        };

        // (A) intra refined parent: subgrid 6-neigh
        for (const auto& p : parents) {
            if (!p.refined) continue;
            int base = p.leaf_base;
            int Nrx = p.Nrx, Nry=p.Nry, Nrz=p.Nrz;
            for (int lk=0; lk<Nrz; ++lk) {
                for (int lj=0; lj<Nry; ++lj) {
                    for (int li=0; li<Nrx; ++li) {
                        int lid = base + (lk*Nry + lj)*Nrx + li;
                        const LeafCell& c = leaves[lid];
                        if (li + 1 < Nrx) {
                            int rid = lid + 1;
                            const LeafCell& r = leaves[rid];
                            double A = c.dy * c.dz;
                            double T = halfResT(A, c.dx*0.5, c.K[0], r.dx*0.5, r.K[0]);
                            add_mm(lid, rid, T);
                        }
                        if (lj + 1 < Nry) {
                            int uid = base + (lk*Nry + (lj+1))*Nrx + li;
                            const LeafCell& ucell = leaves[uid];
                            double A = c.dx * c.dz;
                            double T = halfResT(A, c.dy*0.5, c.K[1], ucell.dy*0.5, ucell.K[1]);
                            add_mm(lid, uid, T);
                        }
                        if (lk + 1 < Nrz) {
                            int wid = base + ((lk+1)*Nry + lj)*Nrx + li;
                            const LeafCell& w = leaves[wid];
                            double A = c.dx * c.dy;
                            double T = halfResT(A, c.dz*0.5, c.K[2], w.dz*0.5, w.K[2]);
                            add_mm(lid, wid, T);
                        }
                    }
                }
            }
        }

        // (B) cross parent faces: traverse in +x,+y,+z in parent grid
        auto parentAt = [&](int i,int j,int k)->int { return k*Nx*Ny + j*Nx + i; };
        for (int k=0; k<Nz; ++k) {
            for (int j=0; j<Ny; ++j) {
                for (int i=0; i<Nx; ++i) {
                    int A = parentAt(i,j,k);
                    const ParentCell& pA = parents[A];
                    // +X
                    if (i + 1 < Nx) {
                        int B = parentAt(i+1,j,k);
                        const ParentCell& pB = parents[B];
                        const auto& LA = parent_face_leaves[1][A]; // +X
                        const auto& LB = parent_face_leaves[0][B]; // -X
                        connectParentFaces_MM(LA, pA, LB, pB, /*dir=*/0, add_mm);
                    }
                    // +Y
                    if (j + 1 < Ny) {
                        int B = parentAt(i,j+1,k);
                        const ParentCell& pB = parents[B];
                        const auto& LA = parent_face_leaves[3][A]; // +Y
                        const auto& LB = parent_face_leaves[2][B]; // -Y
                        connectParentFaces_MM(LA, pA, LB, pB, /*dir=*/1, add_mm);
                    }
                    // +Z
                    if (k + 1 < Nz) {
                        int B = parentAt(i,j,k+1);
                        const ParentCell& pB = parents[B];
                        const auto& LA = parent_face_leaves[5][A]; // +Z
                        const auto& LB = parent_face_leaves[4][B]; // -Z
                        connectParentFaces_MM(LA, pA, LB, pB, /*dir=*/2, add_mm);
                    }
                }
            }
        }

        // dedup leaf_neighbors
        for (auto& nb : leaf_neighbors) {
            std::sort(nb.begin(), nb.end());
            nb.erase(std::unique(nb.begin(), nb.end()), nb.end());
        }
    }

    // dir: 0=x,1=y,2=z
    template<class AddMM>
    void connectParentFaces_MM(
        const std::vector<int>& LA, const ParentCell& pA,
        const std::vector<int>& LB, const ParentCell& pB,
        int dir,
        AddMM add_mm)
    {
        // coarse-coarse
        if (LA.size() == 1 && LB.size() == 1) {
            int a = LA[0], b = LB[0];
            const LeafCell& ca = leaves[a];
            const LeafCell& cb = leaves[b];
            double A = (dir==0)? (ca.dy*ca.dz) : (dir==1? (ca.dx*ca.dz) : (ca.dx*ca.dy));
            double dL = (dir==0)? (ca.dx*0.5) : (dir==1? (ca.dy*0.5) : (ca.dz*0.5));
            double dR = (dir==0)? (cb.dx*0.5) : (dir==1? (cb.dy*0.5) : (cb.dz*0.5));
            double kL = ca.K[dir];
            double kR = cb.K[dir];
            add_mm(a, b, halfResT(A, dL, kL, dR, kR));
            return;
        }

        // coarse-fine (A coarse)
        if (LA.size() == 1 && LB.size() > 1) {
            int a = LA[0];
            const LeafCell& ca = leaves[a];
            for (int b : LB) {
                const LeafCell& cb = leaves[b];
                double A = (dir==0)? (cb.dy*cb.dz) : (dir==1? (cb.dx*cb.dz) : (cb.dx*cb.dy));
                double dL = (dir==0)? (ca.dx*0.5) : (dir==1? (ca.dy*0.5) : (ca.dz*0.5));
                double dR = (dir==0)? (cb.dx*0.5) : (dir==1? (cb.dy*0.5) : (cb.dz*0.5));
                add_mm(a, b, halfResT(A, dL, ca.K[dir], dR, cb.K[dir]));
            }
            return;
        }

        // coarse-fine (B coarse)
        if (LA.size() > 1 && LB.size() == 1) {
            int b = LB[0];
            const LeafCell& cb = leaves[b];
            for (int a : LA) {
                const LeafCell& ca = leaves[a];
                double A = (dir==0)? (ca.dy*ca.dz) : (dir==1? (ca.dx*ca.dz) : (ca.dx*ca.dy));
                double dL = (dir==0)? (ca.dx*0.5) : (dir==1? (ca.dy*0.5) : (ca.dz*0.5));
                double dR = (dir==0)? (cb.dx*0.5) : (dir==1? (cb.dy*0.5) : (cb.dz*0.5));
                add_mm(a, b, halfResT(A, dL, ca.K[dir], dR, cb.K[dir]));
            }
            return;
        }

        // fine-fine: assume same subdivision & aligned, do 1-1 by list order
        if (LA.size() == LB.size()) {
            for (size_t idx=0; idx<LA.size(); ++idx) {
                int a = LA[idx], b = LB[idx];
                const LeafCell& ca = leaves[a];
                const LeafCell& cb = leaves[b];
                double A = (dir==0)? (ca.dy*ca.dz) : (dir==1? (ca.dx*ca.dz) : (ca.dx*ca.dy));
                double dL = (dir==0)? (ca.dx*0.5) : (dir==1? (ca.dy*0.5) : (ca.dz*0.5));
                double dR = (dir==0)? (cb.dx*0.5) : (dir==1? (cb.dy*0.5) : (cb.dz*0.5));
                add_mm(a, b, halfResT(A, dL, ca.K[dir], dR, cb.K[dir]));
            }
            return;
        }

        // fallback (different refinement ratio) - not supported in this first version
        // (do nothing)
    }

    // ---------------------------------------------------------------------
    // 6.7 Build segments (leaf-scale clipping) + MF bucket
    // ---------------------------------------------------------------------
    void buildSegmentsAndMF(std::vector<Connection>& mf_out) {
        segments.clear();
        mf_out.clear();
        leaf_to_segs.clear();
        leaf_to_segs.resize(n_leaf);

        int seg_id_counter = 0;
        for (const auto& frac : fractures) {
            AABB fb = fractureAABB(frac);
            // parent candidate range based on fracture AABB (no threshold needed; intersection only)
            int i0 = std::max(0, (int)std::floor(fb.mn.x / dx));
            int i1 = std::min(Nx-1, (int)std::floor(fb.mx.x / dx));
            int j0 = std::max(0, (int)std::floor(fb.mn.y / dy));
            int j1 = std::min(Ny-1, (int)std::floor(fb.mx.y / dy));
            int k0 = std::max(0, (int)std::floor(fb.mn.z / dz));
            int k1 = std::min(Nz-1, (int)std::floor(fb.mx.z / dz));

            // normal
            Point3 vec1 = frac.vertices[1] - frac.vertices[0];
            Point3 vec2 = frac.vertices[3] - frac.vertices[0];
            Point3 normal = vec1.cross(vec2);
            double nlen = std::max(EPS, normal.norm());
            normal = normal * (1.0 / nlen);

            for (int k=k0; k<=k1; ++k) {
                for (int j=j0; j<=j1; ++j) {
                    for (int i=i0; i<=i1; ++i) {
                        int pid = k*Nx*Ny + j*Nx + i;
                        const ParentCell& p = parents[pid];

                        if (!p.refined) {
                            double epsA = epsAreaForBox(p.dx, p.dy, p.dz);
                            auto poly = clipFractureBox(frac, p.box);
                            double area = polygonArea(poly);
                            if (area > epsA) {
                                Segment seg;
                                seg.id = seg_id_counter++;
                                seg.frac_id = frac.id;
                                seg.matrix_leaf_id = p.leaf_base;
                                seg.parent_id = pid;
                                seg.area = area;
                                seg.center = polygonCenter(poly);
                                seg.normal = normal;
                                seg.aperture = frac.aperture;
                                seg.perm = frac.perm;
                                // Tmf
                                const LeafCell& lc = leaves[seg.matrix_leaf_id];
                                double minh = std::min(lc.dx, std::min(lc.dy, lc.dz));
                                double d_avg = d_avg_factor * minh;
                                double Kn = seg.normal.x*seg.normal.x*lc.K[0] +
                                            seg.normal.y*seg.normal.y*lc.K[1] +
                                            seg.normal.z*seg.normal.z*lc.K[2];
                                seg.T_mf = area * (Kn / std::max(EPS, d_avg));
                                segments.push_back(seg);
                                leaf_to_segs[seg.matrix_leaf_id].push_back(seg.id);
                            }
                        } else {
                            // refined: visit only subcells overlapping fracture AABB
                            int Nrx=p.Nrx, Nry=p.Nry, Nrz=p.Nrz;
                            double dxs = p.dx/Nrx, dys=p.dy/Nry, dzs=p.dz/Nrz;

                            int li0 = (int)std::floor((fb.mn.x - p.box.mn.x) / dxs);
                            int li1 = (int)std::floor((fb.mx.x - p.box.mn.x) / dxs);
                            int lj0 = (int)std::floor((fb.mn.y - p.box.mn.y) / dys);
                            int lj1 = (int)std::floor((fb.mx.y - p.box.mn.y) / dys);
                            int lk0 = (int)std::floor((fb.mn.z - p.box.mn.z) / dzs);
                            int lk1 = (int)std::floor((fb.mx.z - p.box.mn.z) / dzs);
                            li0 = std::max(0, std::min(Nrx-1, li0));
                            li1 = std::max(0, std::min(Nrx-1, li1));
                            lj0 = std::max(0, std::min(Nry-1, lj0));
                            lj1 = std::max(0, std::min(Nry-1, lj1));
                            lk0 = std::max(0, std::min(Nrz-1, lk0));
                            lk1 = std::max(0, std::min(Nrz-1, lk1));

                            for (int lk=lk0; lk<=lk1; ++lk) {
                                for (int lj=lj0; lj<=lj1; ++lj) {
                                    for (int li=li0; li<=li1; ++li) {
                                        int lid = p.leaf_base + (lk*Nry + lj)*Nrx + li;
                                        const LeafCell& lc = leaves[lid];
                                        double epsA = epsAreaForBox(lc.dx, lc.dy, lc.dz);
                                        auto poly = clipFractureBox(frac, lc.box);
                                        double area = polygonArea(poly);
                                        if (area > epsA) {
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
                                            double minh = std::min(lc.dx, std::min(lc.dy, lc.dz));
                                            double d_avg = d_avg_factor * minh;
                                            double Kn = seg.normal.x*seg.normal.x*lc.K[0] +
                                                        seg.normal.y*seg.normal.y*lc.K[1] +
                                                        seg.normal.z*seg.normal.z*lc.K[2];
                                            seg.T_mf = area * (Kn / std::max(EPS, d_avg));
                                            segments.push_back(seg);
                                            leaf_to_segs[seg.matrix_leaf_id].push_back(seg.id);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        n_seg = (int)segments.size();
        std::cout << "Segments built: n_seg = " << n_seg << std::endl;

        // MF edges: leaf -> segnode
        mf_out.reserve(n_seg);
        for (int s=0; s<n_seg; ++s) {
            int u = segments[s].matrix_leaf_id;
            int v = n_leaf + s;
            mf_out.push_back({std::min(u,v), std::max(u,v), segments[s].T_mf, 1});
        }
    }

    // ---------------------------------------------------------------------
    // 6.8 Build FF connections (bucket + neighbor search)
    // ---------------------------------------------------------------------
    void buildFFConnections(std::vector<Connection>& ff_out) {
        ff_out.clear();
        // (A) intra-frac near neighbors within leaf neighborhood
        // distance threshold scaled with local cell size
        for (int s=0; s<n_seg; ++s) {
            const Segment& seg = segments[s];
            int leaf = seg.matrix_leaf_id;

            // candidate leaves: self + MM neighbors (1-hop)
            std::vector<int> cand_leaves;
            cand_leaves.reserve(1 + leaf_neighbors[leaf].size());
            cand_leaves.push_back(leaf);
            for (int nb : leaf_neighbors[leaf]) cand_leaves.push_back(nb);

            // build candidate segments list
            for (int cl : cand_leaves) {
                for (int t : leaf_to_segs[cl]) {
                    if (t <= s) continue; // avoid duplicates
                    if (segments[t].frac_id != seg.frac_id) continue;
                    double dist = (seg.center - segments[t].center).norm();
                    const LeafCell& lc = leaves[leaf];
                    double h = std::sqrt(lc.dx*lc.dx + lc.dy*lc.dy + lc.dz*lc.dz);
                    double r = 2.5 * h;
                    if (dist < std::max(1e-6, r)) {
                        double width = std::sqrt(std::max(EPS, std::min(seg.area, segments[t].area)));
                        double T = seg.perm * seg.aperture * width / std::max(1e-6, dist);
                        int u = n_leaf + s;
                        int v = n_leaf + t;
                        ff_out.push_back({std::min(u,v), std::max(u,v), T, 2});
                    }
                }
            }
        }

        // (B) inter-frac: within same leaf, connect different frac groups
        for (int leaf=0; leaf<n_leaf; ++leaf) {
            const auto& segs = leaf_to_segs[leaf];
            if (segs.size() < 2) continue;
            for (size_t i=0; i<segs.size(); ++i) {
                for (size_t j=i+1; j<segs.size(); ++j) {
                    int s1 = segs[i], s2 = segs[j];
                    if (segments[s1].frac_id == segments[s2].frac_id) continue;
                    int u = n_leaf + s1;
                    int v = n_leaf + s2;
                    double T1 = segments[s1].T_mf;
                    double T2 = segments[s2].T_mf;
                    double T_cross = (T1 * T2) / std::max(EPS, T1 + T2);
                    ff_out.push_back({std::min(u,v), std::max(u,v), T_cross, 2});
                }
            }
        }

        std::cout << "FF edges built (raw): " << ff_out.size() << std::endl;
    }

    // ---------------------------------------------------------------------
    // 6.9 Merge edges, dedup, build adjacency
    // ---------------------------------------------------------------------
    struct ConnKey {
        int type;
        int u;
        int v;
        bool operator==(const ConnKey& o) const { return type==o.type && u==o.u && v==o.v; }
    };
    struct ConnKeyHash {
        size_t operator()(const ConnKey& k) const {
            // 64-bit mix
            uint64_t x = (uint64_t)(k.type & 0xFF);
            x = (x << 28) ^ (uint64_t)k.u;
            x = (x << 28) ^ (uint64_t)k.v;
            // splitmix64-like
            x += 0x9e3779b97f4a7c15ULL;
            x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
            x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
            x = x ^ (x >> 31);
            return (size_t)x;
        }
    };

    void buildAllConnections(const std::vector<Connection>& mm,
                             const std::vector<Connection>& mf,
                             const std::vector<Connection>& ff)
    {
        n_total = n_leaf + n_seg;
        connections.clear();
        connections.reserve(mm.size() + mf.size() + ff.size());

        std::unordered_set<ConnKey, ConnKeyHash> seen;
        seen.reserve((mm.size() + mf.size() + ff.size()) * 2 + 16);

        auto push_unique = [&](const Connection& c) {
            int u = std::min(c.u, c.v);
            int v = std::max(c.u, c.v);
            ConnKey key{c.type, u, v};
            if (seen.insert(key).second) {
                connections.push_back({u, v, c.T, c.type});
            }
        };
        for (const auto& c : mm) push_unique(c);
        for (const auto& c : mf) push_unique(c);
        for (const auto& c : ff) push_unique(c);

        // adjacency
        adj.assign(n_total, {});
        for (const auto& c : connections) {
            adj[c.u].push_back({c.v, c.T, c.type});
            adj[c.v].push_back({c.u, c.T, c.type});
        }
        std::cout << "Connections built (unique): " << connections.size() << std::endl;
    }

    // ---------------------------------------------------------------------
    // 6.10 Wells (reuse your logic: producer on each HF near mid-z)
    // ---------------------------------------------------------------------
    void setupWells() {
        wells.clear();
        well_map.clear();
        std::vector<int> target_fracs = {100,101,102};
        for (int fid : target_fracs) {
            int best_s = -1;
            double best = 1e30;
            for (int s=0; s<n_seg; ++s) {
                if (segments[s].frac_id != fid) continue;
                double dzv = std::abs(segments[s].center.z - Lz*0.5);
                if (dzv < best) { best = dzv; best_s = s; }
            }
            if (best_s >= 0) {
                Well w;
                w.target_node_idx = n_leaf + best_s;
                int leaf_id = segments[best_s].matrix_leaf_id;
                const LeafCell& lc = leaves[leaf_id];
                double re = 0.14 * std::sqrt(lc.dz * lc.dz + lc.dy * lc.dy);
                double rw = well_radius_;
                double k_frac = segments[best_s].perm;
                w.WI = 2.0 * PI * k_frac * lc.dz / std::log(re / rw);
                w.P_bhp = well_pressure_;
                int idx = (int)wells.size();
                wells.push_back(w);
                well_map[w.target_node_idx] = idx;
                std::cout << "  Well " << idx << " on frac " << fid << ": node_idx=" << w.target_node_idx 
                          << ", WI=" << w.WI << ", P_bhp=" << w.P_bhp << " bar" << std::endl;
            } else {
                std::cout << "  WARNING: No segment found for frac " << fid << std::endl;
            }
        }
        std::cout << "Setup wells = " << wells.size() << std::endl;
    }

    // ---------------------------------------------------------------------
    // 6.11 States init
    // ---------------------------------------------------------------------
    void initState() {
        states.assign(n_total, {});
        states_prev = states;
        for (int i=0; i<n_total; ++i) {
            states[i].P = 200.0;
            states[i].Sw = 0.2;
            states[i].Sg = 0.05;
        }
        states_prev = states;
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

    // ---------------------------------------------------------------------
    // 6.12 Residual (same structure as your original, but leaf-based)
    // ---------------------------------------------------------------------

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

    // ---------------------------------------------------------------------
    // 6.13 Newton step (numerical Jacobian) - kept for compatibility
    //      Warning: this is the main scalability bottleneck after LGR.
    // ---------------------------------------------------------------------
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
            }

            BiCGSTAB<SparseMatrix<double>, IncompleteLUT<double>> solver;
            solver.preconditioner().setDroptol(1e-5);
            solver.preconditioner().setFillfactor(40);
            solver.setTolerance(1e-5);
            solver.setMaxIterations(500);
            
            solver.compute(J);
            if (solver.info() != Success) {
                states = backup;
                return false;
            }
            
            VectorXd delta = solver.solve(-Rg);
            
            if (solver.info() != Success) {
                states = backup;
                return false;
            }

            std::vector<State> states_before_ls = states;
            double alpha = 1.0;

            for(int i=0; i<n_total; ++i) {
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

                if (std::abs(dfw_dSw * dSw_new) > F_tol) {
                    omega_w = F_tol / std::max(1e-12, std::abs(dfw_dSw * dSw_new));
                }
                if (std::abs(dfg_dSg * dSg_new) > F_tol) {
                    omega_g = F_tol / std::max(1e-12, std::abs(dfg_dSg * dSg_new));
                }

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

    void exportStaticGeometry() {
        std::ofstream gridFile("grid_info_lgr.csv");
        gridFile << Nx << "," << Ny << "," << Nz << ","
                 << Lx << "," << Ly << "," << Lz << ","
                 << dx << "," << dy << "," << dz << "\n";
        gridFile.close();

        std::ofstream fracFile("fracture_geometry_lgr.csv");
        fracFile << "id,x0,y0,z0,x1,y1,z1,x2,y2,z2,x3,y3,z3\n";
        for (const auto& f : fractures) {
            fracFile << f.id;
            for (int i = 0; i < 4; ++i) {
                fracFile << "," << f.vertices[i].x 
                         << "," << f.vertices[i].y 
                         << "," << f.vertices[i].z;
            }
            fracFile << "\n";
        }
        fracFile.close();
        
        std::cout << "Geometry exported: grid_info_lgr.csv and fracture_geometry_lgr.csv" << std::endl;
    }

    void exportWells() {
        std::ofstream wellFile("well_info_lgr.csv");
        wellFile << "well_id,node_idx,type,x,y,z,WI,P_bhp\n";
        
        for(size_t i=0; i<wells.size(); ++i) {
            int u = wells[i].target_node_idx;
            double x, y, z;
            std::string type;

            if (u < n_leaf) {
                x = leaves[u].center.x;
                y = leaves[u].center.y;
                z = leaves[u].center.z;
                type = "Leaf";
            } else {
                int seg_idx = u - n_leaf;
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
        std::cout << "Wells exported: well_info_lgr.csv" << std::endl;
    }

    // ---------------------------------------------------------------------
    // 6.14 Run with adaptive dt (same pattern)
    // ---------------------------------------------------------------------
    void run(double total_days) {
        std::ofstream file("output_sim_lgr.csv");
        file << "Time,CumOil,CumWater,CumGas,AvgPressure,DT,nLeaf,nSeg,Qo,Qw,Qg\n";

        double t = 0.0;

        const double dt0    = 1e-5;
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

            // 成功推进
            t += dt_try;
            states_prev = states;
            tot_o += so; tot_w += sw; tot_g += sg;

            double avgP = 0.0, minP = 1e30, maxP = -1e30;
            for (int i = 0; i < n_leaf; ++i) {
                avgP += states[i].P;
                minP = std::min(minP, states[i].P);
                maxP = std::max(maxP, states[i].P);
            }
            avgP /= std::max(1, n_leaf);

            double qo = so / std::max(dt_try, 1e-30);
            double qw = sw / std::max(dt_try, 1e-30);
            double qg = sg / std::max(dt_try, 1e-30);

            file << t << "," << tot_o << "," << tot_w << "," << tot_g << "," << avgP << "," << dt_try
                << "," << n_leaf << "," << n_seg
                << "," << qo << "," << qw << "," << qg << "\n";
            file.flush();
            
            std::cout << "  t=" << t << " days, P: avg=" << avgP << " min=" << minP << " max=" << maxP << " bar" << std::endl;

            double fac = std::pow((double)target_iter / (double)std::max(1, actual_iter), 0.5);
            fac = std::max(0.5, std::min(1.5, fac));
            dt_try = std::min(dt_max, std::max(dt0, dt_try) * fac);
        }

        file.close();





        // export leaf final field
        std::ofstream field("final_field_lgr.csv");
        field << "leaf_id,parent_id,x,y,z,dx,dy,dz,P,Sw,Sg\n";
        for (int i=0;i<n_leaf;++i) {
            field << i << "," << leaves[i].parent_id << ","
                  << leaves[i].center.x << "," << leaves[i].center.y << "," << leaves[i].center.z << ","
                  // 添加尺寸输出
                  << leaves[i].dx << "," << leaves[i].dy << "," << leaves[i].dz << ","
                  << states[i].P << "," << states[i].Sw << "," << states[i].Sg << "\n";
        }
        field.close();

        // export segments
        std::ofstream segf("segments_lgr.csv");
        segf << "seg_id,frac_id,leaf_id,parent_id,area,cx,cy,cz,Tmf\n";
        for (const auto& s : segments) {
            segf << s.id << "," << s.frac_id << "," << s.matrix_leaf_id << "," << s.parent_id << ","
                 << s.area << "," << s.center.x << "," << s.center.y << "," << s.center.z << "," << s.T_mf << "\n";
        }
        segf.close();
    }

    // ---------------------------------------------------------------------
    // 6.15 One-click pipeline
    // ---------------------------------------------------------------------
    void preprocess() {
        buildParentGrid();
        generateFractures(num_fractures_, min_length_, max_length_, 
                         max_dip_, min_strike_, max_strike_, 
                         aperture_, perm_f_);
        generateHydraulicFractures();
        markRefinement();
        buildLeafGrid();
        buildParentFaceLeaves();

        std::vector<Connection> mm, mf, ff;
        buildMMConnections(mm);
        buildSegmentsAndMF(mf);
        buildFFConnections(ff);
        buildAllConnections(mm, mf, ff);

        setupWells();
        initState();
        buildJacobianPattern();
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
        std::cout << "DEBUG setFractureParameters: aperture=" << aperture << std::endl;
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
    
    std::vector<std::tuple<double,double,double,double>> getPressureData() {
        std::vector<std::tuple<double,double,double,double>> result;
        for (int i = 0; i < n_leaf; ++i) {
            result.push_back(std::make_tuple(
                leaves[i].center.x,
                leaves[i].center.y,
                leaves[i].center.z,
                states[i].P
            ));
        }
        return result;
    }
    
    std::vector<std::tuple<double,double,double,double>> getFractureVertices() {
        std::vector<std::tuple<double,double,double,double>> result;
        for (const auto& f : fractures) {
            for (int i = 0; i < 4; ++i) {
                result.push_back(std::make_tuple(
                    f.vertices[i].x,
                    f.vertices[i].y,
                    f.vertices[i].z,
                    (double)f.id
                ));
            }
        }
        return result;
    }

    std::vector<std::tuple<double,double,double,double,double,double>> getGridLines() {
        std::vector<std::tuple<double,double,double,double,double,double>> result;
        for (int i = 0; i < n_leaf; ++i) {
            const auto& leaf = leaves[i];
            double x_min = leaf.center.x - leaf.dx * 0.5;
            double x_max = leaf.center.x + leaf.dx * 0.5;
            double y_min = leaf.center.y - leaf.dy * 0.5;
            double y_max = leaf.center.y + leaf.dy * 0.5;
            double z_min = leaf.center.z - leaf.dz * 0.5;
            double z_max = leaf.center.z + leaf.dz * 0.5;

            // 12条边: (x_min,y_min,z_min)-(x_max,y_min,z_min)
            result.push_back(std::make_tuple(x_min, y_min, z_min, x_max, y_min, z_min));
            result.push_back(std::make_tuple(x_min, y_max, z_min, x_max, y_max, z_min));
            result.push_back(std::make_tuple(x_min, y_min, z_max, x_max, y_min, z_max));
            result.push_back(std::make_tuple(x_min, y_max, z_max, x_max, y_max, z_max));

            result.push_back(std::make_tuple(x_min, y_min, z_min, x_min, y_max, z_min));
            result.push_back(std::make_tuple(x_max, y_min, z_min, x_max, y_max, z_min));
            result.push_back(std::make_tuple(x_min, y_min, z_max, x_min, y_max, z_max));
            result.push_back(std::make_tuple(x_max, y_min, z_max, x_max, y_max, z_max));

            result.push_back(std::make_tuple(x_min, y_min, z_min, x_min, y_min, z_max));
            result.push_back(std::make_tuple(x_max, y_min, z_min, x_max, y_min, z_max));
            result.push_back(std::make_tuple(x_min, y_max, z_min, x_min, y_max, z_max));
            result.push_back(std::make_tuple(x_max, y_max, z_min, x_max, y_max, z_max));
        }
        return result;
    }

    std::vector<std::tuple<double,double,double,double>> getInterpolatedPressureField(int nx, int ny, int nz) {
        std::vector<std::tuple<double,double,double,double>> result;
        result.reserve(nx * ny * nz);
        
        if (n_leaf == 0) {
            return result;
        }
        
        double min_p = states[0].P;
        double max_p = states[0].P;
        for (int i = 0; i < n_leaf; ++i) {
            min_p = std::min(min_p, states[i].P);
            max_p = std::max(max_p, states[i].P);
        }
        
        for (int k = 0; k < nz; ++k) {
            double z = Lz * k / (nz - 1);
            for (int j = 0; j < ny; ++j) {
                double y = Ly * j / (ny - 1);
                for (int i = 0; i < nx; ++i) {
                    double x = Lx * i / (nx - 1);
                    
                    double min_dist = 1e30;
                    double nearest_p = min_p;
                    
                    for (int leaf_idx = 0; leaf_idx < n_leaf; ++leaf_idx) {
                        double dx = x - leaves[leaf_idx].center.x;
                        double dy = y - leaves[leaf_idx].center.y;
                        double dz = z - leaves[leaf_idx].center.z;
                        double dist = std::abs(dx) + std::abs(dy) + std::abs(dz);
                        
                        if (dist < min_dist) {
                            min_dist = dist;
                            nearest_p = states[leaf_idx].P;
                            if (dist < 1e-6) {
                                break;
                            }
                        }
                    }
                    
                    result.push_back(std::make_tuple(x, y, z, nearest_p));
                }
            }
        }
        
        return result;
    }

    SimulationResult runSimulation() {
        SimulationResult result;
        std::cout << "=== Simulation Parameters ===" << std::endl;
        std::cout << "Grid: " << Nx << "x" << Ny << "x" << Nz << ", Domain: " << Lx << "x" << Ly << "x" << Lz << std::endl;
        std::cout << "Perm: " << perm_x_ << ", " << perm_y_ << ", " << perm_z_ << " Darcy" << std::endl;
        std::cout << "Porosity: " << porosity_ << std::endl;
        std::cout << "Fractures: " << num_fractures_ << ", Length: " << min_length_ << "-" << max_length_ << " m" << std::endl;
        std::cout << "Fracture aperture: " << aperture_ << " m, perm: " << perm_f_ << " Darcy" << std::endl;
        std::cout << "Well: (" << well_x_ << ", " << well_y_ << ", " << well_z_ << ")" << std::endl;
        std::cout << "Well: P_bhp=" << well_pressure_ << " bar, radius=" << well_radius_ << std::endl;
        std::cout << "Simulation time: " << simulation_time_ << " days" << std::endl;
        std::cout << "=============================" << std::endl;
        
        std::cout << "Preprocessing (Parent->Leaf LGR + EDFM)..." << std::endl;
        preprocess();
        
        std::cout << "Running simulation..." << std::endl;
        run(simulation_time_);
        
        std::cout << "Simulation complete." << std::endl;
        
        // Debug: check pressure range
        double minP = 1e30, maxP = -1e30, avgP = 0.0;
        for (int i = 0; i < n_leaf; ++i) {
            minP = std::min(minP, states[i].P);
            maxP = std::max(maxP, states[i].P);
            avgP += states[i].P;
        }
        avgP /= n_leaf;
        std::cout << "Final pressure range: min=" << minP << " max=" << maxP << " avg=" << avgP << " bar" << std::endl;
        
        for (int i = 0; i < n_leaf; ++i) {
            result.pressure_field.push_back(std::make_tuple(
                leaves[i].center.x,
                leaves[i].center.y,
                leaves[i].center.z,
                states[i].P
            ));
        }
        
        for (const auto& f : fractures) {
            for (int i = 0; i < 4; ++i) {
                result.fracture_vertices.push_back(std::make_tuple(
                    f.vertices[i].x,
                    f.vertices[i].y,
                    f.vertices[i].z,
                    (double)f.id
                ));
            }
        }
        
        return result;
    }

private:
    int num_fractures_{100};
    double min_length_{30.0}, max_length_{80.0};
    double max_dip_{PI/3.0};
    double min_strike_{0.0}, max_strike_{PI};
    double aperture_{0.001};
    double perm_f_{10000.0};
    double well_x_{500.0}, well_y_{250.0}, well_z_{50.0};
    double well_radius_{0.05};
    double well_pressure_{50.0};
    double simulation_time_{100.0};
    double time_step_{1.0};
    double porosity_{0.2};
    double perm_x_{0.001}, perm_y_{0.001}, perm_z_{0.0001};
};

// =============================================================================
// Python Module
// =============================================================================

PYBIND11_MODULE(edfm_core, m) {
    py::class_<SimulationResult>(m, "SimulationResult")
        .def_readwrite("pressure_field", &SimulationResult::pressure_field)
        .def_readwrite("temperature_field", &SimulationResult::temperature_field)
        .def_readwrite("stress_field", &SimulationResult::stress_field)
        .def_readwrite("fracture_vertices", &SimulationResult::fracture_vertices)
        .def_readwrite("fracture_cells", &SimulationResult::fracture_cells);
    
    py::class_<SimulatorLGR>(m, "EDFMSimulator")
        .def(py::init<>())
        .def("setGridParameters", &SimulatorLGR::setGridParameters)
        .def("setFractureParameters", &SimulatorLGR::setFractureParameters)
        .def("setWellParameters", &SimulatorLGR::setWellParameters)
        .def("setSimulationParameters", &SimulatorLGR::setSimulationParameters)
        .def("runSimulation", &SimulatorLGR::runSimulation)
        .def("getPressureData", &SimulatorLGR::getPressureData)
        .def("getFractureVertices", &SimulatorLGR::getFractureVertices)
        .def("getGridLines", &SimulatorLGR::getGridLines)
        .def("getInterpolatedPressureField", &SimulatorLGR::getInterpolatedPressureField);
}