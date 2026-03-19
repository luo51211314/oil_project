// =============================================================================
// 文件名: plot_pressure_field.cpp
// 描述: 读取角点网格 EDFM 模拟结果 (CSV)，绘制 z = const 水平切片上的压力场与裂缝交线
// 依赖: OpenCV
// 编译: g++ -O2 -std=c++17 plot_pressure_field.cpp -o plot_pressure_field `pkg-config --cflags --libs opencv4`
// Windows(MinGW): 按你自己的 OpenCV 路径链接
// =============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <array>
#include <map>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// ======================= 数据结构 =======================

struct Point3D {
    double x = 0, y = 0, z = 0;
};

struct GridInfo {
    int Nx = 0, Ny = 0, Nz = 0;
    double Lx = 0, Ly = 0, Lz = 0;
    double dx = 0, dy = 0, dz = 0;
};

struct CellGeom {
    int id = -1;
    std::array<Point3D, 8> corners;
};

struct FracGeo {
    int id = -1;
    Point3D v[4];
};

// ======================= 工具函数 =======================

vector<double> parseCSVLine(const string& str_in) {
    vector<double> values;
    stringstream ss(str_in);
    string token;
    while (getline(ss, token, ',')) {
        try {
            values.push_back(stod(token));
        } catch (...) {
            // 跳过表头
        }
    }
    return values;
}

double dist3(const Point3D& a, const Point3D& b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    double dz = a.z - b.z;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

void pushUniquePoint(vector<Point3D>& pts, const Point3D& p, double tol = 1e-8) {
    for (const auto& q : pts) {
        if (dist3(p, q) <= tol) return;
    }
    pts.push_back(p);
}

// Jet colormap
Scalar getJetColor(double val, double min_v, double max_v) {
    if (std::abs(max_v - min_v) < 1e-12) return Scalar(0, 255, 0);

    double norm = (val - min_v) / (max_v - min_v);
    norm = std::max(0.0, std::min(1.0, norm));

    double r = 0, g = 0, b = 0;
    if (norm < 0.25) {
        b = 1.0;
        g = norm * 4.0;
    } else if (norm < 0.5) {
        b = 1.0 - (norm - 0.25) * 4.0;
        g = 1.0;
    } else if (norm < 0.75) {
        g = 1.0;
        r = (norm - 0.5) * 4.0;
    } else {
        g = 1.0 - (norm - 0.75) * 4.0;
        r = 1.0;
    }
    return Scalar(b * 255, g * 255, r * 255);
}

// 世界坐标 -> 图像坐标
Point2f worldToImg(double wx, double wy,
                   double minX, double minY,
                   double scale, int imgH) {
    float ix = (float)((wx - minX) * scale);
    float iy = (float)(imgH - (wy - minY) * scale);
    return Point2f(ix, iy);
}

// 线段与水平平面 z = z0 求交
void addSegmentPlaneIntersection(const Point3D& p1, const Point3D& p2,
                                 double z0,
                                 vector<Point3D>& pts,
                                 double tol = 1e-8) {
    double d1 = p1.z - z0;
    double d2 = p2.z - z0;

    bool on1 = std::abs(d1) <= tol;
    bool on2 = std::abs(d2) <= tol;

    // 整条边都在平面上
    if (on1 && on2) {
        pushUniquePoint(pts, p1, tol);
        pushUniquePoint(pts, p2, tol);
        return;
    }

    // 一个端点在平面上
    if (on1) {
        pushUniquePoint(pts, p1, tol);
        return;
    }
    if (on2) {
        pushUniquePoint(pts, p2, tol);
        return;
    }

    // 真正跨过平面
    if (d1 * d2 < 0.0) {
        double t = d1 / (d1 - d2);
        Point3D p;
        p.x = p1.x + t * (p2.x - p1.x);
        p.y = p1.y + t * (p2.y - p1.y);
        p.z = z0;
        pushUniquePoint(pts, p, tol);
    }
}

// 按 XY 平面极角排序交点，形成切片多边形
vector<Point3D> sortPolygonOnXY(const vector<Point3D>& pts) {
    if (pts.size() < 3) return {};

    Point3D c{0, 0, 0};
    for (const auto& p : pts) {
        c.x += p.x;
        c.y += p.y;
        c.z += p.z;
    }
    c.x /= pts.size();
    c.y /= pts.size();
    c.z /= pts.size();

    vector<Point3D> out = pts;
    sort(out.begin(), out.end(), [&](const Point3D& a, const Point3D& b) {
        double aa = atan2(a.y - c.y, a.x - c.x);
        double bb = atan2(b.y - c.y, b.x - c.x);
        return aa < bb;
    });

    return out;
}

// 用 z = const 水平平面切一般六面体 cell，返回交线多边形
vector<Point3D> intersectCellWithZPlane(const CellGeom& cell, double z0, double tol = 1e-8) {
    // 六面体 12 条边
    static const int edges[12][2] = {
        {0,1}, {1,2}, {2,3}, {3,0},
        {4,5}, {5,6}, {6,7}, {7,4},
        {0,4}, {1,5}, {2,6}, {3,7}
    };

    vector<Point3D> pts;
    for (int e = 0; e < 12; ++e) {
        const Point3D& a = cell.corners[edges[e][0]];
        const Point3D& b = cell.corners[edges[e][1]];
        addSegmentPlaneIntersection(a, b, z0, pts, tol);
    }

    if (pts.size() < 3) return {};
    return sortPolygonOnXY(pts);
}

// 裂缝与 z = const 的交点提取
vector<Point3D> intersectFractureWithZPlane(const FracGeo& f, double z0, double tol = 1e-8) {
    vector<Point3D> pts;
    int idx[5] = {0,1,2,3,0};
    for (int i = 0; i < 4; ++i) {
        addSegmentPlaneIntersection(f.v[idx[i]], f.v[idx[i+1]], z0, pts, tol);
    }
    return pts;
}

// 若交点超过 2 个，取最远两点作为交线端点
bool getFarthestPair(const vector<Point3D>& pts, Point3D& a, Point3D& b) {
    if (pts.size() < 2) return false;
    double best = -1.0;
    for (size_t i = 0; i < pts.size(); ++i) {
        for (size_t j = i + 1; j < pts.size(); ++j) {
            double d = dist3(pts[i], pts[j]);
            if (d > best) {
                best = d;
                a = pts[i];
                b = pts[j];
            }
        }
    }
    return best > 1e-12;
}

// ======================= 主程序 =======================

int main(int argc, char** argv) {
    double slice_z = 20; // 默认后面自动取中间层
    if (argc > 1) slice_z = std::stod(argv[1]);

    cout << "=== Corner-Point Grid Visualization Tool ===" << endl;

    // ---------------------------------------------------------
    // 1. 读 grid_info.csv
    // ---------------------------------------------------------
    GridInfo grid;
    {
        ifstream fin("grid_info.csv");
        if (!fin.is_open()) {
            cerr << "Error: Cannot open grid_info.csv" << endl;
            return -1;
        }

        string line;
        if (getline(fin, line)) {
            auto t = parseCSVLine(line);
            if (t.size() >= 9) {
                grid.Nx = (int)t[0];
                grid.Ny = (int)t[1];
                grid.Nz = (int)t[2];
                grid.Lx = t[3];
                grid.Ly = t[4];
                grid.Lz = t[5];
                grid.dx = t[6];
                grid.dy = t[7];
                grid.dz = t[8];
            } else {
                cerr << "Error: Invalid grid_info.csv format" << endl;
                return -1;
            }
        }
    }

    if (slice_z < 0.0) slice_z = 0.5 * grid.Lz;

    cout << "Grid: " << grid.Nx << " x " << grid.Ny << " x " << grid.Nz << endl;
    cout << "Slice Z = " << slice_z << endl;

    // ---------------------------------------------------------
    // 2. 读 cell_geometry.csv
    // ---------------------------------------------------------
    vector<CellGeom> cellGeoms;
    double minX = 1e100, minY = 1e100;
    double maxX = -1e100, maxY = -1e100;

    {
        ifstream fin("cell_geometry.csv");
        if (!fin.is_open()) {
            cerr << "Error: Cannot open cell_geometry.csv" << endl;
            return -1;
        }

        string line;
        getline(fin, line); // skip header

        while (getline(fin, line)) {
            auto t = parseCSVLine(line);

            // cell_id..bbox..corners..face_ids
            // 角点从索引 14 开始，共 24 个数
            if (t.size() < 38) continue;

            CellGeom cg;
            cg.id = (int)t[0];

            for (int k = 0; k < 8; ++k) {
                int base = 14 + 3 * k;
                cg.corners[k] = {t[base], t[base + 1], t[base + 2]};

                minX = std::min(minX, cg.corners[k].x);
                minY = std::min(minY, cg.corners[k].y);
                maxX = std::max(maxX, cg.corners[k].x);
                maxY = std::max(maxY, cg.corners[k].y);
            }

            cellGeoms.push_back(cg);
        }
    }

    cout << "Loaded " << cellGeoms.size() << " cells from cell_geometry.csv" << endl;

    // ---------------------------------------------------------
    // 3. 读 final_field.csv
    //    支持两种格式：
    //    A) cell_id,x,y,z,P,Sw,Sg
    //    B) x,y,z,P,Sw,Sg   (此时默认行号=cell_id)
    // ---------------------------------------------------------
    map<int, double> pressureMap;
    double minP = 1e100, maxP = -1e100;

    {
        ifstream fin("final_field.csv");
        if (!fin.is_open()) {
            cerr << "Error: Cannot open final_field.csv" << endl;
            return -1;
        }

        string line;
        getline(fin, line); // skip header

        int row_id = 0;
        while (getline(fin, line)) {
            auto t = parseCSVLine(line);

            int cell_id = -1;
            double P = 0.0;

            if (t.size() >= 7) {
                // cell_id,x,y,z,P,Sw,Sg
                cell_id = (int)t[0];
                P = t[4];
            } else if (t.size() >= 6) {
                // x,y,z,P,Sw,Sg
                cell_id = row_id;
                P = t[3];
            } else {
                continue;
            }

            pressureMap[cell_id] = P;
            minP = std::min(minP, P);
            maxP = std::max(maxP, P);
            row_id++;
        }
    }

    cout << "Pressure range: " << minP << " -> " << maxP << " bar" << endl;

    // ---------------------------------------------------------
    // 4. 图像尺寸与缩放
    // ---------------------------------------------------------
    double spanX = maxX - minX;
    double spanY = maxY - minY;
    if (spanX <= 0 || spanY <= 0) {
        cerr << "Error: invalid XY bounding box." << endl;
        return -1;
    }

    double scale = 1.0;
    int imgW = (int)std::round(spanX * scale);
    if (imgW > 2000) scale = 2000.0 / spanX;
    if (imgW < 800)  scale = 800.0 / spanX;

    imgW = (int)std::round(spanX * scale);
    int imgH = (int)std::round(spanY * scale);

    Mat img(imgH, imgW, CV_8UC3, Scalar(255,255,255));

    cout << "Image size: " << imgW << " x " << imgH << ", scale = " << scale << endl;

    // ---------------------------------------------------------
    // 5. 画角点网格切片多边形
    // ---------------------------------------------------------
    int drawCount = 0;
    for (const auto& cell : cellGeoms) {
        auto it = pressureMap.find(cell.id);
        if (it == pressureMap.end()) continue;

        vector<Point3D> poly3 = intersectCellWithZPlane(cell, slice_z);
        if (poly3.size() < 3) continue;

        vector<Point> poly2;
        for (const auto& p : poly3) {
            Point2f q = worldToImg(p.x, p.y, minX, minY, scale, imgH);
            poly2.emplace_back((int)std::lround(q.x), (int)std::lround(q.y));
        }

        Scalar color = getJetColor(it->second, minP, maxP);

        const Point* pts[1] = { poly2.data() };
        int npts[] = { (int)poly2.size() };

        fillPoly(img, pts, npts, 1, color, LINE_AA);
        polylines(img, poly2, true, Scalar(120,120,120), 1, LINE_AA);

        drawCount++;
    }

    cout << "Drew " << drawCount << " intersected cells on slice Z=" << slice_z << endl;

    // ---------------------------------------------------------
    // 6. 读 fracture_geometry.csv 并画裂缝交线
    // ---------------------------------------------------------
    vector<FracGeo> fracs;
    {
        ifstream fin("fracture_geometry.csv");
        if (!fin.is_open()) {
            cerr << "Error: Cannot open fracture_geometry.csv" << endl;
            return -1;
        }

        string line;
        getline(fin, line); // skip header

        while (getline(fin, line)) {
            auto t = parseCSVLine(line);
            if (t.size() < 13) continue;

            FracGeo f;
            f.id = (int)t[0];
            f.v[0] = {t[1],  t[2],  t[3]};
            f.v[1] = {t[4],  t[5],  t[6]};
            f.v[2] = {t[7],  t[8],  t[9]};
            f.v[3] = {t[10], t[11], t[12]};
            fracs.push_back(f);
        }
    }

    int fracDrawCount = 0;
    for (const auto& f : fracs) {
        auto pts = intersectFractureWithZPlane(f, slice_z);
        if (pts.empty()) continue;

        // 情况1：裂缝整个就在切片平面内
        if (pts.size() >= 3) {
            // 若所有点都在平面上，可能形成多边形；这里简单画外框
            // 也可以进一步排序成 polygon 再画
            auto poly3 = sortPolygonOnXY(pts);
            if (poly3.size() >= 3) {
                vector<Point> poly2;
                for (const auto& p : poly3) {
                    Point2f q = worldToImg(p.x, p.y, minX, minY, scale, imgH);
                    poly2.emplace_back((int)std::lround(q.x), (int)std::lround(q.y));
                }
                polylines(img, poly2, true, Scalar(0,0,0), 2, LINE_AA);
                fracDrawCount++;
                continue;
            }
        }

        // 情况2：普通交线，取最远两点
        Point3D a, b;
        if (getFarthestPair(pts, a, b)) {
            Point2f p1 = worldToImg(a.x, a.y, minX, minY, scale, imgH);
            Point2f p2 = worldToImg(b.x, b.y, minX, minY, scale, imgH);
            cv::line(img, p1, p2, Scalar(0,0,0), 2, LINE_AA);
            fracDrawCount++;
        }
    }

    cout << "Drew " << fracDrawCount << " fracture intersections." << endl;

    // ---------------------------------------------------------
    // 7. 标注与输出
    // ---------------------------------------------------------
    rectangle(img, Point(0,0), Point(260,90), Scalar(255,255,255), FILLED);
    putText(img, "Z = " + to_string((int)slice_z) + " m",
            Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,0,0), 2);
    putText(img, "Pmax = " + to_string((int)maxP) + " bar",
            Point(10, 50), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(0,0,255), 1);
    putText(img, "Pmin = " + to_string((int)minP) + " bar",
            Point(10, 72), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(255,0,0), 1);

    string outFile = "pressure_slice_z" + to_string((int)slice_z) + "_corner.png";
    imwrite(outFile, img);

    cout << "Saved: " << outFile << endl;
    return 0;
}