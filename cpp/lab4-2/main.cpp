#include <iostream>
#include <fstream>
#include <sstream>
#include <tuple>
#include <vector>
#include <array>
#include <cmath>
#include <functional>
#include <string>
#include <iomanip>
#include <filesystem>
#include <algorithm>
#include <numeric>

using namespace std;

using Point3D = array<double, 3>;

Point3D make_point(double x, double y, double z) {
    return {x, y, z};
}

double get_x(const Point3D& p) { return p[0]; }
double get_y(const Point3D& p) { return p[1]; }
double get_z(const Point3D& p) { return p[2]; }

double rastrigin_3d(const Point3D& p) {
    double x = get_x(p), y = get_y(p), z = get_z(p);
    const double A = 10.0;
    return A * 3 + (x*x - A * cos(2 * M_PI * x))
                + (y*y - A * cos(2 * M_PI * y))
                + (z*z - A * cos(2 * M_PI * z));
}

double rosenbrock_3d(const Point3D& p) {
    double x = get_x(p), y = get_y(p), z = get_z(p);
    return 100 * pow(y - x*x, 2) + pow(1 - x, 2)
         + 100 * pow(z - y*y, 2) + pow(1 - y, 2);
}

double schwefel_3d(const Point3D& p) {
    double x = get_x(p), y = get_y(p), z = get_z(p);
    const double A = 418.9829;
    auto term = [](double v) {
        return -v * sin(sqrt(fabs(v)));
    };
    return A * 3 + term(x) + term(y) + term(z);
}

double distance(const Point3D& a, const Point3D& b) {
    double dx = get_x(a) - get_x(b);
    double dy = get_y(a) - get_y(b);
    double dz = get_z(a) - get_z(b);
    return sqrt(dx*dx + dy*dy + dz*dz);
}

Point3D centroid(const vector<Point3D>& points) {
    Point3D c = {0, 0, 0};
    for (const auto& p : points) {
        c[0] += get_x(p);
        c[1] += get_y(p);
        c[2] += get_z(p);
    }
    c[0] /= points.size();
    c[1] /= points.size();
    c[2] /= points.size();
    return c;
}

Point3D reflect(const Point3D& point, const Point3D& cent, double alpha = 1.0) {
    return make_point(
        get_x(cent) + alpha * (get_x(cent) - get_x(point)),
        get_y(cent) + alpha * (get_y(cent) - get_y(point)),
        get_z(cent) + alpha * (get_z(cent) - get_z(point))
    );
}

struct Tetrahedron {
    array<Point3D, 4> vertices;
    array<double, 4> values;
    
    void evaluate(function<double(const Point3D&)> func) {
        for (int i = 0; i < 4; ++i) {
            values[i] = func(vertices[i]);
        }
    }
    
    auto best_index() const {
        return min_element(values.begin(), values.end()) - values.begin();
    }
    
    auto worst_index() const {
        return max_element(values.begin(), values.end()) - values.begin();
    }
    
    Point3D best_vertex() const { return vertices[best_index()]; }
    Point3D worst_vertex() const { return vertices[worst_index()]; }
    double best_value() const { return values[best_index()]; }
    double worst_value() const { return values[worst_index()]; }
    
    double size() const {
        double max_dist = 0;
        for (int i = 0; i < 4; ++i)
            for (int j = i+1; j < 4; ++j)
                max_dist = max(max_dist, distance(vertices[i], vertices[j]));
        return max_dist;
    }
};

struct RegularSimplex3DResult {
    vector<Tetrahedron> tetrahedrons;
    vector<Point3D> best_trajectory;
    Point3D optimum;
    double f_opt;
    int iterations;
};

struct NelderMead3DResult {
    vector<Tetrahedron> tetrahedrons;
    vector<Point3D> best_trajectory;
    Point3D optimum;
    double f_opt;
    int iterations;
};

// ==================== МЕТОДЫ ОПТИМИЗАЦИИ ====================

Tetrahedron create_regular_tetrahedron(const Point3D& center, double side) {
    Tetrahedron t;
    double r = side * sqrt(6) / 4;
    
    t.vertices[0] = make_point(
        get_x(center) + r * 2 * sqrt(2) / 3,
        get_y(center) - r * sqrt(2) / 3,
        get_z(center) - r * sqrt(2) / 3
    );
    t.vertices[1] = make_point(
        get_x(center) - r * sqrt(2) / 3,
        get_y(center) + r * 2 * sqrt(2) / 3,
        get_z(center) - r * sqrt(2) / 3
    );
    t.vertices[2] = make_point(
        get_x(center) - r * sqrt(2) / 3,
        get_y(center) - r * sqrt(2) / 3,
        get_z(center) + r * 2 * sqrt(2) / 3
    );
    t.vertices[3] = make_point(
        get_x(center) - r * sqrt(2) / 3,
        get_y(center) - r * sqrt(2) / 3,
        get_z(center) - r * sqrt(2) / 3
    );
    
    return t;
}

RegularSimplex3DResult regular_simplex_3d(
    const Point3D& x0,
    function<double(const Point3D&)> func,
    double initial_side = 1.0,
    double eps = 1e-6,
    double beta = 0.5
) {
    RegularSimplex3DResult result;
    
    Tetrahedron current = create_regular_tetrahedron(x0, initial_side);
    current.evaluate(func);
    
    result.tetrahedrons.push_back(current);
    result.best_trajectory.push_back(current.best_vertex());
    
    int iter = 0;
    const int MAX_ITER = 1000;
    
    while (current.size() > eps && iter < MAX_ITER) {
        auto best = current.best_index();
        auto worst = current.worst_index();
        
        vector<Point3D> good_vertices;
        for (int i = 0; i < 4; ++i)
            if (i != worst) good_vertices.push_back(current.vertices[i]);
        Point3D cent = centroid(good_vertices);
        
        Point3D reflected = reflect(current.vertices[worst], cent, 1.0);
        double f_reflected = func(reflected);
        
        bool improved = false;
        
        if (f_reflected < current.best_value()) {
            Point3D expanded = reflect(current.vertices[worst], cent, 2.0);
            double f_expanded = func(expanded);
            
            Tetrahedron new_tet;
            new_tet = current;
            new_tet.vertices[worst] = (f_expanded < f_reflected) ? expanded : reflected;
            new_tet.evaluate(func);
            current = new_tet;
            improved = true;
        }
        else if (f_reflected < current.worst_value()) {
            Tetrahedron new_tet = current;
            new_tet.vertices[worst] = reflected;
            new_tet.evaluate(func);
            current = new_tet;
            improved = true;
        }
        else {
            Point3D contracted = reflect(current.vertices[worst], cent, 0.5);
            double f_contracted = func(contracted);
            
            if (f_contracted < current.worst_value()) {
                Tetrahedron new_tet = current;
                new_tet.vertices[worst] = contracted;
                new_tet.evaluate(func);
                current = new_tet;
                improved = true;
            }
        }
        
        if (!improved) {
            Point3D best_v = current.best_vertex();
            for (int i = 0; i < 4; ++i) {
                if (i != best) {
                    current.vertices[i] = make_point(
                        get_x(best_v) + beta * (get_x(current.vertices[i]) - get_x(best_v)),
                        get_y(best_v) + beta * (get_y(current.vertices[i]) - get_y(best_v)),
                        get_z(best_v) + beta * (get_z(current.vertices[i]) - get_z(best_v))
                    );
                }
            }
            current.evaluate(func);
        }
        
        result.tetrahedrons.push_back(current);
        result.best_trajectory.push_back(current.best_vertex());
        iter++;
    }
    
    result.optimum = current.best_vertex();
    result.f_opt = current.best_value();
    result.iterations = iter;
    
    return result;
}

NelderMead3DResult nelder_mead_3d(
    const Point3D& x0,
    function<double(const Point3D&)> func,
    double initial_side = 1.0,
    double eps = 1e-6,
    double alpha = 1.0,
    double gamma = 2.0,
    double rho = 0.5,
    double sigma = 0.5
) {
    NelderMead3DResult result;
    
    Tetrahedron current;
    current.vertices[0] = x0;
    current.vertices[1] = make_point(get_x(x0) + initial_side, get_y(x0), get_z(x0));
    current.vertices[2] = make_point(get_x(x0), get_y(x0) + initial_side, get_z(x0));
    current.vertices[3] = make_point(get_x(x0), get_y(x0), get_z(x0) + initial_side);
    current.evaluate(func);
    
    result.tetrahedrons.push_back(current);
    result.best_trajectory.push_back(current.best_vertex());
    
    int iter = 0;
    const int MAX_ITER = 1000;
    
    while (current.size() > eps && iter < MAX_ITER) {
        array<int, 4> order = {0, 1, 2, 3};
        sort(order.begin(), order.end(), [&](int a, int b) {
            return current.values[a] < current.values[b];
        });
        
        int best = order[0];
        int good = order[1];
        int bad = order[2];
        int worst = order[3];
        
        Point3D cent = centroid({current.vertices[best], current.vertices[good], current.vertices[bad]});
        
        Point3D reflected = reflect(current.vertices[worst], cent, alpha);
        double f_reflected = func(reflected);
        
        if (f_reflected < current.values[best]) {
            Point3D expanded = reflect(current.vertices[worst], cent, alpha * gamma);
            double f_expanded = func(expanded);
            
            current.vertices[worst] = (f_expanded < f_reflected) ? expanded : reflected;
        }
        else if (f_reflected < current.values[bad]) {
            current.vertices[worst] = reflected;
        }
        else if (f_reflected < current.values[worst]) {
            Point3D contracted_out = reflect(current.vertices[worst], cent, alpha * rho);
            double f_contracted = func(contracted_out);
            
            if (f_contracted < f_reflected) {
                current.vertices[worst] = contracted_out;
            } else {
                for (int i = 0; i < 4; ++i) {
                    if (i != best) {
                        current.vertices[i] = make_point(
                            get_x(current.vertices[best]) + sigma * (get_x(current.vertices[i]) - get_x(current.vertices[best])),
                            get_y(current.vertices[best]) + sigma * (get_y(current.vertices[i]) - get_y(current.vertices[best])),
                            get_z(current.vertices[best]) + sigma * (get_z(current.vertices[i]) - get_z(current.vertices[best]))
                        );
                    }
                }
            }
        }
        else {
            Point3D contracted_in = reflect(current.vertices[worst], cent, -rho);
            double f_contracted = func(contracted_in);
            
            if (f_contracted < current.values[worst]) {
                current.vertices[worst] = contracted_in;
            } else {
                for (int i = 0; i < 4; ++i) {
                    if (i != best) {
                        current.vertices[i] = make_point(
                            get_x(current.vertices[best]) + sigma * (get_x(current.vertices[i]) - get_x(current.vertices[best])),
                            get_y(current.vertices[best]) + sigma * (get_y(current.vertices[i]) - get_y(current.vertices[best])),
                            get_z(current.vertices[best]) + sigma * (get_z(current.vertices[i]) - get_z(current.vertices[best]))
                        );
                    }
                }
            }
        }
        
        current.evaluate(func);
        result.tetrahedrons.push_back(current);
        result.best_trajectory.push_back(current.best_vertex());
        iter++;
    }
    
    result.optimum = current.best_vertex();
    result.f_opt = current.best_value();
    result.iterations = iter;
    
    return result;
}

string format_double(double val) {
    ostringstream oss;
    oss << fixed << setprecision(6) << val;
    return oss.str();
}

string get_output_dir() {
    string out_dir = "./output";
    if (!filesystem::exists(out_dir)) {
        filesystem::create_directory(out_dir);
    }
    return out_dir;
}

string get_color_gradient(double t, bool is_regular) {
    int hue = is_regular ? 200 : 30;
    int sat = 70;
    int light = static_cast<int>(80 - 60 * t);
    
    ostringstream oss;
    oss << "'hsl(" << hue << "," << sat << "%," << light << "%)'";
    return oss.str();
}

template<typename ResultType>
void create_html_tetrahedrons(
    const ResultType& result,
    const string& filepath,
    const string& title,
    const string& method_name,
    bool is_regular
) {
    const auto& tets = result.tetrahedrons;
    
    ofstream f(filepath);
    if (!f.is_open()) {
        cerr << "Error: cannot create HTML " << filepath << endl;
        return;
    }
    
    if (tets.empty()) {
        cerr << "Error: empty tetrahedron sequence" << endl;
        return;
    }
    
    stringstream traces;
    
    for (size_t t = 0; t < tets.size(); ++t) {
        double progress = static_cast<double>(t) / (tets.size() - 1);
        string color = get_color_gradient(progress, is_regular);
        
        const array<pair<int,int>, 6> edges = {{
            {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}
        }};
        
        for (const auto& e : edges) {
            int i = e.first, j = e.second;
            
            traces << "        {\n";
            traces << "            type: 'scatter3d',\n";
            traces << "            mode: 'lines',\n";
            traces << "            x: [" << format_double(get_x(tets[t].vertices[i])) << ","
                   << format_double(get_x(tets[t].vertices[j])) << "],\n";
            traces << "            y: [" << format_double(get_y(tets[t].vertices[i])) << ","
                   << format_double(get_y(tets[t].vertices[j])) << "],\n";
            traces << "            z: [" << format_double(get_z(tets[t].vertices[i])) << ","
                   << format_double(get_z(tets[t].vertices[j])) << "],\n";
            traces << "            line: { color: " << color << ", width: 2 },\n";
            traces << "            showlegend: false,\n";
            traces << "            hoverinfo: 'none'\n";
            traces << "        },\n";
        }
    }
    
    string traj_x, traj_y, traj_z;
    for (size_t t = 0; t < tets.size(); ++t) {
        if (t > 0) {
            traj_x += ",";
            traj_y += ",";
            traj_z += ",";
        }
        Point3D best = tets[t].best_vertex();
        traj_x += format_double(get_x(best));
        traj_y += format_double(get_y(best));
        traj_z += format_double(get_z(best));
    }
    
    traces << "        {\n";
    traces << "            type: 'scatter3d',\n";
    traces << "            mode: 'lines+markers',\n";
    traces << "            x: [" << traj_x << "],\n";
    traces << "            y: [" << traj_y << "],\n";
    traces << "            z: [" << traj_z << "],\n";
    traces << "            line: { color: '#1f77b4', width: 4 },\n";
    traces << "            marker: { size: 4, color: '#1f77b4' },\n";
    traces << "            name: 'Trajectory',\n";
    traces << "            showlegend: true\n";
    traces << "        },\n";
    
    Point3D start = result.best_trajectory[0];
    traces << "        {\n";
    traces << "            type: 'scatter3d',\n";
    traces << "            mode: 'markers',\n";
    traces << "            x: [" << format_double(get_x(start)) << "],\n";
    traces << "            y: [" << format_double(get_y(start)) << "],\n";
    traces << "            z: [" << format_double(get_z(start)) << "],\n";
    traces << "            marker: { size: 15, color: '#2ca02c', symbol: 'diamond', line: {color: 'black', width: 2} },\n";
    traces << "            name: 'Start',\n";
    traces << "            showlegend: true\n";
    traces << "        },\n";
    
    Point3D end = result.optimum;
    traces << "        {\n";
    traces << "            type: 'scatter3d',\n";
    traces << "            mode: 'markers',\n";
    traces << "            x: [" << format_double(get_x(end)) << "],\n";
    traces << "            y: [" << format_double(get_y(end)) << "],\n";
    traces << "            z: [" << format_double(get_z(end)) << "],\n";
    traces << "            marker: { size: 18, color: '#d62728', symbol: 'star', line: {color: 'black', width: 2} },\n";
    traces << "            name: 'Optimum',\n";
    traces << "            showlegend: true\n";
    traces << "        }\n";
    
    // HTML шаблон
    f << "<!DOCTYPE html>\n";
    f << "<html>\n";
    f << "<head>\n";
    f << "    <meta charset=\"UTF-8\">\n";
    f << "    <title>" << title << " - " << method_name << "</title>\n";
    f << "    <script src=\"https://cdn.plot.ly/plotly-2.27.0.min.js \"></script>\n";
    f << "    <style>\n";
    f << "        body { \n";
    f << "            margin: 0; \n";
    f << "            font-family: Arial, sans-serif;\n";
    f << "            background: #f0f0f0;\n";
    f << "        }\n";
    f << "        #plot { \n";
    f << "            width: 100vw; \n";
    f << "            height: 85vh; \n";
    f << "        }\n";
    f << "        .header { \n";
    f << "            height: 15vh;\n";
    f << "            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);\n";
    f << "            color: white;\n";
    f << "            display: flex;\n";
    f << "            align-items: center;\n";
    f << "            justify-content: space-between;\n";
    f << "            padding: 0 30px;\n";
    f << "        }\n";
    f << "        .header h2 { margin: 0; font-weight: 300; }\n";
    f << "        .stats {\n";
    f << "            background: rgba(255,255,255,0.2);\n";
    f << "            padding: 10px 20px;\n";
    f << "            border-radius: 20px;\n";
    f << "            font-size: 14px;\n";
    f << "        }\n";
    f << "        .legend-info {\n";
    f << "            position: absolute;\n";
    f << "            bottom: 20px;\n";
    f << "            left: 20px;\n";
    f << "            background: rgba(255,255,255,0.95);\n";
    f << "            padding: 15px;\n";
    f << "            border-radius: 10px;\n";
    f << "            box-shadow: 0 2px 10px rgba(0,0,0,0.2);\n";
    f << "            font-size: 13px;\n";
    f << "        }\n";
    f << "        .legend-item { margin: 5px 0; }\n";
    f << "        .color-box {\n";
    f << "            display: inline-block;\n";
    f << "            width: 25px;\n";
    f << "            height: 12px;\n";
    f << "            margin-right: 8px;\n";
    f << "            vertical-align: middle;\n";
    f << "            border: 1px solid #999;\n";
    f << "        }\n";
    f << "    </style>\n";
    f << "</head>\n";
    f << "<body>\n";
    f << "    <div class=\"header\">\n";
    f << "        <h2>" << title << " - " << method_name << "</h2>\n";
    f << "        <div class=\"stats\">\n";
    f << "            Tetrahedrons: " << tets.size() << " | Iterations: " << result.iterations <<
          " | f_min: " << fixed << setprecision(6) << result.f_opt << "\n";
    f << "        </div>\n";
    f << "    </div>\n";
    f << "    <div id=\"plot\"></div>\n";
    f << "    <div class=\"legend-info\">\n";
    f << "        <div class=\"legend-item\"><strong>Color scale (tetrahedrons):</strong></div>\n";
    f << "        <div class=\"legend-item\"><span class=\"color-box\" style=\"background: hsl(" <<
          (is_regular ? 200 : 30) << ",70%,80%)\"></span> Start (light)</div>\n";
    f << "        <div class=\"legend-item\"><span class=\"color-box\" style=\"background: hsl(" <<
          (is_regular ? 200 : 30) << ",70%,20%)\"></span> End (dark)</div>\n";
    f << "        <div class=\"legend-item\" style=\"margin-top:10px;\"><strong>Points:</strong></div>\n";
    f << "        <div class=\"legend-item\">&#128312; Green diamond = Start point</div>\n";
    f << "        <div class=\"legend-item\">&#128308; Red star = Optimum</div>\n";
    f << "        <div class=\"legend-item\">&#128309; Blue line = Best vertex trajectory</div>\n";
    f << "    </div>\n";
    f << "    <script>\n";
    f << "        var traces = [\n" << traces.str() << "\n        ];\n";
    f << "        var layout = {\n";
    f << "            scene: {\n";
    f << "                xaxis: { title: 'X', gridcolor: 'rgb(200,200,200)' },\n";
    f << "                yaxis: { title: 'Y', gridcolor: 'rgb(200,200,200)' },\n";
    f << "                zaxis: { title: 'Z', gridcolor: 'rgb(200,200,200)' },\n";
    f << "                bgcolor: 'rgb(245,245,245)',\n";
    f << "                camera: { eye: { x: 1.8, y: 1.8, z: 1.5 } }\n";
    f << "            },\n";
    f << "            margin: { l: 0, r: 0, b: 0, t: 0 },\n";
    f << "            showlegend: true,\n";
    f << "            legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(255,255,255,0.9)' }\n";
    f << "        };\n";
    f << "        var config = { responsive: true, displayModeBar: true, displaylogo: false };\n";
    f << "        Plotly.newPlot('plot', traces, layout, config);\n";
    f << "    </script>\n";
    f << "</body>\n";
    f << "</html>\n";
    
    f.close();
}

void run_comparison_test(
    const string& name,
    function<double(const Point3D&)> func,
    const Point3D& x0,
    double initial_side = 2.0
) {
    cout << "\n========================================" << endl;
    cout << "Function: " << name << endl;
    cout << "Start: (" << get_x(x0) << ", " << get_y(x0) << ", " << get_z(x0) << ")" << endl;
    cout << "========================================" << endl;
    
    cout << "\n--- Regular Simplex ---" << endl;
    auto reg_result = regular_simplex_3d(x0, func, initial_side);
    cout << "Iterations: " << reg_result.iterations << endl;
    cout << "Optimum: (" << fixed << setprecision(6)
         << get_x(reg_result.optimum) << ", " << get_y(reg_result.optimum)
         << ", " << get_z(reg_result.optimum) << "), f = " << reg_result.f_opt << endl;
    
    cout << "\n--- Nelder-Mead ---" << endl;
    auto nm_result = nelder_mead_3d(x0, func, initial_side);
    cout << "Iterations: " << nm_result.iterations << endl;
    cout << "Optimum: (" << get_x(nm_result.optimum) << ", " << get_y(nm_result.optimum)
         << ", " << get_z(nm_result.optimum) << "), f = " << nm_result.f_opt << endl;
    
    string out_dir = get_output_dir();
    
    create_html_tetrahedrons(reg_result, out_dir + "/" + name + "_regular.html",
                            name, "Regular Simplex", true);
    create_html_tetrahedrons(nm_result, out_dir + "/" + name + "_nelder_mead.html",
                            name, "Nelder-Mead", false);
}

int main(int argc, const char* argv[]) {
    cout << "Simplex Methods 3D Visualization" << endl;
    
    run_comparison_test("Rastrigin", rastrigin_3d, make_point(0.5, 0.5, 0.5), 1.0);
    run_comparison_test("Rosenbrock", rosenbrock_3d, make_point(2.0, 2.0, 2.0), 1.0);
    run_comparison_test("Schwefel", schwefel_3d, make_point(450.0, 450.0, 450.0), 10.0);
    
    return 0;
}
