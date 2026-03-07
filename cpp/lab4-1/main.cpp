#include <iostream>
#include <fstream>
#include <sstream>
#include <tuple>
#include <vector>
#include <cmath>
#include <functional>
#include <string>
#include <iomanip>
#include <filesystem>

using namespace std;

using Point3D = tuple<double, double, double>;

double get_x(const Point3D& p) { return get<0>(p); }
double get_y(const Point3D& p) { return get<1>(p); }
double get_z(const Point3D& p) { return get<2>(p); }

Point3D make_point(double x, double y, double z) {
    return make_tuple(x, y, z);
}

double rastrigin_3d(Point3D p) {
    double x = get_x(p), y = get_y(p), z = get_z(p);
    const double A = 10.0;
    return A * 3 + (x*x - A * cos(2 * M_PI * x))
                + (y*y - A * cos(2 * M_PI * y))
                + (z*z - A * cos(2 * M_PI * z));
}

double rosenbrock_3d(Point3D p) {
    double x = get_x(p), y = get_y(p), z = get_z(p);
    return 100 * pow(y - x*x, 2) + pow(1 - x, 2)
         + 100 * pow(z - y*y, 2) + pow(1 - y, 2);
}

double schwefel_3d(Point3D p) {
    double x = get_x(p), y = get_y(p), z = get_z(p);
    const double A = 418.9829;
    auto term = [](double v) {
        return -v * sin(sqrt(fabs(v)));
    };
    return A * 3 + term(x) + term(y) + term(z);
}

Point3D exploratory_search_3d(Point3D x_base, double (*func)(Point3D), double delta) {
    double x = get_x(x_base), y = get_y(x_base), z = get_z(x_base);
    double f_base = func(x_base);
    Point3D x_new = x_base;
    double f_new = f_base;
    
    vector<Point3D> directions = {
        make_point(x + delta, y, z), make_point(x - delta, y, z),
        make_point(x, y + delta, z), make_point(x, y - delta, z),
        make_point(x, y, z + delta), make_point(x, y, z - delta)
    };
    
    for (const auto& dir : directions) {
        double f_test = func(dir);
        if (f_test < f_new) {
            x_new = dir;
            f_new = f_test;
        }
    }
    return x_new;
}

struct HookeJeevesResult3D {
    vector<Point3D> trajectory;
    Point3D optimum;
    double f_opt;
    int iterations;
    int func_evals;
};

HookeJeevesResult3D hooke_jeeves_3d(Point3D x0, double (*func)(Point3D),
                                     double delta = 0.5, double eps = 1e-6, double alpha = 2.0) {
    HookeJeevesResult3D result;
    auto x_base = x0;
    double f_base = func(x_base);
    result.func_evals = 1;
    result.trajectory.push_back(x_base);
    
    int iter = 0;
    while (delta > eps) {
        auto x_new = exploratory_search_3d(x_base, func, delta);
        result.func_evals += 6;
        double f_new = func(x_new);
        result.func_evals++;
        
        if (f_new < f_base) {
            auto x_pattern = make_point(
                2 * get_x(x_new) - get_x(x_base),
                2 * get_y(x_new) - get_y(x_base),
                2 * get_z(x_new) - get_z(x_base)
            );
            auto x_pattern_result = exploratory_search_3d(x_pattern, func, delta);
            result.func_evals += 6;
            double f_pattern = func(x_pattern_result);
            result.func_evals++;
            
            if (f_pattern < f_new) {
                x_base = x_pattern_result;
                f_base = f_pattern;
            } else {
                x_base = x_new;
                f_base = f_new;
            }
            result.trajectory.push_back(x_base);
        } else {
            delta /= alpha;
        }
        iter++;
        if (iter > 1000) break;
    }
    
    result.optimum = x_base;
    result.f_opt = f_base;
    result.iterations = iter;
    return result;
}

HookeJeevesResult3D hooke_jeeves_optimized_3d(Point3D x0, double (*func)(Point3D),
                                               double delta = 0.5, double eps = 1e-6, double alpha = 2.0) {
    HookeJeevesResult3D result;
    auto x_base = x0;
    double f_base = func(x_base);
    result.func_evals = 1;
    result.trajectory.push_back(x_base);
    
    int iter = 0;
    while (delta > eps) {
        auto x_new = exploratory_search_3d(x_base, func, delta);
        result.func_evals += 6;
        double f_new = func(x_new);
        result.func_evals++;
        
        if (f_new < f_base) {
            auto x_current = x_new;
            double f_current = f_new;
            result.trajectory.push_back(x_current);
            
            while (true) {
                auto x_pattern = make_point(
                    2 * get_x(x_current) - get_x(x_base),
                    2 * get_y(x_current) - get_y(x_base),
                    2 * get_z(x_current) - get_z(x_base)
                );
                auto x_pattern_result = exploratory_search_3d(x_pattern, func, delta);
                result.func_evals += 6;
                double f_pattern = func(x_pattern_result);
                result.func_evals++;
                
                if (f_pattern < f_current) {
                    x_base = x_current;
                    f_base = f_current;
                    x_current = x_pattern_result;
                    f_current = f_pattern;
                    result.trajectory.push_back(x_current);
                } else {
                    x_base = x_current;
                    f_base = f_current;
                    break;
                }
            }
        } else {
            delta /= alpha;
        }
        iter++;
        if (iter > 1000) break;
    }
    
    result.optimum = x_base;
    result.f_opt = f_base;
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

void create_html_plotly(const HookeJeevesResult3D& result, const string& filepath,
                        const string& title, const string& method_name) {
    ofstream f(filepath);
    if (!f.is_open()) {
        cerr << "Ошибка: не могу создать HTML " << filepath << endl;
        return;
    }
    
    if (result.trajectory.empty()) {
        cerr << "Ошибка: пустая траектория для " << title << endl;
        return;
    }
    
    string x_data, y_data, z_data;
    for (size_t i = 0; i < result.trajectory.size(); ++i) {
        if (i > 0) {
            x_data += ",";
            y_data += ",";
            z_data += ",";
        }
        x_data += format_double(get_x(result.trajectory[i]));
        y_data += format_double(get_y(result.trajectory[i]));
        z_data += format_double(get_z(result.trajectory[i]));
    }
    
    double start_x = get_x(result.trajectory[0]);
    double start_y = get_y(result.trajectory[0]);
    double start_z = get_z(result.trajectory[0]);
    double end_x = get_x(result.optimum);
    double end_y = get_y(result.optimum);
    double end_z = get_z(result.optimum);
    
    f << R"(<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>)" << title << " - " << method_name << R"(</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body { 
            margin: 0; 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f5f5;
        }
        #plot { 
            width: 100vw; 
            height: 90vh; 
        }
        .header { 
            height: 10vh;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header h2 {
            margin: 0;
            font-weight: 300;
        }
        .stats {
            background: rgba(255,255,255,0.2);
            padding: 10px 20px;
            border-radius: 20px;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h2>)" << title << " — " << method_name << R"(</h2>
        <div class="stats">
            Точек: )" << result.trajectory.size() << R"( | 
            Итераций: )" << result.iterations << R"( | 
            f_min: )" << fixed << setprecision(6) << result.f_opt << R"(
        </div>
    </div>
    <div id="plot"></div>
    <script>
        console.log('Loading Plotly...');
        
        var trace = {
            type: 'scatter3d',
            mode: 'lines+markers',
            x: [)" << x_data << R"(],
            y: [)" << y_data << R"(],
            z: [)" << z_data << R"(],
            line: { 
                color: '#1f77b4', 
                width: 4
            },
            marker: { 
                size: 6, 
                color: '#ff7f0e',
                symbol: 'circle'
            },
            name: 'Траектория'
        };
        
        var startPoint = {
            type: 'scatter3d',
            mode: 'markers',
            x: [)" << format_double(start_x) << R"(],
            y: [)" << format_double(start_y) << R"(],
            z: [)" << format_double(start_z) << R"(],
            marker: {
                size: 10,
                color: '#2ca02c',
                symbol: 'diamond'
            },
            name: 'Старт'
        };
        
        var endPoint = {
            type: 'scatter3d',
            mode: 'markers',
            x: [)" << format_double(end_x) << R"(],
            y: [)" << format_double(end_y) << R"(],
            z: [)" << format_double(end_z) << R"(],
            marker: {
                size: 12,
                color: '#d62728',
                symbol: 'star'
            },
            name: 'Оптимум'
        };
        
        var layout = {
            scene: {
                xaxis: { title: 'X', gridcolor: 'rgb(200,200,200)', zerolinecolor: 'rgb(255,0,0)' },
                yaxis: { title: 'Y', gridcolor: 'rgb(200,200,200)', zerolinecolor: 'rgb(0,255,0)' },
                zaxis: { title: 'Z', gridcolor: 'rgb(200,200,200)', zerolinecolor: 'rgb(0,0,255)' },
                bgcolor: 'rgb(240,240,240)',
                camera: {
                    eye: { x: 1.5, y: 1.5, z: 1.2 }
                }
            },
            margin: { l: 0, r: 0, b: 0, t: 0 },
            showlegend: true,
            legend: {
                x: 0.02,
                y: 0.98,
                bgcolor: 'rgba(255,255,255,0.9)'
            }
        };
        
        var config = {
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToAdd: ['resetCameraDefault3d'],
            toImageButtonOptions: {
                format: 'png',
                filename: 'trajectory',
                height: 1080,
                width: 1920,
                scale: 2
            }
        };
        
        console.log('Data:', trace);
        console.log('Start:', startPoint);
        console.log('End:', endPoint);
        
        Plotly.newPlot('plot', [trace, startPoint, endPoint], layout, config)
            .then(function() {
                console.log('Plotly loaded successfully!');
            })
            .catch(function(err) {
                console.error('Plotly error:', err);
            });
    </script>
</body>
</html>)";
}

void visualize_results(const HookeJeevesResult3D& result_basic,
                       const HookeJeevesResult3D& result_optimized,
                       const string& func_name) {
    string out_dir = get_output_dir();
    
    string html_basic = out_dir + "/" + func_name + "_basic.html";
    string html_optimized = out_dir + "/" + func_name + "_optimized.html";
    
    create_html_plotly(result_basic, html_basic, func_name, "Базовый метод");
    create_html_plotly(result_optimized, html_optimized, func_name, "Оптимизированный метод");
}

void run_test_3d(const string& name, double (*func)(Point3D), Point3D x0,
                 double delta = 0.5, double eps = 1e-6, double alpha = 2.0) {
    cout << "\n========================================" << endl;
    cout << "Функция: " << name << endl;
    cout << "Начальная точка: (" << get_x(x0) << ", " << get_y(x0) << ", " << get_z(x0) << ")" << endl;
    cout << "========================================" << endl;
    
    auto result_basic = hooke_jeeves_3d(x0, func, delta, eps, alpha);
    auto result_optimized = hooke_jeeves_optimized_3d(x0, func, delta, eps, alpha);
    
    cout << "\n--- Базовый метод ---" << endl;
    cout << "Итераций: " << result_basic.iterations << endl;
    cout << "Вычислений функции: " << result_basic.func_evals << endl;
    cout << "Минимум: (" << fixed << setprecision(6)
         << get_x(result_basic.optimum) << ", " << get_y(result_basic.optimum)
         << ", " << get_z(result_basic.optimum) << "), f = " << result_basic.f_opt << endl;
    cout << "Точек в траектории: " << result_basic.trajectory.size() << endl;
    
    cout << "\n--- Оптимизированный метод ---" << endl;
    cout << "Итераций: " << result_optimized.iterations << endl;
    cout << "Вычислений функции: " << result_optimized.func_evals << endl;
    cout << "Минимум: (" << get_x(result_optimized.optimum)
         << ", " << get_y(result_optimized.optimum)
         << ", " << get_z(result_optimized.optimum) << "), f = " << result_optimized.f_opt << endl;
    cout << "Точек в траектории: " << result_optimized.trajectory.size() << endl;
    
    visualize_results(result_basic, result_optimized, name);
}

int main(int argc, const char* argv[]) {
    cout << "Оптимизация методом Хука-Дживса в 3D" << endl;
    
    run_test_3d("Schwefel", schwefel_3d, make_point(450.0, 450.0, 450.0), 2.0);
    run_test_3d("Rastrigin", rastrigin_3d, make_point(4.5, 4.5, 4.5));
    run_test_3d("Rosenbrock", rosenbrock_3d, make_point(2.0, 2.0, 2.0));
    
    return 0;
}
