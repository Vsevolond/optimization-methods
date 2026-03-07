#include <iostream>
#include <tuple>
#include <vector>
#include <cmath>
#include <functional>
#include <string>
#include <iomanip>
#include <algorithm>
#include <sciplot/sciplot.hpp>

using namespace sciplot;

double rastrigin(std::tuple<double, double> x) {
    double x1 = std::get<0>(x);
    double x2 = std::get<1>(x);
    const double A = 10.0;
    return A * 2 + (x1*x1 - A * cos(2 * M_PI * x1)) + (x2*x2 - A * cos(2 * M_PI * x2));
}

double rosenbrock(std::tuple<double, double> x) {
    double x1 = std::get<0>(x);
    double x2 = std::get<1>(x);
    return 100 * pow(x2 - x1*x1, 2) + pow(1 - x1, 2);
}

double schwefel(std::tuple<double, double> x) {
    double x1 = std::get<0>(x);
    double x2 = std::get<1>(x);
    const double A = 418.9829;
    return A * 2 - x1 * sin(sqrt(fabs(x1))) - x2 * sin(sqrt(fabs(x2)));
}

double get_x1(const std::tuple<double, double>& x) { return std::get<0>(x); }
double get_x2(const std::tuple<double, double>& x) { return std::get<1>(x); }

std::tuple<double, double> make_point(double x1, double x2) {
    return std::make_tuple(x1, x2);
}

double distance(const std::tuple<double, double>& a, const std::tuple<double, double>& b) {
    double dx = get_x1(a) - get_x1(b);
    double dy = get_x2(a) - get_x2(b);
    return sqrt(dx*dx + dy*dy);
}

struct RegularSimplex {
    std::tuple<double, double> center;
    double side;
    double angle;
    
    std::vector<std::tuple<double, double>> get_vertices() const {
        std::vector<std::tuple<double, double>> v;
        double r = side / sqrt(3.0);
        
        for (int i = 0; i < 3; ++i) {
            double theta = angle + i * 2.0 * M_PI / 3.0;
            double x = get_x1(center) + r * cos(theta);
            double y = get_x2(center) + r * sin(theta);
            v.push_back(make_point(x, y));
        }
        return v;
    }
    
    std::vector<double> evaluate(double (*func)(std::tuple<double, double>)) const {
        auto v = get_vertices();
        std::vector<double> values;
        for (const auto& p : v) {
            values.push_back(func(p));
        }
        return values;
    }
    
    auto best_index(double (*func)(std::tuple<double, double>)) const {
        auto values = evaluate(func);
        return std::distance(values.begin(),
               std::min_element(values.begin(), values.end()));
    }
    
    auto worst_index(double (*func)(std::tuple<double, double>)) const {
        auto values = evaluate(func);
        return std::distance(values.begin(),
               std::max_element(values.begin(), values.end()));
    }
    
    std::tuple<double, double> best_vertex(double (*func)(std::tuple<double, double>)) const {
        auto v = get_vertices();
        return v[best_index(func)];
    }
    
    std::tuple<double, double> worst_vertex(double (*func)(std::tuple<double, double>)) const {
        auto v = get_vertices();
        return v[worst_index(func)];
    }
    
    double size() const {
        return side;
    }
};

struct SimplexResult {
    std::vector<RegularSimplex> simplexes;
    std::vector<std::tuple<double, double>> trajectory;
    std::tuple<double, double> optimum;
    double f_opt;
    int iterations;
};

SimplexResult regular_simplex_method(
    std::tuple<double, double> x0,
    double (*func)(std::tuple<double, double>),
    double initial_side = 1.0,
    double eps = 1e-6,
    double alpha = 1.0,
    double beta = 0.5
) {
    SimplexResult result;
    
    RegularSimplex current;
    current.center = x0;
    current.side = initial_side;
    current.angle = 0.0;
    
    result.simplexes.push_back(current);
    result.trajectory.push_back(current.best_vertex(func));
    
    int iter = 0;
    const int MAX_ITER = 1000;
    
    while (current.side > eps && iter < MAX_ITER) {
        auto vertices = current.get_vertices();
        auto values = current.evaluate(func);
        auto best = current.best_index(func);
        auto worst = current.worst_index(func);
        
        double dx = get_x1(vertices[best]) - get_x1(vertices[worst]);
        double dy = get_x2(vertices[best]) - get_x2(vertices[worst]);
        double len = sqrt(dx*dx + dy*dy);
        
        if (len > 1e-10) {
            dx /= len;
            dy /= len;
        }
        
        RegularSimplex reflected;
        reflected.side = current.side;
        
        auto middle = 3 - best - worst;
        auto mid_point = make_point(
            (get_x1(vertices[best]) + get_x1(vertices[middle])) / 2.0,
            (get_x2(vertices[best]) + get_x2(vertices[middle])) / 2.0
        );
        
        reflected.center = make_point(
            2 * get_x1(mid_point) - get_x1(current.center),
            2 * get_x2(mid_point) - get_x2(current.center)
        );
        
        reflected.angle = current.angle + M_PI;
        
        auto reflected_values = reflected.evaluate(func);
        double current_best = values[best];
        double reflected_best = *std::min_element(reflected_values.begin(), reflected_values.end());
        
        bool improved = false;
        
        if (reflected_best < current_best) {
            current = reflected;
            improved = true;
            
        } else {
            for (int dir = -1; dir <= 1; dir += 2) {
                RegularSimplex rotated;
                rotated.side = current.side;
                
                double angle_step = dir * M_PI / 3.0;
                rotated.angle = current.angle + angle_step;
                
                double cos_a = cos(angle_step);
                double sin_a = sin(angle_step);
                
                double vx = get_x1(vertices[best]);
                double vy = get_x2(vertices[best]);
                double cx = get_x1(current.center);
                double cy = get_x2(current.center);
                
                double new_cx = vx + (cx - vx) * cos_a - (cy - vy) * sin_a;
                double new_cy = vy + (cx - vx) * sin_a + (cy - vy) * cos_a;
                
                rotated.center = make_point(new_cx, new_cy);
                
                auto rotated_values = rotated.evaluate(func);
                double rotated_best = *std::min_element(rotated_values.begin(), rotated_values.end());
                
                if (rotated_best < current_best) {
                    current = rotated;
                    improved = true;
                    break;
                }
            }
        }
        
        if (!improved) {
            auto best_v = vertices[best];
            current.side *= beta;
            
            current.center = make_point(
                get_x1(best_v) + beta * (get_x1(current.center) - get_x1(best_v)),
                get_x2(best_v) + beta * (get_x2(current.center) - get_x2(best_v))
            );
        }
        
        result.simplexes.push_back(current);
        result.trajectory.push_back(current.best_vertex(func));
        
        iter++;
    }
    
    auto final_vertices = current.get_vertices();
    auto final_values = current.evaluate(func);
    auto final_best = current.best_index(func);
    result.optimum = final_vertices[final_best];
    result.f_opt = final_values[final_best];
    result.iterations = iter;
    
    return result;
}

void visualize_regular_simplex(
    const SimplexResult& result,
    double (*func)(std::tuple<double, double>),
    const std::string& func_name,
    double x_min, double x_max, double y_min, double y_max
) {
    const int n = 50;
    std::vector<double> x_grid, y_grid, z_grid;
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double x = x_min + (x_max - x_min) * i / (n - 1);
            double y = y_min + (y_max - y_min) * j / (n - 1);
            x_grid.push_back(x);
            y_grid.push_back(y);
            z_grid.push_back(func(make_point(x, y)));
        }
    }
    
    std::vector<double> tx, ty, tz;
    for (const auto& p : result.trajectory) {
        tx.push_back(get_x1(p));
        ty.push_back(get_x2(p));
        tz.push_back(func(p));
    }
    
    Plot3D plot3d;
    plot3d.xlabel("x1");
    plot3d.ylabel("x2");
    plot3d.zlabel("f(x)");
    
    plot3d.drawWithVecs("lines", x_grid, y_grid, z_grid)
        .label("function")
        .lineColor("gray")
        .lineWidth(1);
    
    plot3d.drawCurve(tx, ty, tz)
        .label("trajectory (best vertices)")
        .lineColor("green")
        .lineWidth(3);
    
    plot3d.drawPoints(tx, ty, tz)
        .label("iteration points")
        .lineColor("green")
        .pointType(7)
        .pointSize(2);
    
    Plot2D plot2d;
    plot2d.xlabel("x1");
    plot2d.ylabel("x2");
    
    for (size_t s = 0; s < result.simplexes.size(); ++s) {
        const auto& simp = result.simplexes[s];
        auto vertices = simp.get_vertices();
        auto values = simp.evaluate(func);
        
        auto worst = std::distance(values.begin(),
                    std::max_element(values.begin(), values.end()));
        
        for (int i = 0; i < 3; ++i) {
            int j = (i + 1) % 3;
            std::vector<double> edge_x = {get_x1(vertices[i]), get_x1(vertices[j])};
            std::vector<double> edge_y = {get_x2(vertices[i]), get_x2(vertices[j])};
            
            plot2d.drawCurve(edge_x, edge_y)
                .lineColor("black")
                .lineWidth(1)
                .label(s == 0 && i == 0 ? "simplex edges" : "");
        }
        
        for (int i = 0; i < 3; ++i) {
            std::vector<double> px = {get_x1(vertices[i])};
            std::vector<double> py = {get_x2(vertices[i])};
            
            if (i == worst) {
                plot2d.drawPoints(px, py)
                    .lineColor("red")
                    .pointType(7)
                    .pointSize(2)
                    .label(s == 0 ? "worst vertex (max f)" : "");
            } else {
                plot2d.drawPoints(px, py)
                    .lineColor("blue")
                    .pointType(7)
                    .pointSize(1.5)
                    .label(s == 0 && i == (worst+1)%3 ? "other vertices" : "");
            }
        }
    }
    
    plot2d.drawCurve(tx, ty)
        .label("trajectory")
        .lineColor("green")
        .lineWidth(2);
    
    plot2d.drawPoints(tx, ty)
        .label("best points")
        .lineColor("green")
        .pointType(5)
        .pointSize(2);
    
    Figure fig = {{plot3d, plot2d}};
    fig.title("Regular Simplex Method for " + func_name);
    
    Canvas canvas = {{fig}};
    canvas.size(1200, 600);
    canvas.show();
}

void run_regular_simplex_test(
    const std::string& name,
    double (*func)(std::tuple<double, double>),
    std::tuple<double, double> x0,
    double x_min, double x_max, double y_min, double y_max
) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Функция: " << name << std::endl;
    std::cout << "Начальная точка: (" << get_x1(x0) << ", " << get_x2(x0) << ")" << std::endl;
    std::cout << "========================================" << std::endl;
    
    auto result = regular_simplex_method(x0, func, 1.0, 1e-6);
    
    std::cout << "Итераций: " << result.iterations << std::endl;
    std::cout << "Найденный минимум: (" << std::fixed << std::setprecision(6)
              << get_x1(result.optimum) << ", " << get_x2(result.optimum) << ")" << std::endl;
    std::cout << "Значение функции: " << result.f_opt << std::endl;
    std::cout << "Всего симплексов: " << result.simplexes.size() << std::endl;
    
    std::cout << "\n--- Первые 10 точек траектории (лучшие вершины) ---" << std::endl;
    for (size_t i = 0; i < std::min(result.trajectory.size(), size_t(10)); ++i) {
        auto p = result.trajectory[i];
        std::cout << "Iter " << i << ": (" << std::setw(10) << get_x1(p)
                  << ", " << std::setw(10) << get_x2(p) << ") f=" << func(p) << std::endl;
    }
    
    visualize_regular_simplex(result, func, name, x_min, x_max, y_min, y_max);
}

int main(int argc, const char* argv[]) {
    std::cout << "Метод простых регулярных симплексов" << std::endl;
    std::cout << "===================================" << std::endl;
    
    run_regular_simplex_test("Rastrigin", rastrigin,
             make_point(4.5, 4.5),
             -5.0, 5.0, -5.0, 5.0);
    
    run_regular_simplex_test("Rosenbrock", rosenbrock,
             make_point(-1.5, 2.5),
             -2.0, 2.0, -1.0, 3.0);
    
    run_regular_simplex_test("Schwefel", schwefel,
             make_point(350.0, 350.0),
             -500.0, 500.0, -500.0, 500.0);
    
    return EXIT_SUCCESS;
}
