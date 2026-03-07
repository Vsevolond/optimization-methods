#include <iostream>
#include <tuple>
#include <vector>
#include <cmath>
#include <functional>
#include <string>
#include <iomanip>
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


std::tuple<double, double> exploratory_search(
    std::tuple<double, double> x_base,
    double (*func)(std::tuple<double, double>),
    double delta,
    std::vector<std::tuple<double, double, bool>>& probes
) {
    double x1 = get_x1(x_base);
    double x2 = get_x2(x_base);
    double f_base = func(x_base);
    
    std::tuple<double, double> x_new = x_base;
    double f_new = f_base;
    
    auto x_test = make_point(x1 + delta, x2);
    double f_test = func(x_test);
    probes.push_back(std::make_tuple(x1 + delta, x2, f_test < f_base));
    if (f_test < f_new) {
        x_new = x_test;
        f_new = f_test;
    }
    
    x_test = make_point(x1 - delta, x2);
    f_test = func(x_test);
    probes.push_back(std::make_tuple(x1 - delta, x2, f_test < f_base));
    if (f_test < f_new) {
        x_new = x_test;
        f_new = f_test;
    }
    
    x_test = make_point(x1, x2 + delta);
    f_test = func(x_test);
    probes.push_back(std::make_tuple(x1, x2 + delta, f_test < f_base));
    if (f_test < f_new) {
        x_new = x_test;
        f_new = f_test;
    }
    
    x_test = make_point(x1, x2 - delta);
    f_test = func(x_test);
    probes.push_back(std::make_tuple(x1, x2 - delta, f_test < f_base));
    if (f_test < f_new) {
        x_new = x_test;
        f_new = f_test;
    }
    
    return x_new;
}


struct HookeJeevesResult {
    std::vector<std::tuple<double, double>> trajectory;
    std::vector<std::vector<std::tuple<double, double, bool>>> all_probes;
    std::tuple<double, double> optimum;
    double f_opt;
    int iterations;
};


HookeJeevesResult hooke_jeeves(
    std::tuple<double, double> x0,
    double (*func)(std::tuple<double, double>),
    double delta = 0.1,
    double eps = 1e-6,
    double alpha = 2.0
) {
    HookeJeevesResult result;
    auto x_base = x0;
    double f_base = func(x_base);
    
    result.trajectory.push_back(x_base);
    
    int iter = 0;
    while (delta > eps) {
        std::vector<std::tuple<double, double, bool>> probes;
        auto x_new = exploratory_search(x_base, func, delta, probes);
        result.all_probes.push_back(probes);
        
        double f_new = func(x_new);
        
        if (f_new < f_base) {
            auto x_pattern = make_point(
                2 * get_x1(x_new) - get_x1(x_base),
                2 * get_x2(x_new) - get_x2(x_base)
            );
            
            std::vector<std::tuple<double, double, bool>> pattern_probes;
            auto x_pattern_result = exploratory_search(x_pattern, func, delta, pattern_probes);
            result.all_probes.push_back(pattern_probes);
            
            double f_pattern = func(x_pattern_result);
            
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


HookeJeevesResult hooke_jeeves_optimized(
    std::tuple<double, double> x0,
    double (*func)(std::tuple<double, double>),
    double delta = 0.1,
    double eps = 1e-6,
    double alpha = 2.0
) {
    HookeJeevesResult result;
    auto x_base = x0;
    double f_base = func(x_base);
    
    result.trajectory.push_back(x_base);
    
    int iter = 0;
    while (delta > eps) {
        std::vector<std::tuple<double, double, bool>> probes;
        auto x_new = exploratory_search(x_base, func, delta, probes);
        result.all_probes.push_back(probes);
        
        double f_new = func(x_new);
        
        if (f_new < f_base) {
            auto x_current = x_new;
            double f_current = f_new;
            
            result.trajectory.push_back(x_current);
            
            while (true) {
                auto x_pattern = make_point(
                    2 * get_x1(x_current) - get_x1(x_base),
                    2 * get_x2(x_current) - get_x2(x_base)
                );
                
                std::vector<std::tuple<double, double, bool>> pattern_probes;
                auto x_pattern_result = exploratory_search(x_pattern, func, delta, pattern_probes);
                result.all_probes.push_back(pattern_probes);
                
                double f_pattern = func(x_pattern_result);
                
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


void visualize_method(
    const HookeJeevesResult& result,
    double (*func)(std::tuple<double, double>),
    const std::string& method_name,
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
    
    std::vector<double> blue_x, blue_y, red_x, red_y;
    for (const auto& probe_set : result.all_probes) {
        for (const auto& probe : probe_set) {
            double px = std::get<0>(probe);
            double py = std::get<1>(probe);
            bool decreases = std::get<2>(probe);
            
            if (decreases) {
                blue_x.push_back(px);
                blue_y.push_back(py);
            } else {
                red_x.push_back(px);
                red_y.push_back(py);
            }
        }
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
        .label("trajectory")
        .lineColor("green")
        .lineWidth(3);
    
    Plot2D plot2d;
    plot2d.xlabel("x1");
    plot2d.ylabel("x2");
    
    plot2d.drawCurve(tx, ty)
        .label("trajectory")
        .lineColor("green")
        .lineWidth(2);
    
    plot2d.drawPoints(tx, ty)
        .label("relaxation points")
        .lineColor("green")
        .pointType(7)
        .pointSize(2);
    
    if (!blue_x.empty()) {
        plot2d.drawPoints(blue_x, blue_y)
            .label("decreasing")
            .lineColor("blue")
            .pointType(1)
            .pointSize(1);
    }
    
    if (!red_x.empty()) {
        plot2d.drawPoints(red_x, red_y)
            .label("increasing")
            .lineColor("red")
            .pointType(1)
            .pointSize(1);
    }
    
    Figure fig = {{plot3d, plot2d}};
    fig.title(method_name + " method for " + func_name);
    
    Canvas canvas = {{fig}};
    canvas.size(1200, 600);
    canvas.show();
}

void visualize_comparison(
    const HookeJeevesResult& result_basic,
    const HookeJeevesResult& result_optimized,
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
    
    auto prepare_data = [&](const HookeJeevesResult& result) {
        std::vector<double> tx, ty, tz, blue_x, blue_y, red_x, red_y;
        
        for (const auto& p : result.trajectory) {
            tx.push_back(get_x1(p));
            ty.push_back(get_x2(p));
            tz.push_back(func(p));
        }
        
        for (const auto& probe_set : result.all_probes) {
            for (const auto& probe : probe_set) {
                double px = std::get<0>(probe);
                double py = std::get<1>(probe);
                bool decreases = std::get<2>(probe);
                
                if (decreases) {
                    blue_x.push_back(px);
                    blue_y.push_back(py);
                } else {
                    red_x.push_back(px);
                    red_y.push_back(py);
                }
            }
        }
        
        return std::make_tuple(tx, ty, tz, blue_x, blue_y, red_x, red_y);
    };
    
    auto [tx1, ty1, tz1, bx1, by1, rx1, ry1] = prepare_data(result_basic);
    auto [tx2, ty2, tz2, bx2, by2, rx2, ry2] = prepare_data(result_optimized);
    
    Plot3D plot3d_basic;
    plot3d_basic.xlabel("x1");
    plot3d_basic.ylabel("x2");
    plot3d_basic.zlabel("f(x)");
    plot3d_basic.drawWithVecs("lines", x_grid, y_grid, z_grid).lineColor("gray").lineWidth(1);
    plot3d_basic.drawCurve(tx1, ty1, tz1).lineColor("green").lineWidth(3).label("trajectory");
    
    Plot2D plot2d_basic;
    plot2d_basic.xlabel("x1");
    plot2d_basic.ylabel("x2");
    plot2d_basic.drawCurve(tx1, ty1).lineColor("green").lineWidth(2);
    plot2d_basic.drawPoints(tx1, ty1).lineColor("green").pointType(7).pointSize(2).label("relaxation");
    if (!bx1.empty()) plot2d_basic.drawPoints(bx1, by1).lineColor("blue").pointType(1).pointSize(1).label("dec");
    if (!rx1.empty()) plot2d_basic.drawPoints(rx1, ry1).lineColor("red").pointType(1).pointSize(1).label("inc");
    
    Plot3D plot3d_opt;
    plot3d_opt.xlabel("x1");
    plot3d_opt.ylabel("x2");
    plot3d_opt.zlabel("f(x)");
    plot3d_opt.drawWithVecs("lines", x_grid, y_grid, z_grid).lineColor("gray").lineWidth(1);
    plot3d_opt.drawCurve(tx2, ty2, tz2).lineColor("green").lineWidth(3).label("trajectory");
    
    Plot2D plot2d_opt;
    plot2d_opt.xlabel("x1");
    plot2d_opt.ylabel("x2");
    plot2d_opt.drawCurve(tx2, ty2).lineColor("green").lineWidth(2);
    plot2d_opt.drawPoints(tx2, ty2).lineColor("green").pointType(7).pointSize(2).label("relaxation");
    
    if (!bx2.empty()) plot2d_opt.drawPoints(bx2, by2).lineColor("blue").pointType(1).pointSize(1).label("dec");
    
    if (!rx2.empty()) plot2d_opt.drawPoints(rx2, ry2).lineColor("red").pointType(1).pointSize(1).label("inc");
    
    Figure fig = {{plot3d_basic, plot3d_opt}, {plot2d_basic, plot2d_opt}};
    fig.title("Comparison for " + func_name);
    
    Canvas canvas = {{fig}};
    canvas.size(1200, 900);
    canvas.show();
}

void run_test(
    const std::string& name,
    double (*func)(std::tuple<double, double>),
    std::tuple<double, double> x0,
    double x_min, double x_max, double y_min, double y_max
) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Функция: " << name << std::endl;
    std::cout << "Начальная точка: (" << get_x1(x0) << ", " << get_x2(x0) << ")" << std::endl;
    std::cout << "========================================" << std::endl;
    
    auto result_basic = hooke_jeeves(x0, func, 0.5, 1e-6);
    auto result_optimized = hooke_jeeves_optimized(x0, func, 0.5, 1e-6);
    
    std::cout << "\n--- Базовый метод (один поиск по образцу) ---" << std::endl;
    std::cout << "Итераций: " << result_basic.iterations << std::endl;
    std::cout << "Найденный минимум: (" << std::fixed << std::setprecision(6) << get_x1(result_basic.optimum)
              << ", " << get_x2(result_basic.optimum) << ")" << std::endl;
    std::cout << "Значение функции: " << result_basic.f_opt << std::endl;
    std::cout << "Точек в траектории: " << result_basic.trajectory.size() << std::endl;
    
    std::cout << "\n--- Оптимизированный метод (повторяющийся поиск по образцу) ---" << std::endl;
    std::cout << "Итераций: " << result_optimized.iterations << std::endl;
    std::cout << "Найденный минимум: (" << get_x1(result_optimized.optimum)
              << ", " << get_x2(result_optimized.optimum) << ")" << std::endl;
    std::cout << "Значение функции: " << result_optimized.f_opt << std::endl;
    std::cout << "Точек в траектории: " << result_optimized.trajectory.size() << std::endl;
    
    std::cout << "\n--- Сравнение точек релаксационной последовательности ---" << std::endl;
    size_t max_points = std::max(result_basic.trajectory.size(), result_optimized.trajectory.size());
    
    std::cout << "Шаг | Базовый метод (x1, x2)       | Оптимизированный метод (x1, x2)" << std::endl;
    std::cout << "----|------------------------------|--------------------------------" << std::endl;
    
    for (size_t i = 0; i < max_points; ++i) {
        std::cout << std::setw(3) << i << " | ";
        if (i < result_basic.trajectory.size()) {
            auto p = result_basic.trajectory[i];
            std::cout << "(" << std::setw(10) << std::fixed << std::setprecision(6) << get_x1(p)
                      << ", " << std::setw(10) << get_x2(p) << ")";
        } else {
            std::cout << "                              ";
        }
        std::cout << " | ";
        if (i < result_optimized.trajectory.size()) {
            auto p = result_optimized.trajectory[i];
            std::cout << "(" << std::setw(10) << get_x1(p) << ", " << std::setw(10) << get_x2(p) << ")";
        }
        std::cout << std::endl;
    }
    
    visualize_comparison(result_basic, result_optimized, func, name, x_min, x_max, y_min, y_max);
}

int main(int argc, const char* argv[]) {
    std::cout << "Оптимизация методом Хука-Дживса" << std::endl;
    std::cout << "================================" << std::endl;
    
    run_test("Rastrigin", rastrigin,
             make_point(4.5, 4.5),
             -5.0, 5.0, -5.0, 5.0);
    
    run_test("Rosenbrock", rosenbrock,
             make_point(-1.5, 2.5),
             -2.0, 2.0, -1.0, 3.0);
    
    run_test("Schwefel", schwefel,
             make_point(350.0, 350.0),
             -500.0, 500.0, -500.0, 500.0);
    
    return EXIT_SUCCESS;
}
