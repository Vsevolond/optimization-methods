#include <iostream>
#include <tuple>
#include <vector>
#include <cmath>
#include <functional>
#include <string>
#include <iomanip>
#include <limits>
#include <sciplot/sciplot.hpp>

using namespace sciplot;

double rastrigin(std::tuple<double, double> x) {
    double x1 = std::get<0>(x);
    double x2 = std::get<1>(x);
    const double A = 10.0;
    return A * 2 + (x1 * x1 - A * cos(2 * M_PI * x1)) + (x2 * x2 - A * cos(2 * M_PI * x2));
}

double rosenbrock(std::tuple<double, double> x) {
    double x1 = std::get<0>(x);
    double x2 = std::get<1>(x);
    return 100 * pow(x2 - x1 * x1, 2) + pow(1 - x1, 2);
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

std::tuple<double, double> gradient(double (*func)(std::tuple<double, double>),
                                    std::tuple<double, double> x,
                                    double h = 1e-6) {
    double x1 = get_x1(x);
    double x2 = get_x2(x);
    double f0 = func(x);
    double dx1 = (func(make_point(x1 + h, x2)) - f0) / h;
    double dx2 = (func(make_point(x1, x2 + h)) - f0) / h;
    return make_point(dx1, dx2);
}

double golden_section(const std::function<double(double)>& g,
                      double a,
                      double b,
                      double tol = 1e-6) {
    const double phi = (1.0 + sqrt(5.0)) / 2.0;
    double c = b - (b - a) / phi;
    double d = a + (b - a) / phi;

    while (fabs(c - d) > tol) {
        if (g(c) < g(d)) {
            b = d;
        } else {
            a = c;
        }
        c = b - (b - a) / phi;
        d = a + (b - a) / phi;
    }
    return (a + b) / 2.0;
}

std::tuple<double, double> coordinate_descent_2d(
    const std::function<double(double, double)>& g,
    double tol = 1e-6,
    int max_iter = 100)
{
    double lam0 = 0.0, lam1 = 0.0;
    double prev_f = g(lam0, lam1);
    
    for (int iter = 0; iter < max_iter; ++iter) {
        auto g0 = [&](double l0) { return g(l0, lam1); };
        lam0 = golden_section(g0, -10.0, 10.0, tol);
        
        auto g1 = [&](double l1) { return g(lam0, l1); };
        lam1 = golden_section(g1, -10.0, 10.0, tol);
        
        double cur_f = g(lam0, lam1);
        if (fabs(cur_f - prev_f) < tol) break;
        prev_f = cur_f;
    }
    return make_point(lam0, lam1);
}

struct MileContrellResult {
    std::vector<std::tuple<double, double>> trajectory;
    std::tuple<double, double> optimum;
    double f_opt;
    int iterations;
};

MileContrellResult mile_contrell(
    std::tuple<double, double> x0,
    double (*func)(std::tuple<double, double>),
    double eps = 1e-6,
    int max_iter = 1000,
    double lambda_tol = 1e-6)
{
    MileContrellResult result;
    result.trajectory.push_back(x0);

    std::tuple<double, double> x_cur = x0;
    std::tuple<double, double> x_prev = x0;
    std::tuple<double, double> delta_x = make_point(0.0, 0.0);
    int k = 0;
    const int n = 2;

    while (k < max_iter) {
        auto grad = gradient(func, x_cur);
        double norm_grad = std::hypot(get_x1(grad), get_x2(grad));
        if (norm_grad < eps) break;

        if (k > 0 && (k % (n + 1) == 0)) {
            delta_x = make_point(0.0, 0.0);
        }

        bool use_delta = (std::hypot(get_x1(delta_x), get_x2(delta_x)) > 1e-12);

        auto g = [&](double lam0, double lam1) {
            double x1 = get_x1(x_cur) - lam0 * get_x1(grad) + lam1 * get_x1(delta_x);
            double x2 = get_x2(x_cur) - lam0 * get_x2(grad) + lam1 * get_x2(delta_x);
            return func(make_point(x1, x2));
        };

        double lam0_opt, lam1_opt;
        if (!use_delta) {
            auto g0 = [&](double lam0) { return g(lam0, 0.0); };
            lam0_opt = golden_section(g0, 0.0, 10.0, lambda_tol);
            lam1_opt = 0.0;
        } else {
            auto lambda_opt = coordinate_descent_2d(g, lambda_tol);
            lam0_opt = get_x1(lambda_opt);
            lam1_opt = get_x2(lambda_opt);
        }

        double x1_new = get_x1(x_cur) - lam0_opt * get_x1(grad) + lam1_opt * get_x1(delta_x);
        double x2_new = get_x2(x_cur) - lam0_opt * get_x2(grad) + lam1_opt * get_x2(delta_x);
        auto x_new = make_point(x1_new, x2_new);

        delta_x = make_point(
            get_x1(x_new) - get_x1(x_cur),
            get_x2(x_new) - get_x2(x_cur)
        );

        x_prev = x_cur;
        x_cur = x_new;
        result.trajectory.push_back(x_cur);
        ++k;
    }

    result.optimum = x_cur;
    result.f_opt = func(x_cur);
    result.iterations = k;
    return result;
}

void visualize_mile_contrell(
    const MileContrellResult& result,
    double (*func)(std::tuple<double, double>),
    const std::string& func_name,
    double x_min, double x_max, double y_min, double y_max)
{
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

    auto start = result.trajectory[0];
    auto end = result.trajectory.back();

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
    
    plot3d.drawPoints(std::vector<double>{get_x1(start)},
                      std::vector<double>{get_x2(start)},
                      std::vector<double>{func(start)})
        .label("start")
        .lineColor("blue")
        .pointType(7)
        .pointSize(3);
    
    plot3d.drawPoints(std::vector<double>{get_x1(end)},
                      std::vector<double>{get_x2(end)},
                      std::vector<double>{func(end)})
        .label("end")
        .lineColor("red")
        .pointType(7)
        .pointSize(3);

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
    
    plot2d.drawPoints(std::vector<double>{get_x1(start)},
                      std::vector<double>{get_x2(start)})
        .label("start")
        .lineColor("blue")
        .pointType(7)
        .pointSize(3);
    
    plot2d.drawPoints(std::vector<double>{get_x1(end)},
                      std::vector<double>{get_x2(end)})
        .label("end")
        .lineColor("red")
        .pointType(7)
        .pointSize(3);

    Figure fig = {{plot3d, plot2d}};
    fig.title("Mile–Contrell method for " + func_name);

    Canvas canvas = {{fig}};
    canvas.size(1200, 600);
    canvas.show();
}

void run_test_mile_contrell(
    const std::string& name,
    double (*func)(std::tuple<double, double>),
    std::tuple<double, double> x0,
    double x_min, double x_max, double y_min, double y_max)
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "Функция: " << name << std::endl;
    std::cout << "Начальная точка: (" << get_x1(x0) << ", " << get_x2(x0) << ")" << std::endl;
    std::cout << "========================================" << std::endl;

    auto result = mile_contrell(x0, func, 1e-6, 1000, 1e-6);

    std::cout << "\n--- Метод Миля–Контрелла ---" << std::endl;
    std::cout << "Итераций: " << result.iterations << std::endl;
    std::cout << "Найденный минимум: (" << std::fixed << std::setprecision(6)
              << get_x1(result.optimum) << ", " << get_x2(result.optimum) << ")" << std::endl;
    std::cout << "Значение функции: " << result.f_opt << std::endl;
    std::cout << "Точек в траектории: " << result.trajectory.size() << std::endl;

//    std::cout << "\n--- Релаксационная последовательность ---" << std::endl;
//    for (size_t i = 0; i < result.trajectory.size(); ++i) {
//        auto p = result.trajectory[i];
//        std::cout << "Шаг " << std::setw(3) << i << ": ("
//                  << std::setw(10) << get_x1(p) << ", "
//                  << std::setw(10) << get_x2(p) << ")" << std::endl;
//    }

    visualize_mile_contrell(result, func, name, x_min, x_max, y_min, y_max);
}

int main() {
    run_test_mile_contrell("Rastrigin", rastrigin,
                           make_point(0.5, 0.5),
                           -5.0, 5.0, -5.0, 5.0);
    
    run_test_mile_contrell("Rosenbrock", rosenbrock,
                           make_point(2.0, 2.0),
                           -2.0, 2.0, -1.0, 3.0);

    run_test_mile_contrell("Schwefel", schwefel,
                           make_point(360.0, 360.0),
                           -500.0, 500.0, -500.0, 500.0);

    return EXIT_SUCCESS;
}
