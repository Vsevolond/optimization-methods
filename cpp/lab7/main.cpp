#include <iostream>
#include <tuple>
#include <vector>
#include <cmath>
#include <functional>
#include <string>
#include <iomanip>
#include <limits>
#include <deque>
#include <sciplot/sciplot.hpp>

using namespace sciplot;

double rosenbrock(std::tuple<double, double> x) {
    double x1 = std::get<0>(x);
    double x2 = std::get<1>(x);
    return 100 * pow(x2 - x1 * x1, 2) + pow(1 - x1, 2);
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

std::vector<double> coordinate_descent_nd(
    const std::function<double(const std::vector<double>&)>& g,
    const std::vector<double>& lower_bounds,
    const std::vector<double>& upper_bounds,
    double tol = 1e-6,
    int max_iter = 1000)
{
    auto n = lower_bounds.size();
    std::vector<double> lambda(n, 0.0);
    double prev_val = g(lambda);
    for (int iter = 0; iter < max_iter; ++iter) {
        for (int i = 0; i < n; ++i) {
            auto g_i = [&](double l) {
                auto lam_copy = lambda;
                lam_copy[i] = l;
                return g(lam_copy);
            };
            double opt = golden_section(g_i, lower_bounds[i], upper_bounds[i], tol);
            lambda[i] = opt;
        }
        double cur_val = g(lambda);
        if (std::abs(cur_val - prev_val) < tol) break;
        prev_val = cur_val;
    }
    return lambda;
}

struct MileContrellResult {
    std::vector<std::tuple<double, double>> trajectory;
    std::tuple<double, double> optimum;
    double f_opt;
    int iterations;
};

MileContrellResult mile_contrell_with_m(
    std::tuple<double, double> x0,
    double (*func)(std::tuple<double, double>),
    int m,
    double eps = 1e-6,
    int max_iter = 1000,
    double lambda_tol = 1e-5)
{
    MileContrellResult result;
    result.trajectory.push_back(x0);

    std::tuple<double, double> x_cur = x0;
    std::deque<std::tuple<double, double>> directions;
    int k = 0;
    const int n = 2;

    int dim_lambda = m + 1;
    std::vector<double> lb(dim_lambda, -100.0);
    std::vector<double> ub(dim_lambda, 100.0);
    lb[0] = 0.0;

    while (k < max_iter) {
        auto grad = gradient(func, x_cur);
        double norm_grad = std::hypot(get_x1(grad), get_x2(grad));
        if (norm_grad < eps) break;

        if (k > 0 && (k % (n + 1) == 0)) {
            directions.clear();
        }

        auto g_func = [&](const std::vector<double>& lam) {
            double x1 = get_x1(x_cur) - lam[0] * get_x1(grad);
            double x2 = get_x2(x_cur) - lam[0] * get_x2(grad);
            for (size_t i = 1; i < lam.size(); ++i) {
                if (i-1 < directions.size()) {
                    x1 += lam[i] * get_x1(directions[i-1]);
                    x2 += lam[i] * get_x2(directions[i-1]);
                }
            }
            return func(make_point(x1, x2));
        };

        auto lambda_opt = coordinate_descent_nd(g_func, lb, ub, lambda_tol, 1000);

        double x1_new = get_x1(x_cur) - lambda_opt[0] * get_x1(grad);
        double x2_new = get_x2(x_cur) - lambda_opt[0] * get_x2(grad);
        for (size_t i = 1; i < lambda_opt.size(); ++i) {
            if (i-1 < directions.size()) {
                x1_new += lambda_opt[i] * get_x1(directions[i-1]);
                x2_new += lambda_opt[i] * get_x2(directions[i-1]);
            }
        }
        auto x_new = make_point(x1_new, x2_new);

        auto delta = make_point(x1_new - get_x1(x_cur), x2_new - get_x2(x_cur));
        directions.push_back(delta);
        if (directions.size() > static_cast<size_t>(m)) {
            directions.pop_front();
        }

        x_cur = x_new;
        result.trajectory.push_back(x_cur);
        ++k;

        if (k % 100 == 0) {
            std::cout << "  iter " << k << ": f = " << func(x_cur) << " at ("
                      << get_x1(x_cur) << ", " << get_x2(x_cur) << ")" << std::endl;
        }
    }

    result.optimum = x_cur;
    result.f_opt = func(x_cur);
    result.iterations = k;
    return result;
}

void visualize_rosenbrock_comparison(
    const std::vector<int>& m_values,
    const std::vector<MileContrellResult>& results,
    double x_min, double x_max, double y_min, double y_max)
{
    std::vector<Plot2D> plots;
    for (size_t k = 0; k < m_values.size(); ++k) {
        int m = m_values[k];
        const auto& res = results[k];

        std::vector<double> tx, ty;
        for (const auto& p : res.trajectory) {
            tx.push_back(get_x1(p));
            ty.push_back(get_x2(p));
        }
        auto start = res.trajectory.front();
        auto end = res.trajectory.back();

        Plot2D plot;
        plot.xlabel("x1");
        plot.ylabel("x2");
        plot.xrange(x_min, x_max);
        plot.yrange(y_min, y_max);
        
        plot.drawCurve(tx, ty)
            .label("m = " + std::to_string(m))
            .lineColor("green")
            .lineWidth(2);
        
        plot.drawPoints(std::vector<double>{get_x1(start)}, std::vector<double>{get_x2(start)})
            .label("start")
            .lineColor("blue")
            .pointType(7)
            .pointSize(3);
        
        plot.drawPoints(std::vector<double>{get_x1(end)}, std::vector<double>{get_x2(end)})
            .label("end")
            .lineColor("red")
            .pointType(7)
            .pointSize(3);

        plots.push_back(plot);
    }

    Figure fig = { { plots[0], plots[1], plots[2] },
                   { plots[3], plots[4], plots[5] } };
    fig.title("Траектории метода Миля–Контрелла на функции Розенброка");
    Canvas canvas = {{fig}};
    canvas.size(1800, 1200);
    canvas.show();

    std::vector<double> m_dbl(m_values.begin(), m_values.end());
    std::vector<double> iter_vals;
    for (const auto& res : results) iter_vals.push_back(res.iterations);

    Plot2D iterPlot;
    iterPlot.xlabel("m (количество предыдущих направлений)");
    iterPlot.ylabel("Число итераций");
    iterPlot.drawPoints(m_dbl, iter_vals)
        .label("Итерации")
        .lineColor("blue")
        .pointType(7)
        .pointSize(3);
    iterPlot.drawCurve(m_dbl, iter_vals)
        .label("Линия тренда")
        .lineColor("red")
        .lineWidth(2);

    Figure iterFig = {{iterPlot}};
    iterFig.title("Зависимость числа итераций от количества направлений");
    Canvas iterCanvas = {{iterFig}};
    iterCanvas.size(800, 600);
    iterCanvas.show();
}

void run_m_comparison() {
    std::cout << "\n========== Сравнение по m на функции Розенброка ==========" << std::endl;

    std::tuple<double, double> x0 = make_point(2.0, 2.0);
    std::vector<int> m_values = {2, 3, 5, 7, 10, 100};
    std::vector<MileContrellResult> results;

    for (int m : m_values) {
        std::cout << "Запуск с m = " << m << " ..." << std::endl;
        auto res = mile_contrell_with_m(x0, rosenbrock, m, 1e-6, 1000, 1e-5);
        results.push_back(res);
        std::cout << "  Итераций: " << res.iterations << ", минимум: ("
                  << get_x1(res.optimum) << ", " << get_x2(res.optimum) << "), f = "
                  << res.f_opt << std::endl;
    }

    double x_min = -2.0, x_max = 2.0, y_min = -1.0, y_max = 3.0;
    visualize_rosenbrock_comparison(m_values, results, x_min, x_max, y_min, y_max);
}

int main() {
    std::cout << "Метод Миля–Контрелла (обобщённый)" << std::endl;
    std::cout << "===================================" << std::endl;

    run_m_comparison();

    return EXIT_SUCCESS;
}
