#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <iomanip>
#include <algorithm>
#include <sciplot/sciplot.hpp>

using namespace sciplot;

// Целевая функция f(x) = |x^3 - 3x|
double f(double x) {
    return std::abs(x * x * x - 3.0 * x);
}

double df(const std::function<double(double)>& func, double x, double h = 1e-5) {
    return (func(x + h) - func(x - h)) / (2.0 * h);
}

double d2f(const std::function<double(double)>& func, double x, double h = 1e-4) {
    return (func(x + h) - 2.0 * func(x) + func(x - h)) / (h * h);
}

struct Parabola {
    double x1, x2, x3;
    double f1, f2, f3;
    double a, b, c;
    double x_min;
    double f_min;
};

struct MethodResult {
    std::string name;
    double x_min;
    double f_min;
    int iterations;
    std::vector<double> x_history;
    std::vector<double> f_history;
    std::string color;
    double a, b;
    bool success;
    std::vector<Parabola> parabolas;
};

MethodResult sven_method(const std::function<double(double)>& func, double x0,
                         double h0 = 0.01, int max_iter = 50) {
    MethodResult result;
    result.name = "Свенн";
    result.color = "blue";
    result.success = false;
    result.x_history.push_back(x0);
    result.f_history.push_back(func(x0));
    
    double f0 = func(x0);
    double f1 = func(x0 + h0);
    double h = h0;
    
    if (f1 > f0) {
        h = -h;
        f1 = func(x0 + h);
    }
    
    if (f1 > f0) {
        double f_left = func(x0 - h0);
        double f_right = func(x0 + h0);
        
        if (f_left > f0 && f_right > f0) {
            result.a = x0 - std::abs(h0);
            result.b = x0 + std::abs(h0);
            result.x_min = x0;
            result.f_min = f0;
            result.iterations = 1;
            result.success = true;
            return result;
        } else {
            h = (f_left < f_right) ? -h0 : h0;
            f1 = func(x0 + h);
        }
    }
    
    double x_prev = x0;
    double x_curr = x0 + h;
    double f_prev = f0;
    double f_curr = f1;
    result.x_history.push_back(x_curr);
    result.f_history.push_back(f_curr);
    
    int iter = 1;
    for (int k = 1; k <= max_iter && iter < max_iter; ++k) {
        double h2 = h * std::pow(2.0, k);
        double x_next = x0 + h2;
        double f_next = func(x_next);
        
        result.x_history.push_back(x_next);
        result.f_history.push_back(f_next);
        iter++;
        
        if (f_next >= f_curr) {
            result.a = std::min(x_prev, x_next);
            result.b = std::max(x_prev, x_next);
            if (f_curr < f0) {
                result.success = true;
            }
            result.x_min = x_curr;
            result.f_min = f_curr;
            result.iterations = iter;
            return result;
        }
        
        x_prev = x_curr;
        x_curr = x_next;
        f_prev = f_curr;
        f_curr = f_next;
    }
    
    result.a = std::min(x_prev, x_curr);
    result.b = std::max(x_prev, x_curr);
    result.x_min = x_curr;
    result.f_min = f_curr;
    result.iterations = iter;
    return result;
}

MethodResult fibonacci_method(const std::function<double(double)>& func,
                              double a, double b, double eps = 1e-5) {
    MethodResult result;
    result.name = "Фибоначчи";
    result.color = "red";
    result.a = a;
    result.b = b;
    result.success = false;
    
    std::vector<double> F = {1.0, 1.0};
    while (F.back() < (b - a) / eps) {
        F.push_back(F[F.size() - 1] + F[F.size() - 2]);
    }
    
    auto N = F.size();
    double x1 = a + (F[N - 2] / F[N - 1]) * (b - a);
    double x2 = a + (F[N - 1] / F[N - 1]) * (b - a);
    double f1 = func(x1);
    double f2 = func(x2);
    
    result.x_history.push_back(x1);
    result.f_history.push_back(f1);
    result.x_history.push_back(x2);
    result.f_history.push_back(f2);
    
    int iter = 2;
    for (int k = 1; k <= N - 3; ++k) {
        if (std::abs(b - a) < eps) break;
        
        if (f1 > f2) {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = a + (F[N - k - 1] / F[N - k]) * (b - a);
            f2 = func(x2);
            result.x_history.push_back(x2);
            result.f_history.push_back(f2);
        } else {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = a + (F[N - k - 2] / F[N - k]) * (b - a);
            f1 = func(x1);
            result.x_history.push_back(x1);
            result.f_history.push_back(f1);
        }
        iter++;
    }
    
    result.x_min = (a + b) / 2.0;
    result.f_min = func(result.x_min);
    result.iterations = iter;
    
    double f_at_min = result.f_min;
    double f_left = func(result.x_min - eps);
    double f_right = func(result.x_min + eps);
    
    if (f_left < f_at_min || f_right < f_at_min) {
        result.success = false;
    } else {
        result.success = true;
    }
    
    return result;
}

MethodResult reverse_variable_step_method(const std::function<double(double)>& func,
                                          double x0, double h0 = 0.1,
                                          double eps = 1e-5, int max_iter = 1000) {
    MethodResult result;
    result.name = "Обратный переменный шаг";
    result.color = "green";
    result.success = false;
    
    double x = x0;
    double fx = func(x);
    double h = h0;
    
    result.x_history.push_back(x);
    result.f_history.push_back(fx);
    
    int iter = 0;
    int no_improve_count = 0;
    
    while (iter < max_iter && no_improve_count < 50) {
        double grad = df(func, x);
        
        if (std::abs(grad) < eps) {
            double second_deriv = d2f(func, x);
            if (second_deriv > 0) {
                result.success = true;
                break;
            } else if (second_deriv < 0) {
                h = (rand() % 2 == 0) ? h0 : -h0;
            }
        }
        
        double x_new = x - h * grad;
        double f_new = func(x_new);
        
        if (f_new < fx - eps) {
            x = x_new;
            fx = f_new;
            h *= 1.2;
            result.x_history.push_back(x);
            result.f_history.push_back(fx);
            no_improve_count = 0;
        } else {
            h *= -0.5;
            no_improve_count++;
            if (std::abs(h) < eps) {
                h = h0 * ((rand() % 2 == 0) ? 1 : -1);
            }
        }
        iter++;
    }
    
    result.x_min = x;
    result.f_min = fx;
    result.iterations = int(result.x_history.size());
    result.a = *std::min_element(result.x_history.begin(), result.x_history.end());
    result.b = *std::max_element(result.x_history.begin(), result.x_history.end());
    
    double f_left = func(result.x_min - eps);
    double f_right = func(result.x_min + eps);
    result.success = (fx <= f_left + eps && fx <= f_right + eps);
    
    return result;
}

MethodResult powell_method(const std::function<double(double)>& func,
                           double x0, double h0 = 0.1,
                           double eps = 1e-5, int max_iter = 1000) {
    MethodResult result;
    result.name = "Пауэлл";
    result.color = "purple";
    result.success = false;
    
    double x = x0;
    double fx = func(x);
    double h = h0;
    
    result.x_history.push_back(x);
    result.f_history.push_back(fx);
    
    int iter = 0;
    int stagnation_count = 0;
    
    while (iter < max_iter && stagnation_count < 20) {
        double x1 = x - h;
        double x2 = x;
        double x3 = x + h;
        double f1 = func(x1);
        double f2 = func(x2);
        double f3 = func(x3);
        
        Parabola par;
        par.x1 = x1; par.f1 = f1;
        par.x2 = x2; par.f2 = f2;
        par.x3 = x3; par.f3 = f3;
        
        double d1 = (f1 - f2) / (x1 - x2);
        double d2 = (f3 - f2) / (x3 - x2);
        par.c = (d2 - d1) / (x3 - x1);
        par.b = d1 + par.c * (x2 - x1);
        par.a = f2;
        
        if (std::abs(par.c) > 1e-10 && par.c > 0) {
            par.x_min = x2 - par.b / (2.0 * par.c);
            par.f_min = par.a - par.b * par.b / (4.0 * par.c);
        } else {
            if (f1 < f2 && f1 < f3) {
                par.x_min = x1;
                par.f_min = f1;
            } else if (f3 < f2 && f3 < f1) {
                par.x_min = x3;
                par.f_min = f3;
            } else {
                par.x_min = x2;
                par.f_min = f2;
            }
        }
        
        result.parabolas.push_back(par);
        
        if (f1 >= f2 && f3 >= f2) {
            double denom = 2.0 * (f1 - 2.0 * f2 + f3);
            if (std::abs(denom) > 1e-10) {
                double x_parabola = x2 - h * (f3 - f1) / denom;
                double f_parabola = func(x_parabola);
                
                if (f_parabola < f2 - eps) {
                    x = x_parabola;
                    fx = f_parabola;
                    result.x_history.push_back(x);
                    result.f_history.push_back(fx);
                    stagnation_count = 0;
                } else {
                    stagnation_count++;
                    h *= 0.5;
                }
            } else {
                stagnation_count++;
                h *= 0.5;
            }
        } else {
            if (f1 < f3) {
                x = x1;
                fx = f1;
            } else {
                x = x3;
                fx = f3;
            }
            result.x_history.push_back(x);
            result.f_history.push_back(fx);
            stagnation_count = 0;
            h *= 1.2;
        }
        
        iter++;
        if (h < eps) break;
    }
    
    result.x_min = x;
    result.f_min = fx;
    result.iterations = int(result.x_history.size());
    result.a = *std::min_element(result.x_history.begin(), result.x_history.end());
    result.b = *std::max_element(result.x_history.begin(), result.x_history.end());
    
    double f_left = func(result.x_min - eps);
    double f_right = func(result.x_min + eps);
    result.success = (fx <= f_left + eps && fx <= f_right + eps);
    
    return result;
}

bool check_unimodality(const std::function<double(double)>& func,
                       double a, double b, int n = 500, double tol = 1e-6) {
    std::cout << "\n=== Проверка унимодальности на [" << a << ", " << b << "] ===" << std::endl;
    
    double step = (b - a) / n;
    double min_val = 1e300, max_val = -1e300;
    double min_x = a;
    
    for (int i = 0; i <= n; ++i) {
        double x = a + i * step;
        double y = func(x);
        if (y < min_val) {
            min_val = y;
            min_x = x;
        }
        max_val = std::max(max_val, y);
    }
    
    std::cout << "Минимум на отрезке: x = " << min_x << ", f = " << min_val << std::endl;
    std::cout << "Максимум на отрезке: f = " << max_val << std::endl;
    
    int min_count = 0;
    for (int i = 1; i < n; ++i) {
        double x_prev = a + (i-1) * step;
        double x_curr = a + i * step;
        double x_next = a + (i+1) * step;
        double f_prev = func(x_prev);
        double f_curr = func(x_curr);
        double f_next = func(x_next);
        
        if (f_curr < f_prev && f_curr < f_next) {
            min_count++;
        }
    }
    
    std::cout << "Число локальных минимумов на отрезке: " << min_count << std::endl;
    
    bool is_unimodal = (min_count <= 1);
    std::cout << "Функция " << (is_unimodal ? "унимодальна" : "НЕ унимодальна")
              << " на данном отрезке" << std::endl;
    
    return is_unimodal;
}

bool check_rain_rule(const std::function<double(double)>& func, double x_min,
                     const std::string& method_name) {
    std::cout << "\n=== Правило дождя: " << method_name << " ===" << std::endl;
    std::cout << "Точка: x = " << std::fixed << std::setprecision(6) << x_min << std::endl;
    
    double fpp = d2f(func, x_min);
    double f_val = func(x_min);
    double f_left = func(x_min - 1e-4);
    double f_right = func(x_min + 1e-4);
    
    std::cout << "f(x) = " << f_val << std::endl;
    std::cout << "f(x-h) = " << f_left << ", f(x+h) = " << f_right << std::endl;
    std::cout << "f''(x) = " << std::scientific << fpp << std::endl;
    
    bool is_minimum = (fpp > 0);
    bool is_maximum = (fpp < 0);
    
    if (std::abs(fpp) < 1e-3) {
        std::cout << "Возможно, точка негладкости (f'' ≈ 0)" << std::endl;
    }
    
    std::cout << "Вывод: " << (is_minimum ? "МИНИМУМ ✓" :
                               (is_maximum ? "МАКСИМУМ ✗" : "СЕДЛО/НЕГЛАДКАЯ ТОЧКА ?")) << std::endl;
    
    return is_minimum;
}

void plot_results(const std::vector<MethodResult>& results,
                  const std::function<double(double)>& func,
                  double x_min_plot, double x_max_plot) {
    Plot2D plot1;
    
    std::vector<double> x_vals, y_vals;
    for (double x = x_min_plot; x <= x_max_plot; x += 0.001) {
        x_vals.push_back(x);
        y_vals.push_back(func(x));
    }
    plot1.drawCurve(x_vals, y_vals).label("f(x) = |x³ - 3x|").lineColor("black");
    
    std::vector<double> x_mins = {-1.0, 1.0};
    std::vector<double> y_mins = {2.0, 2.0};
    plot1.drawPoints(x_mins, y_mins)
        .pointType(2)
        .pointSize(2)
        .lineColor("dark-green")
        .label("Аналит. минимумы (x=±1)");
    
    for (const auto& res : results) {
        if (res.x_history.size() > 1) {
            std::vector<double> x_hist(res.x_history.begin(), res.x_history.end());
            std::vector<double> y_hist(res.f_history.begin(), res.f_history.end());
            
            plot1.drawCurve(x_hist, y_hist)
                .lineColor(res.color)
                .lineWidth(1)
                .label(res.name + " path");
        }
        
        std::vector<double> x_hist(res.x_history.begin(), res.x_history.end());
        std::vector<double> y_hist(res.f_history.begin(), res.f_history.end());
        plot1.drawPoints(x_hist, y_hist)
            .pointType(7)
            .pointSize(0.8)
            .lineColor(res.color);
        
        std::vector<double> x_min_vec = {res.x_min};
        std::vector<double> y_min_vec = {res.f_min};
        std::string label = res.name + ": " +
                           std::to_string(res.iterations) + " it" +
                           (res.success ? "" : " [FAIL]");
        
        plot1.drawPoints(x_min_vec, y_min_vec)
            .pointType(9)
            .pointSize(2.5)
            .lineColor(res.success ? res.color : "orange")
            .label(label);
    }
    
    plot1.xlabel("x");
    plot1.ylabel("f(x)");
    plot1.legend().atOutsideBottom().displayHorizontal();
    plot1.grid().show();
    plot1.size(1200, 800);
    
    Plot2D plot2;
    
    plot2.drawCurve(x_vals, y_vals).label("f(x)").lineColor("black").lineWidth(2);
    
    const MethodResult* powell_res = nullptr;
    for (const auto& res : results) {
        if (res.name == "Пауэлл") {
            powell_res = &res;
            break;
        }
    }
    
    if (powell_res && !powell_res->parabolas.empty()) {
        std::vector<std::string> parabola_colors = {"red", "blue", "green", "purple", "orange"};
        
        for (size_t p = 0; p < std::min(powell_res->parabolas.size(), size_t(5)); ++p) {
            const auto& par = powell_res->parabolas[p];
            
            std::vector<double> px = {par.x1, par.x2, par.x3};
            std::vector<double> py = {par.f1, par.f2, par.f3};
            plot2.drawPoints(px, py)
                .pointType(4 + int(p))
                .pointSize(1.5)
                .lineColor(parabola_colors[p])
                .label("Iter " + std::to_string(p+1) + " points");
            
            std::vector<double> par_x, par_y;
            double x_start = std::min({par.x1, par.x2, par.x3}) - 0.2;
            double x_end = std::max({par.x1, par.x2, par.x3}) + 0.2;
            
            for (double x = x_start; x <= x_end; x += 0.01) {
                double L1 = ((x - par.x2) * (x - par.x3)) / ((par.x1 - par.x2) * (par.x1 - par.x3));
                double L2 = ((x - par.x1) * (x - par.x3)) / ((par.x2 - par.x1) * (par.x2 - par.x3));
                double L3 = ((x - par.x1) * (x - par.x2)) / ((par.x3 - par.x1) * (par.x3 - par.x2));
                double y = par.f1 * L1 + par.f2 * L2 + par.f3 * L3;
                
                par_x.push_back(x);
                par_y.push_back(y);
            }
            
            plot2.drawCurve(par_x, par_y)
                .lineColor(parabola_colors[p])
                .lineWidth(1)
                .label("Parabola iter " + std::to_string(p+1));
            
            if (std::abs(par.c) > 1e-10) {
                std::vector<double> px_min = {par.x_min};
                std::vector<double> py_min = {par.f_min};
                plot2.drawPoints(px_min, py_min)
                    .pointType(9)
                    .pointSize(2)
                    .lineColor(parabola_colors[p]);
            }
        }
        
        std::vector<double> fx = {powell_res->x_min};
        std::vector<double> fy = {powell_res->f_min};
        plot2.drawPoints(fx, fy)
            .pointType(11)
            .pointSize(3)
            .lineColor("purple")
            .label("Final result");
    }
    
    plot2.xlabel("x");
    plot2.ylabel("f(x)");
    plot2.legend().atOutsideBottom().displayHorizontal();
    plot2.grid().show();
    plot2.size(1200, 800);
    
    Figure fig1 = {{plot1}};
    Figure fig2 = {{plot2}};
    
    Canvas canvas = {{fig1}, {fig2}};
    canvas.size(1400, 1000);
    canvas.show();
}

int main() {
    std::cout << "=== Оптимизация f(x) = |x³ - 3x| ===" << std::endl;
    std::cout << "Глобальные минимумы: x = 0, ±√3 ≈ ±1.732 (f = 0)" << std::endl;
    std::cout << "Локальные максимумы: x = ±1 (f = 2)" << std::endl;
    std::cout << "Точки негладкости: x = 0, ±√3" << std::endl;
    std::cout << std::endl;
    
    std::vector<double> start_points = {0.5, -0.5, 1.5, -1.5};
    
    for (double x0 : start_points) {
        std::cout << "\n\n========================================" << std::endl;
        std::cout << "НАЧАЛЬНАЯ ТОЧКА x0 = " << x0 << std::endl;
        std::cout << "========================================" << std::endl;
        
        std::vector<MethodResult> results;
        double eps = 1e-5;
        
        std::cout << "\n--- Метод Свенна ---" << std::endl;
        auto sven_res = sven_method(f, x0, 0.05);
        std::cout << "Отрезок: [" << sven_res.a << ", " << sven_res.b << "]" << std::endl;
        std::cout << "Итераций: " << sven_res.iterations << std::endl;
        
        check_unimodality(f, sven_res.a, sven_res.b);
        
        std::cout << "\n--- Метод Фибоначчи ---" << std::endl;
        auto fib_res = fibonacci_method(f, sven_res.a, sven_res.b, eps);
        std::cout << "x_min = " << fib_res.x_min << ", f_min = " << fib_res.f_min << std::endl;
        std::cout << "Итераций: " << fib_res.iterations << std::endl;
        results.push_back(fib_res);
        
        std::cout << "\n--- Метод обратного переменного шага ---" << std::endl;
        auto rev_res = reverse_variable_step_method(f, x0, 0.1, eps);
        std::cout << "x_min = " << rev_res.x_min << ", f_min = " << rev_res.f_min << std::endl;
        std::cout << "Итераций: " << rev_res.iterations << std::endl;
        results.push_back(rev_res);
        
        std::cout << "\n--- Метод Пауэлла ---" << std::endl;
        auto powell_res = powell_method(f, x0, 0.1, eps);
        std::cout << "x_min = " << powell_res.x_min << ", f_min = " << powell_res.f_min << std::endl;
        std::cout << "Итераций: " << powell_res.iterations << std::endl;
        std::cout << "Построено парабол: " << powell_res.parabolas.size() << std::endl;
        results.push_back(powell_res);
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "ПРОВЕРКА ПРАВИЛОМ ДОЖДЯ" << std::endl;
        std::cout << "========================================" << std::endl;
        for (const auto& res : results) {
            check_rain_rule(f, res.x_min, res.name);
        }
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "СВОДНАЯ ТАБЛИЦА (x0 = " << x0 << ")" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << std::setw(28) << "Метод"
                  << std::setw(15) << "x_min"
                  << std::setw(15) << "f_min"
                  << std::setw(10) << "Итер."
                  << std::setw(10) << "Успех" << std::endl;
        std::cout << "--------------------------------------------------------" << std::endl;
        
        for (const auto& res : results) {
            std::cout << std::setw(28) << res.name
                      << std::setw(15) << std::fixed << std::setprecision(6) << res.x_min
                      << std::setw(15) << std::setprecision(6) << res.f_min
                      << std::setw(10) << res.iterations
                      << std::setw(10) << (res.success ? "YES" : "NO") << std::endl;
        }
        
        if (x0 == start_points[0]) {
            plot_results(results, f, -2.5, 2.5);
        }
    }
    
    return 0;
}
