#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <iomanip>
#include <algorithm>
#include <sciplot/sciplot.hpp>

using namespace std;
using namespace sciplot;

//double f1(double x) {
//    return 5.0 - 24.0 * x + 17.0 * x * x - (11.0 / 3.0) * x * x * x + 0.25 * x * x * x * x;
//}
//
//double f2(double x) {
//    return -((x * x - 4.0) * (x * x - 4.0) + x / 2.0);
//}

double f1(double x) {
    return (x + 3.0) * (x + 3.0);
}

double f2(double x) {
    return 12.0 - (x - 3.0) * (x - 3.0);
}

vector<double> min_to_utility(const vector<double>& values) {
    double v_min = *min_element(values.begin(), values.end());
    double v_max = *max_element(values.begin(), values.end());

    vector<double> result(values.size());

    if (fabs(v_max - v_min) < 1e-12) {
        for (size_t i = 0; i < values.size(); i++)
            result[i] = 1.0;

        return result;
    }

    for (size_t i = 0; i < values.size(); i++)
        result[i] = (v_max - values[i]) / (v_max - v_min);

    return result;
}

vector<double> max_to_utility(const vector<double>& values) {
    double v_min = *min_element(values.begin(), values.end());
    double v_max = *max_element(values.begin(), values.end());

    vector<double> result(values.size());

    if (fabs(v_max - v_min) < 1e-12) {
        for (size_t i = 0; i < values.size(); i++)
            result[i] = 1.0;

        return result;
    }

    for (size_t i = 0; i < values.size(); i++)
        result[i] = (values[i] - v_min) / (v_max - v_min);

    return result;
}

bool dominates(double f1_a, double f2_a, double f1_b, double f2_b) {
    return (f1_a <= f1_b && f2_a >= f2_b) && (f1_a < f1_b || f2_a > f2_b);
}

vector<bool> pareto_mask(const vector<double>& f1_values, const vector<double>& f2_values) {
    size_t n = f1_values.size();
    vector<bool> mask(n, true);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            if (i != j && dominates(f1_values[j], f2_values[j], f1_values[i], f2_values[i])) {
                mask[i] = false;
                break;
            }
        }
    }

    return mask;
}

double utility1(double x, double f1_min, double f1_max) {
    if (fabs(f1_max - f1_min) < 1e-12)
        return 1.0;

    return (f1_max - f1(x)) / (f1_max - f1_min);
}

double utility2(double x, double f2_min, double f2_max) {
    if (fabs(f2_max - f2_min) < 1e-12)
        return 1.0;

    return (f2(x) - f2_min) / (f2_max - f2_min);
}

double swo_function(double x,
                    double lambda1,
                    double lambda2,
                    double f1_min,
                    double f1_max,
                    double f2_min,
                    double f2_max)
{
    return lambda1 * utility1(x, f1_min, f1_max)
         + lambda2 * utility2(x, f2_min, f2_max);
}

double golden_section_max(double left,
                          double right,
                          double lambda1,
                          double lambda2,
                          double f1_min,
                          double f1_max,
                          double f2_min,
                          double f2_max)
{
    double k = (sqrt(5.0) - 1.0) / 2.0;

    double c = right - k * (right - left);
    double d = left + k * (right - left);

    while (fabs(right - left) > 1e-8) {
        double f_c = swo_function(c, lambda1, lambda2, f1_min, f1_max, f2_min, f2_max);
        double f_d = swo_function(d, lambda1, lambda2, f1_min, f1_max, f2_min, f2_max);

        if (f_c > f_d) right = d;
        else left = c;

        c = right - k * (right - left);
        d = left + k * (right - left);
    }

    return (left + right) / 2.0;
}

void show_plots(const vector<double>& x_grid,
                const vector<double>& f1_grid,
                const vector<double>& f2_grid,
                const vector<double>& u1_grid,
                const vector<double>& u2_grid,
                const vector<double>& s_grid,
                const vector<bool>& pareto_flags,
                double x_star,
                double f1_star,
                double f2_star,
                double u1_star,
                double u2_star,
                double s_star)
{
    vector<double> pareto_x;
    vector<double> pareto_f1;
    vector<double> pareto_f2;
    vector<double> pareto_s;
    vector<double> dom_f1;
    vector<double> dom_f2;

    for (size_t i = 0; i < x_grid.size(); i++) {
        if (pareto_flags[i]) {
            pareto_x.push_back(x_grid[i]);
            pareto_f1.push_back(f1_grid[i]);
            pareto_f2.push_back(f2_grid[i]);
            pareto_s.push_back(s_grid[i]);

        } else {
            dom_f1.push_back(f1_grid[i]);
            dom_f2.push_back(f2_grid[i]);
        }
    }

    double x_min = *min_element(x_grid.begin(), x_grid.end());
    double x_max = *max_element(x_grid.begin(), x_grid.end());

    double y_min_criteria = min(*min_element(f1_grid.begin(), f1_grid.end()),
                                *min_element(f2_grid.begin(), f2_grid.end()));
    double y_max_criteria = max(*max_element(f1_grid.begin(), f1_grid.end()),
                                *max_element(f2_grid.begin(), f2_grid.end()));

    double criteria_margin = 0.05 * (y_max_criteria - y_min_criteria);

    if (fabs(criteria_margin) < 1e-9)
        criteria_margin = 1.0;

    double f1_min_all = *min_element(f1_grid.begin(), f1_grid.end());
    double f1_max_all = *max_element(f1_grid.begin(), f1_grid.end());
    double f2_min_all = *min_element(f2_grid.begin(), f2_grid.end());
    double f2_max_all = *max_element(f2_grid.begin(), f2_grid.end());

    double pareto_x_margin = 0.05 * (f1_max_all - f1_min_all);
    double pareto_y_margin = 0.05 * (f2_max_all - f2_min_all);

    if (fabs(pareto_x_margin) < 1e-9)
        pareto_x_margin = 1.0;

    if (fabs(pareto_y_margin) < 1e-9)
        pareto_y_margin = 1.0;

    double s_min = *min_element(s_grid.begin(), s_grid.end());
    double s_max = *max_element(s_grid.begin(), s_grid.end());

    double s_margin = 0.05 * (s_max - s_min);

    if (fabs(s_margin) < 1e-9)
        s_margin = 0.1;

    Plot2D p1;
    p1.xlabel("x");
    p1.ylabel("Значение");
    p1.xrange(x_min, x_max);
    p1.yrange(y_min_criteria - criteria_margin, y_max_criteria + criteria_margin);

    p1.drawCurve(x_grid, f1_grid)
        .label("f1(x)")
        .lineColor("blue")
        .lineWidth(2);

    p1.drawCurve(x_grid, f2_grid)
        .label("f2(x)")
        .lineColor("red")
        .lineWidth(2);

    if (!pareto_x.empty()) {
        p1.drawPoints(pareto_x, pareto_f1)
            .label("Парето на f1")
            .pointType(7)
            .pointSize(1.2)
            .lineColor("blue");

        p1.drawPoints(pareto_x, pareto_f2)
            .label("Парето на f2")
            .pointType(7)
            .pointSize(1.2)
            .lineColor("red");
    }

    p1.drawPoints(vector<double>{x_star}, vector<double>{f1_star})
        .label("Компромисс на f1")
        .pointType(5)
        .pointSize(2.2)
        .lineColor("blue");

    p1.drawPoints(vector<double>{x_star}, vector<double>{f2_star})
        .label("Компромисс на f2")
        .pointType(5)
        .pointSize(2.2)
        .lineColor("red");

    Plot2D p2;
    p2.xlabel("f1");
    p2.ylabel("f2");
    p2.xrange(f1_min_all - pareto_x_margin, f1_max_all + pareto_x_margin);
    p2.yrange(f2_min_all - pareto_y_margin, f2_max_all + pareto_y_margin);

    if (!dom_f1.empty()) {
        p2.drawPoints(dom_f1, dom_f2)
            .label("Доминируемые")
            .pointType(7)
            .lineColor("#add8e6");
    }

    if (!pareto_f1.empty()) {
        p2.drawPoints(pareto_f1, pareto_f2)
            .label("Парето")
            .pointType(7)
            .pointSize(1.2)
            .lineColor("red");
    }

    p2.drawPoints(vector<double>{f1_star}, vector<double>{f2_star})
        .label("Компромисс")
        .pointType(5)
        .pointSize(2.2)
        .lineColor("magenta");

    Plot2D p3;
    p3.xlabel("x");
    p3.ylabel("S(x)");
    p3.xrange(x_min, x_max);
    p3.yrange(s_min - s_margin, s_max + s_margin);

    p3.drawCurve(x_grid, s_grid)
        .label("S(x)")
        .lineColor("dark-green")
        .lineWidth(2);

    if (!pareto_x.empty()) {
        p3.drawPoints(pareto_x, pareto_s)
            .label("Парето-точки")
            .pointType(7)
            .pointSize(1.2)
            .lineColor("magenta");
    }

    p3.drawPoints(vector<double>{x_star}, vector<double>{s_star})
        .label("Компромисс")
        .pointType(5)
        .pointSize(2.2)
        .lineColor("black");

    Plot2D p4;
    p4.xlabel("x");
    p4.ylabel("Полезность");
    p4.xrange(x_min, x_max);
    p4.yrange(0.0, 1.05);

    p4.drawCurve(x_grid, u1_grid)
        .label("u1(x)")
        .lineColor("blue")
        .lineWidth(2);

    p4.drawCurve(x_grid, u2_grid)
        .label("u2(x)")
        .lineColor("red")
        .lineWidth(2);

    p4.drawPoints(vector<double>{x_star}, vector<double>{u1_star})
        .label("u1(x*)")
        .pointType(5)
        .pointSize(2.0)
        .lineColor("blue");

    p4.drawPoints(vector<double>{x_star}, vector<double>{u2_star})
        .label("u2(x*)")
        .pointType(5)
        .pointSize(2.0)
        .lineColor("red");

    Figure fig = {{p1, p2}, {p3, p4}};
    fig.title("Метод взвешенных критериев (SWO)");

    Canvas canvas = {{fig}};
    canvas.size(1400, 900);
    canvas.show();
}

int main() {
    double a = -5.0;
    double b = 5.0;

    vector<double> x_disc = {-4.0, -2.0, 0.0, 2.0, 4.0};
    vector<string> names = {"Прибор 1", "Прибор 2", "Прибор 3", "Прибор 4", "Прибор 5"};

    double lambda1 = 0.5;
    double lambda2 = 0.5;

    int n = 4000;

    vector<double> x_grid;
    vector<double> f1_grid;
    vector<double> f2_grid;

    for (int i = 0; i < n; i++) {
        double x = a + (b - a) * i / (n - 1.0);

        x_grid.push_back(x);
        f1_grid.push_back(f1(x));
        f2_grid.push_back(f2(x));
    }

    vector<double> f1_disc;
    vector<double> f2_disc;

    for (double x : x_disc) {
        f1_disc.push_back(f1(x));
        f2_disc.push_back(f2(x));
    }

    vector<bool> pareto_disc = pareto_mask(f1_disc, f2_disc);
    vector<bool> pareto_flags = pareto_mask(f1_grid, f2_grid);

    vector<double> u1_grid = min_to_utility(f1_grid);
    vector<double> u2_grid = max_to_utility(f2_grid);

    vector<double> s_grid;
    for (size_t i = 0; i < x_grid.size(); i++) {
        double s = lambda1 * u1_grid[i] + lambda2 * u2_grid[i];
        s_grid.push_back(s);
    }

    double f1_min = *min_element(f1_grid.begin(), f1_grid.end());
    double f1_max = *max_element(f1_grid.begin(), f1_grid.end());
    double f2_min = *min_element(f2_grid.begin(), f2_grid.end());
    double f2_max = *max_element(f2_grid.begin(), f2_grid.end());

    double x_star = golden_section_max(a, b, lambda1, lambda2, f1_min, f1_max, f2_min, f2_max);
    double f1_star = f1(x_star);
    double f2_star = f2(x_star);

    double u1_star = utility1(x_star, f1_min, f1_max);
    double u2_star = utility2(x_star, f2_min, f2_max);
    double s_star = lambda1 * u1_star + lambda2 * u2_star;

    cout << fixed << setprecision(6);

    cout << "=== ДИСКРЕТНЫЙ ВАРИАНТ ===\n" << endl;
    cout << left
         << setw(12) << "Объект"
         << right
         << setw(12) << "x"
         << setw(16) << "f1(x)"
         << setw(16) << "f2(x)"
         << setw(12) << "S(x)"
         << "      "
         << left << setw(10) << "Статус"
         << "\n";

    cout << string(88, '-') << "\n";

    for (size_t i = 0; i < x_disc.size(); i++) {
        string status = pareto_disc[i] ? "Парето" : "-";
        double s_disc = lambda1 * (u1_grid[0]) + lambda2 * (u2_grid[0]);

        double u1_disc = (f1_max - f1_disc[i]) / (f1_max - f1_min);
        double u2_disc = (f2_disc[i] - f2_min) / (f2_max - f2_min);
        s_disc = lambda1 * u1_disc + lambda2 * u2_disc;

        cout << left  << setw(12) << names[i]
             << right << setw(12) << x_disc[i]
             << setw(16) << f1_disc[i]
             << setw(16) << f2_disc[i]
             << setw(12) << s_disc
             << "      "
             << left << setw(10) << status
             << "\n";
    }

    cout << string(88, '-') << "\n\n";

    cout << "=== НЕПРЕРЫВНЫЙ ВАРИАНТ ===\n" << endl;
    cout << left << setw(8) << "x*"     << "= " << x_star << "\n";
    cout << left << setw(8) << "f1(x*)" << "= " << f1_star << "\n";
    cout << left << setw(8) << "f2(x*)" << "= " << f2_star << "\n";
    cout << left << setw(8) << "u1(x*)" << "= " << u1_star << "\n";
    cout << left << setw(8) << "u2(x*)" << "= " << u2_star << "\n";
    cout << left << setw(8) << "S(x*)"  << "= " << s_star << "\n\n";

    show_plots(x_grid, f1_grid, f2_grid, u1_grid, u2_grid, s_grid,
               pareto_flags, x_star, f1_star, f2_star, u1_star, u2_star, s_star);

    return 0;
}
