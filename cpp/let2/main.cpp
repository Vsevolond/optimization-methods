#include <iostream>
#include <tuple>
#include <vector>
#include <cmath>
#include <string>
#include <iomanip>
#include <algorithm>
#include <limits>
#include <sciplot/sciplot.hpp>

using namespace sciplot;


double f_rast(double x, double y) {
    const double A = 10.0;
    return 2*A + (x*x - A*cos(2*M_PI*x)) + (y*y - A*cos(2*M_PI*y));
}

double f_rosen(double x, double y) {
    const double a = 1.0, b = 100.0;
    return (a - x)*(a - x) + b*(y - x*x)*(y - x*x);
}

double f_schwef(double x, double y) {
    return 418.9829*2 - (x*sin(sqrt(fabs(x))) + y*sin(sqrt(fabs(y))));
}


using Point   = std::vector<double>;
using Simplex = std::vector<Point>;


double norm2(const Point& a, const Point& b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i)
        s += (a[i] - b[i]) * (a[i] - b[i]);
    return sqrt(s);
}

Point add(const Point& a, const Point& b) {
    Point r(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        r[i] = a[i] + b[i];
    return r;
}

Point sub(const Point& a, const Point& b) {
    Point r(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        r[i] = a[i] - b[i];
    return r;
}

Point scale(double s, const Point& a) {
    Point r(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        r[i] = s * a[i];
    return r;
}


Simplex regular_simplex(const Point& x0, double a = 1.0) {
    int n = (int)x0.size();
    double P = (sqrt((double)(n + 1)) + n - 1) / (n * sqrt(2.0));
    double Q = (sqrt((double)(n + 1)) - 1)     / (n * sqrt(2.0));

    Simplex S(n + 1, Point(n));
    S[0] = x0;

    for (int k = 0; k < n; ++k) {
        Point v(n, Q);
        v[k] = P;
        S[k + 1] = add(x0, v);
    }

    double d = norm2(S[1], S[0]);
    double s = a / d;

    for (int i = 1; i <= n; ++i)
        S[i] = add(x0, scale(s, sub(S[i], x0)));

    return S;
}


struct SimplexResult {
    Point                best;
    double               f_best;
    std::vector<Simplex> hist;
    double               a_final;
    int                  iterations;
};


SimplexResult regular_simp_method(
    double (*f)(double, double),
    const Point& x0,
    double eps   = 1e-3,
    double a0    = 1.0,
    int    maxit = 200
) {
    int n    = (int)x0.size();
    double a = a0;
    Simplex S = regular_simplex(x0, a);

    SimplexResult result;
    result.hist.push_back(S);

    Point last_worst;
    bool  has_last  = false;
    auto  tol_cycle = [](double a_) { return 1e-9 * std::max(a_, 1.0); };

    int iter = 0;
    for (; iter < maxit; ++iter) {
        if (a < eps) break;

        std::vector<double> vals(n + 1);
        for (int i = 0; i <= n; ++i)
            vals[i] = f(S[i][0], S[i][1]);

        int iw = (int)(std::max_element(vals.begin(), vals.end()) - vals.begin());
        int ib = (int)(std::min_element(vals.begin(), vals.end()) - vals.begin());

        Point worst = S[iw];

        Point c(n, 0.0);
        for (int i = 0; i <= n; ++i) {
            if (i == iw) continue;
            for (int j = 0; j < n; ++j)
                c[j] += S[i][j];
        }
        for (int j = 0; j < n; ++j)
            c[j] /= n;

        Point newp  = sub(scale(2.0, c), worst);
        bool  cycle = has_last && (norm2(newp, last_worst) < tol_cycle(a));

        if (cycle) {
            Point best = S[ib];
            for (int i = 0; i <= n; ++i)
                if (i != ib)
                    S[i] = add(best, scale(0.5, sub(S[i], best)));
            a        *= 0.5;
            has_last  = false;
        } else {
            last_worst = worst;
            has_last   = true;
            S[iw]      = newp;
        }

        result.hist.push_back(S);
    }

    std::vector<double> vals(n + 1);
    for (int i = 0; i <= n; ++i)
        vals[i] = f(S[i][0], S[i][1]);

    int ib = (int)(std::min_element(vals.begin(), vals.end()) - vals.begin());

    result.best       = S[ib];
    result.f_best     = vals[ib];
    result.a_final    = a;
    result.iterations = iter;

    return result;
}


void visualize(
    const SimplexResult& result,
    double (*func)(double, double),
    const std::string& func_name,
    double x_min, double x_max,
    double y_min, double y_max
) {
    const int n = 50;
    std::vector<double> xg, yg, zg;

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double x = x_min + (x_max - x_min) * i / (n - 1);
            double y = y_min + (y_max - y_min) * j / (n - 1);
            xg.push_back(x);
            yg.push_back(y);
            zg.push_back(func(x, y));
        }

    std::vector<double> tx, ty, tz;
    for (const auto& S : result.hist) {
        std::vector<double> vals;
        for (const auto& p : S)
            vals.push_back(func(p[0], p[1]));
        int ib = (int)(std::min_element(vals.begin(), vals.end()) - vals.begin());
        tx.push_back(S[ib][0]);
        ty.push_back(S[ib][1]);
        tz.push_back(vals[ib]);
    }

    Plot3D plot3d;
    plot3d.xlabel("x1");
    plot3d.ylabel("x2");
    plot3d.zlabel("f(x)");
    plot3d.drawWithVecs("lines", xg, yg, zg).label("function").lineColor("gray").lineWidth(1);
    plot3d.drawCurve(tx, ty, tz).label("trajectory").lineColor("green").lineWidth(3);
    plot3d.drawPoints(tx, ty, tz).label("points").lineColor("green").pointType(7).pointSize(2);

    Plot2D plot2d;
    plot2d.xlabel("x1");
    plot2d.ylabel("x2");

    for (size_t s = 0; s < result.hist.size(); ++s) {
        const auto& S = result.hist[s];
        std::vector<double> vals;
        for (const auto& p : S)
            vals.push_back(func(p[0], p[1]));

        int worst = (int)(std::max_element(vals.begin(), vals.end()) - vals.begin());
        int m     = (int)S.size();

        for (int i = 0; i < m; ++i) {
            int j = (i + 1) % m;
            plot2d.drawCurve(
                std::vector<double>{S[i][0], S[j][0]},
                std::vector<double>{S[i][1], S[j][1]}
            ).lineColor("black").lineWidth(1).label(s == 0 && i == 0 ? "simplex edges" : "");
        }

        for (int i = 0; i < m; ++i) {
            std::vector<double> ppx = {S[i][0]};
            std::vector<double> ppy = {S[i][1]};
            if (i == worst)
                plot2d.drawPoints(ppx, ppy)
                      .lineColor("red").pointType(7).pointSize(2)
                      .label(s == 0 ? "worst vertex" : "");
            else
                plot2d.drawPoints(ppx, ppy)
                      .lineColor("blue").pointType(7).pointSize(1.5)
                      .label(s == 0 && i == (worst + 1) % m ? "other vertices" : "");
        }
    }

    plot2d.drawCurve(tx, ty).label("trajectory").lineColor("green").lineWidth(2);
    plot2d.drawPoints(tx, ty).label("best points").lineColor("green").pointType(5).pointSize(2);

    Figure fig = {{plot3d, plot2d}};
    fig.title("Regular Simplex Method — " + func_name);

    Canvas canvas = {{fig}};
    canvas.size(1200, 600);
    canvas.show();
}


void run_test(
    const std::string& name,
    double (*func)(double, double),
    Point  x0,
    double a0,
    double x_min, double x_max,
    double y_min, double y_max
) {
    std::cout << "\n========================================\n";
    std::cout << "Функция: " << name << "\n";
    std::cout << "Начальная точка: (" << x0[0] << ", " << x0[1] << ")\n";
    std::cout << "========================================\n";

    auto result = regular_simp_method(func, x0, 1e-3, a0);

    std::cout << "Итераций: "          << result.iterations << "\n";
    std::cout << "Найденный минимум: (" << std::fixed << std::setprecision(6)
              << result.best[0] << ", " << result.best[1] << ")\n";
    std::cout << "Значение функции:  "  << result.f_best   << "\n";
    std::cout << "Финальный размер:  "  << result.a_final  << "\n";

    visualize(result, func, name, x_min, x_max, y_min, y_max);
}


int main() {
    run_test("Rastrigin",  f_rast,
             {-2.5, -1.5}, 1.0,
             -5.0, 5.0, -5.0, 5.0);

    run_test("Rosenbrock", f_rosen,
             {2.9, 1.4}, 1.0,
             -2.0, 2.0, -1.0, 3.0);

    run_test("Schwefel",   f_schwef,
             {420.0, 420.0}, 1.0,
             -500.0, 500.0, -500.0, 500.0);

    return EXIT_SUCCESS;
}
