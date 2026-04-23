#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <numbers>

#include <sciplot/sciplot.hpp>

using namespace std;
using namespace sciplot;
using namespace numbers;

const int DIM = 3;
const double LOWER_BOUND = 0.5;
const double UPPER_BOUND = 12.0;

struct Cell {
    vector<double> x;
    double fitness = numeric_limits<double>::infinity();
};

double ackley(const vector<double>& x) {
    const int n = static_cast<int>(x.size());

    double sumSquares = 0.0;
    double sumCos = 0.0;

    for (double xi : x) {
        sumSquares += xi * xi;
        sumCos += cos(2.0 * pi * xi);
    }

    double term1 = -20.0 * exp(-0.2 * sqrt(sumSquares / n));
    double term2 = -exp(sumCos / n);

    return term1 + term2 + 20.0;
}

bool fitnessLess(const Cell& a, const Cell& b) {
    return a.fitness < b.fitness;
}

Cell randomCell(mt19937& gen) {
    uniform_real_distribution<double> coordinateDistribution(LOWER_BOUND, UPPER_BOUND);

    Cell randomGeneratedCell;
    randomGeneratedCell.x.resize(DIM);

    for (int coordinateIndex = 0; coordinateIndex < DIM; coordinateIndex++) {
        randomGeneratedCell.x[coordinateIndex] = coordinateDistribution(gen);
    }

    return randomGeneratedCell;
}

double mutateCoordinate(double parentCoordinateValue, double cloneCoordinateValue, double mutationParameter, mt19937& gen) {
    uniform_real_distribution<double> unitDistribution(0.0, 1.0);

    while (true) {
        double u = unitDistribution(gen);
        double mutatedCoordinateValue = cloneCoordinateValue;

        if (u > 0.5) { /// y_i = x_i + U[0, b - x_i] * r
            uniform_real_distribution<double> positiveShiftDistribution(0.0, UPPER_BOUND - parentCoordinateValue);
            mutatedCoordinateValue = cloneCoordinateValue + positiveShiftDistribution(gen) * mutationParameter;
            
        } else if (u < 0.5) { /// y_i = x_i - U[0, b - x_i] * r
            uniform_real_distribution<double> negativeShiftDistribution(0.0, parentCoordinateValue - LOWER_BOUND);
            mutatedCoordinateValue = cloneCoordinateValue - negativeShiftDistribution(gen) * mutationParameter;
            
        } else {
            continue;
        }

        if (mutatedCoordinateValue >= LOWER_BOUND && mutatedCoordinateValue <= UPPER_BOUND) {
            return mutatedCoordinateValue;
        }
    }
}

Cell mutateClone(const Cell& parentCell, const Cell& cloneCell, double mutationParameter, mt19937& gen) {
    Cell mutantClone = cloneCell;

    for (int coordinateIndex = 0; coordinateIndex < DIM; coordinateIndex++) {
        mutantClone.x[coordinateIndex] = mutateCoordinate(
            parentCell.x[coordinateIndex],
            cloneCell.x[coordinateIndex],
            mutationParameter,
            gen
        );
    }

    return mutantClone;
}

void showConvergencePlot(const vector<double>& bestHistory, const vector<double>& meanHistory) {
    vector<double> iterations(bestHistory.size());
    iota(iterations.begin(), iterations.end(), 1.0);

    Plot2D plot;
    plot.xlabel("Итерация");
    plot.ylabel("Значение функции");
    plot.grid().show();

    plot.drawCurve(iterations, bestHistory)
        .label("Лучшее значение")
        .lineWidth(2);

    plot.drawCurve(iterations, meanHistory)
        .label("Среднее по популяции")
        .lineWidth(2);

    plot.legend()
        .atOutsideBottom()
        .displayHorizontal();

    Figure fig = {{plot}};
    Canvas canvas = {{fig}};
    canvas.size(1000, 700);
    canvas.show();
}

void showTrajectoryPlot(const vector<double>& x1History, const vector<double>& x2History, const vector<double>& x3History) {
    Plot3D plot;
    plot.xlabel("x1");
    plot.ylabel("x2");
    plot.zlabel("x3");

    plot.drawCurve(x1History, x2History, x3History)
        .label("Траектория")
        .lineWidth(2);

    plot.drawPoints(x1History, x2History, x3History)
        .label("Точки");

    vector<double> startX = {x1History.front()};
    vector<double> startY = {x2History.front()};
    vector<double> startZ = {x3History.front()};

    vector<double> endX = {x1History.back()};
    vector<double> endY = {x2History.back()};
    vector<double> endZ = {x3History.back()};

    plot.drawPoints(startX, startY, startZ)
        .label("Старт")
        .pointType(7)
        .pointSize(2)
        .lineColor("blue");

    plot.drawPoints(endX, endY, endZ)
        .label("Финиш")
        .pointType(7)
        .pointSize(2)
        .lineColor("red");

    Figure fig = {{plot}};
    Canvas canvas = {{fig}};
    canvas.size(1000, 700);
    canvas.show();
}

int main() {
    const int populationSize = 40;
    const int selectedParentsCount = 10;
    const int replacedWorstCellsCount = 6;
    const int maxIterations = 200;

    const int clonesPerParentCount = 8;
    const double mutationParameter = 0.3;

    random_device randomDevice;
    mt19937 generator(randomDevice());

    vector<Cell> population(populationSize);

    /// начальная популяция
    for (int cellIndex = 0; cellIndex < populationSize; cellIndex++) {
        population[cellIndex] = randomCell(generator);
    }

    for (int cellIndex = 0; cellIndex < populationSize; cellIndex++) {
        population[cellIndex].fitness = ackley(population[cellIndex].x);
    }

    vector<double> bestHistory;
    vector<double> meanHistory;

    vector<double> bestX1History;
    vector<double> bestX2History;
    vector<double> bestX3History;

    int iterationCounter = 0;

    while (true) {
        /// сортировка
        sort(population.begin(), population.end(), fitnessLess);

        /// выбор родителей
        vector<Cell> selectedParentCells(selectedParentsCount);
        for (int parentIndex = 0; parentIndex < selectedParentsCount; parentIndex++) {
            selectedParentCells[parentIndex] = population[parentIndex];
        }

        /// клонирование
        vector<vector<Cell>> clones(selectedParentsCount, vector<Cell>(clonesPerParentCount));
        for (int parentIndex = 0; parentIndex < selectedParentsCount; parentIndex++) {
            for (int cloneIndex = 0; cloneIndex < clonesPerParentCount; cloneIndex++) {
                clones[parentIndex][cloneIndex] = selectedParentCells[parentIndex];
            }
        }

        /// мутация
        vector<vector<Cell>> mutantClones(selectedParentsCount, vector<Cell>(clonesPerParentCount));
        for (int parentIndex = 0; parentIndex < selectedParentsCount; parentIndex++) {
            for (int cloneIndex = 0; cloneIndex < clonesPerParentCount; cloneIndex++) {
                mutantClones[parentIndex][cloneIndex] = mutateClone(
                    selectedParentCells[parentIndex],
                    clones[parentIndex][cloneIndex],
                    mutationParameter,
                    generator
                );
            }
        }

        for (int parentIndex = 0; parentIndex < selectedParentsCount; parentIndex++) {
            for (int cloneIndex = 0; cloneIndex < clonesPerParentCount; cloneIndex++) {
                mutantClones[parentIndex][cloneIndex].fitness = ackley(mutantClones[parentIndex][cloneIndex].x);
            }
        }

        vector<Cell> newPopulation = population;

        /// новая популяция: улучшенные родители + остальные
        for (int parentIndex = 0; parentIndex < selectedParentsCount; parentIndex++) {
            Cell bestMutantClone = mutantClones[parentIndex][0];

            for (int cloneIndex = 1; cloneIndex < clonesPerParentCount; cloneIndex++) {
                if (mutantClones[parentIndex][cloneIndex].fitness < bestMutantClone.fitness) {
                    bestMutantClone = mutantClones[parentIndex][cloneIndex];
                }
            }

            if (bestMutantClone.fitness < selectedParentCells[parentIndex].fitness) {
                newPopulation[parentIndex] = bestMutantClone;
                
            } else {
                newPopulation[parentIndex] = selectedParentCells[parentIndex];
            }
        }

        population = newPopulation;

        sort(population.begin(), population.end(), fitnessLess);

        /// обновление худших клеток (замена случайными)
        for (int replacedIndex = 0; replacedIndex < replacedWorstCellsCount; replacedIndex++) {
            int currentWorstCellIndex = populationSize - 1 - replacedIndex;
            population[currentWorstCellIndex] = randomCell(generator);
        }

        for (int replacedIndex = 0; replacedIndex < replacedWorstCellsCount; replacedIndex++) {
            int currentWorstCellIndex = populationSize - 1 - replacedIndex;
            population[currentWorstCellIndex].fitness = ackley(population[currentWorstCellIndex].x);
        }

        iterationCounter = iterationCounter + 1;

        sort(population.begin(), population.end(), fitnessLess);

        double meanFitness = 0.0;
        for (const Cell& currentCell : population) {
            meanFitness += currentCell.fitness;
        }
        meanFitness /= population.size();

        bestHistory.push_back(population.front().fitness);
        meanHistory.push_back(meanFitness);

        bestX1History.push_back(population.front().x[0]);
        bestX2History.push_back(population.front().x[1]);
        bestX3History.push_back(population.front().x[2]);

        if (iterationCounter == maxIterations) {
            break;
        }
    }

    sort(population.begin(), population.end(), fitnessLess);
    Cell bestCell = population.front();

    cout << fixed << setprecision(10);

    cout << "Найденное лучшее решение:\n";
    for (int coordinateIndex = 0; coordinateIndex < DIM; coordinateIndex++) {
        cout << "x[" << coordinateIndex + 1 << "] = " << bestCell.x[coordinateIndex] << "\n";
    }
    cout << "f(x) = " << bestCell.fitness << "\n\n";

    showConvergencePlot(bestHistory, meanHistory);
    showTrajectoryPlot(bestX1History, bestX2History, bestX3History);

    return 0;
}
