// eigenproj.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <Eigen/Dense>

using namespace std;
using Eigen::MatrixXd;

/**
 * Each element of the product matrix A1[i][j] is computed from a unique row and
 * column of the factor matrices, X[i][k], W1[k][j], and B1[i][j]
 */

 // Matrix size constants.
constexpr int M = 1;
constexpr int N = 2;
constexpr int P = 3;

//data
float X_d[M][N] = { { 1.0, 0.5 } };
float W1_d[N][P] = { {0.5, 0.3, 0.5}, {0.2, 0.4, 0.6} };
float B1_d[M][P] = { { 0.1, 0.2, 0.3 } };

// variable for timing task execution
__int64 timed;

double activationFunction(double x) {
    // sigmoid activation function
    double sigresult = 1 / (1 + exp(x));
    return sigresult;
}

int main()
{
    // creating matrices for neural network
    MatrixXd input(1, 2);
    MatrixXd weight(2, 3);
    MatrixXd bias(1, 3);

    // initializing matrices with data
    input(0, 0) = 1.0; input(0, 1) = 0.5;
    weight(0, 0) = 0.5; weight(0, 1) = 0.3; weight(0, 2) = 0.5; weight(1, 0) = 0.2; weight(1, 1) = 0.4; weight(1, 2) = 0.6;
    bias(0, 0) = 0.1; bias(0, 1) = 0.2; bias(0, 2) = 0.3;

    // performing matrix multiplication
    MatrixXd a = (input * weight) + bias;
    MatrixXd result = a.unaryExpr(&activationFunction);

    cout << result << endl;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
