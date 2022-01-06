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
    double sigresult = 1 / (1 + exp(-x));
    return sigresult;
}

int main()
{

    // creating matrices Layer 1 for neural network
    MatrixXd input_1(1, 2);
    MatrixXd weight_1(2, 3);
    MatrixXd bias_1(1, 3);

    // initializing Layer 1 matrices with data
    input_1(0, 0) = 1.0; input_1(0, 1) = 0.5;
    weight_1(0, 0) = 0.5; weight_1(0, 1) = 0.3; weight_1(0, 2) = 0.5; weight_1(1, 0) = 0.2; weight_1(1, 1) = 0.4; weight_1(1, 2) = 0.6;
    bias_1(0, 0) = 0.1; bias_1(0, 1) = 0.2; bias_1(0, 2) = 0.3;

    // performing Layer 1 matrix multiplication and applying activation function
    MatrixXd a_1 = (input_1 * weight_1) + bias_1;
    MatrixXd result_1 = a_1.unaryExpr(&activationFunction);

    cout << "Layer 1 = " << result_1 << endl;

    // creating matrices Layer 2 for neural network
    MatrixXd weight_2(3, 2);
    MatrixXd bias_2(1, 2);

    // initializing Layer 2 matrices with data
    weight_2(0, 0) = 0.1; weight_2(0, 1) = 0.4; weight_2(1, 0) = 0.2; weight_2(1, 1) = 0.5; weight_2(2, 0) = 0.3; weight_2(2, 1) = 0.6;
    bias_2(0, 0) = 0.1; bias_2(0, 1) = 0.2;

    // performing Layer 2 matrix multiplication and applying activation function
    MatrixXd a_2 = (result_1 * weight_2) + bias_2;
    MatrixXd result_2 = a_2.unaryExpr(&activationFunction);

    cout << "Layer 2 = " << result_2 << endl;

    // creating matrices Layer 3 for neural network
    MatrixXd weight_3(2, 2);
    MatrixXd bias_3(1, 2);

    // initializing Layer 3 matrices with data
    weight_3(0, 0) = 0.1; weight_3(0, 1) = 0.3; weight_3(1, 0) = 0.2; weight_3(1, 1) = 0.4;
    bias_3(0, 0) = 0.1; bias_3(0, 1) = 0.2;

    // performing Layer 3 matrix multiplication and applying activation function
    MatrixXd a_3 = (result_2 * weight_3) + bias_3;
    MatrixXd result = a_2.unaryExpr(&activationFunction);

    cout << "Layer 3 = " << result << endl;

}

