# Neural Network Layer Program

This short program uses the Eigen C++ library to emulate the operation of a neural network layer, where it takes 
a 1x2 matrix input, a 2x3 weight matrix, and a 1x3 bias matrix. Matrix multiplication is performed on the input and 
weight matrices, and the bias matrix is added to the result. An activation function h(x) is applied to the result of the 
addition to obtain the input to the next layer of the neural network. In this code, the activation function is the sigmoid
function: