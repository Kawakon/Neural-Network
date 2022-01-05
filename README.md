# Neural Network Layer Program

This short program uses the Eigen C++ library to emulate the operation of a neural network, where it takes 
a 1x2 input matrix and produces a 1x2 output matrix. There are three layers within the neural network that manipulate 
the input matrix. Matrix multiplication is performed on the input and weight matrices of each layer, and a bias matrix is added 
to the result. An activation function h(x) is applied to the result of the addition to obtain the input to the next layer 
of the neural network. In this code, the activation function is the sigmoid function: