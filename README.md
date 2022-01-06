# Neural Network Layer Program

This short program uses the Eigen C++ library to emulate the operation of a neural network, where it takes 
a 1x2 input matrix and produces a 1x2 output matrix. There are three layers within the neural network that manipulate 
the input matrix. Matrix multiplication is performed on the input and weight matrices of each layer, and a bias matrix is added 
to the result. An activation function h(x) is applied to the result of the addition to obtain the input to the next layer 
of the neural network. In this code, the activation function is the sigmoid function:

![image](https://user-images.githubusercontent.com/43174428/148300924-47c72f43-e304-440c-a53d-7580ece948d3.png)

The neural network input matrix X is shown below:

![image](https://user-images.githubusercontent.com/43174428/148300675-50213597-3257-44fa-990a-3ef91d81883c.png)

The weight (W1) and bias (B1) matrices of the first layer are:

![image](https://user-images.githubusercontent.com/43174428/148301275-2b871b77-82ed-4127-ab68-7d7110913f68.png)

The weight (W2) and bias (B2) matrices of the second layer are:

![image](https://user-images.githubusercontent.com/43174428/148301348-9aae71f8-e848-406d-9142-38ff5a3678b6.png)

The weight (W3) and bias (B3) matrices of the third layer are:

![image](https://user-images.githubusercontent.com/43174428/148301425-0af67805-6549-4017-9473-8aaf55e7781c.png)
