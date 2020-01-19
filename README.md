# What is a Neural Network?

Think about this function of housing prices as very simple neural network:

<p align="center">
<img src="images/nn.png" width="60%" height="60%">
</p>

The little circle is a a representation of a single "**neuron**" of a neural network. The neuron takes an input, the size of the house, to try to explain the variance in an output, the price of the house. The neuron uses an algorithm to process the input into output, in this case the activation function in the graph called Rectified Linear Unit function (ReLU). A **Neural Network** is created by stacking neurons together. Each of the neurons stacked implements ReLU, or some other slightly non-linear function.

<p align="center">
<img src="images/nn2.png" width="60%" height="60%">
</p>

The Neural Network needs a large number of inputs and their output, in the example a large dataset of houses prices associated with their size. The "**hidden layers**" are made by the neurons itself.
The first layer, the **input layer**, and the hidden layers are connected, as every input feature is connected to every "hidden" feature.

<p align="center">
<img src="images/nn3.png" width="60%" height="60%">
</p>

# Supervised Learning with Neural Networks

There are different types of neural networks, for example Convolution Neural Network (CNN) used often for image application. Recurrent Neural Networks (RNN) are used for one-dimensional sequence data such as translating English to Chinese or a temporal component such as text transcript. Hybrid Neural Networks architecture can be used for autonomous driving model.

<p align="center">
<img src="images/supervised_types.png" width="60%" height="60%">
</p>

- Structured data: it has a defined meaning, such as price and size. 
- Unstructured data: it does not a define meaning by itself, like a pixel, raw audio, text.

# Binary Classification and Logistic Regression

In a binary classification problem, the result is a discrete value output.

The feature matrix shape is made "stacking" the number of features (<img src="/tex/322d8f61a96f4dd07a0c599482268dfe.svg?invert_in_darkmode&sanitize=true" align=middle width=17.32124954999999pt height=14.15524440000002pt/>) in different columns, one for every observation (<img src="/tex/0e51a2dede42189d77627c4d742822c3.svg?invert_in_darkmode&sanitize=true" align=middle width=14.433101099999991pt height=14.15524440000002pt/>): <img src="/tex/3ba00a479d4b9167a41bad118af23b8c.svg?invert_in_darkmode&sanitize=true" align=middle width=134.9372904pt height=24.65753399999998pt/>. The output shape is a 1 by <img src="/tex/0e51a2dede42189d77627c4d742822c3.svg?invert_in_darkmode&sanitize=true" align=middle width=14.433101099999991pt height=14.15524440000002pt/> dimensional matrix; <img src="/tex/afebc7a1bbc7e178ce750acd5e17539a.svg?invert_in_darkmode&sanitize=true" align=middle width=119.66713934999999pt height=24.65753399999998pt/>

Logistic regression is used for binary classification, when the output labels are either 0 or 1: <img src="/tex/8d1721d6e5ba5edb77d9504fcb1338c3.svg?invert_in_darkmode&sanitize=true" align=middle width=108.9362934pt height=24.65753399999998pt/>, where <img src="/tex/323ce571e0e93da4e5376548a6837dc7.svg?invert_in_darkmode&sanitize=true" align=middle width=68.92287929999999pt height=22.831056599999986pt/>.

The parameters used in Logistic regression are:

* The input features vector: <img src="/tex/438e62960e8497e2f053bc64f80ca0a7.svg?invert_in_darkmode&sanitize=true" align=middle width=56.72753459999999pt height=22.465723500000017pt/>, where <img src="/tex/f97e4290cfc54a698ac3de94d2b49538.svg?invert_in_darkmode&sanitize=true" align=middle width=17.32124954999999pt height=14.15524440000002pt/> is the number of features
* The training label: <img src="/tex/6a1114fa49613587b6a98ce2227cdeee.svg?invert_in_darkmode&sanitize=true" align=middle width=52.48464539999999pt height=21.18721440000001pt/>
* The weights: <img src="/tex/5930ae0b04c2747333f98d30adac3d0f.svg?invert_in_darkmode&sanitize=true" align=middle width=59.54339159999999pt height=22.465723500000017pt/>
* The threshold: <img src="/tex/ed47273cf1f7a79141a98ca7e868fe51.svg?invert_in_darkmode&sanitize=true" align=middle width=7.054796099999991pt height=22.831056599999986pt/>
* The output: <img src="/tex/3725898a78dcc79fd49640d4845c1438.svg?invert_in_darkmode&sanitize=true" align=middle width=112.44255989999998pt height=27.6567522pt/>
* The Sigmoid function: <img src="/tex/f469c5de7f8a3a028f42399365add185.svg?invert_in_darkmode&sanitize=true" align=middle width=93.59773169999998pt height=27.77565449999998pt/> where <img src="/tex/222750d021b4c146842ab8584e655d07.svg?invert_in_darkmode&sanitize=true" align=middle width=89.39266379999998pt height=27.6567522pt/>

# Logistic Regression Loss and Cost Function

The **Loss function** measures the discrepancy between the prediction (<img src="/tex/5a92eb88b1ce29767bb5287374cf8fbd.svg?invert_in_darkmode&sanitize=true" align=middle width=8.21920935pt height=14.15524440000002pt/>) and the desired output (<img src="/tex/deceeaf6940a8c7a5a02373728002b0f.svg?invert_in_darkmode&sanitize=true" align=middle width=8.649225749999989pt height=14.15524440000002pt/>). In other words, the loss function computes the error for a single training example.

The **Cost function** is the average of the loss function of the entire training set. We are going to find the parameters <img src="/tex/31fae8b8b78ebe01cbfbe2fe53832624.svg?invert_in_darkmode&sanitize=true" align=middle width=12.210846449999991pt height=14.15524440000002pt/> and <img src="/tex/4bdc8d9bcfb35e1c9bfb51fc69687dfc.svg?invert_in_darkmode&sanitize=true" align=middle width=7.054796099999991pt height=22.831056599999986pt/> that minimize the overall cost function.

The loss function computes the error for a single training example; the cost function is the average of the loss functions of the entire training set.

<p align="center"><img src="/tex/38ca16bf77d69de7f6099740f09c4703.svg?invert_in_darkmode&sanitize=true" align=middle width=541.50746925pt height=44.89738935pt/></p>

# Logistic Regression Gradient Descent

The gradient descent algorithm is:

<p align="center"><img src="/tex/43f87f495e438d7c33b25b70a709382d.svg?invert_in_darkmode&sanitize=true" align=middle width=142.1551065pt height=34.7253258pt/></p>

<p align="center"><img src="/tex/e296d438b605ffd614b2be1f3617f1f1.svg?invert_in_darkmode&sanitize=true" align=middle width=131.84300745pt height=34.7253258pt/></p>

where: <img src="/tex/31fae8b8b78ebe01cbfbe2fe53832624.svg?invert_in_darkmode&sanitize=true" align=middle width=12.210846449999991pt height=14.15524440000002pt/> and <img src="/tex/4bdc8d9bcfb35e1c9bfb51fc69687dfc.svg?invert_in_darkmode&sanitize=true" align=middle width=7.054796099999991pt height=22.831056599999986pt/> represents the weights and the threshold, <img src="/tex/5fc6094a9c29537af5f99e0fceb76364.svg?invert_in_darkmode&sanitize=true" align=middle width=17.35165739999999pt height=14.15524440000002pt/> is the assignment ("update") math symbol and <img src="/tex/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode&sanitize=true" align=middle width=10.57650494999999pt height=14.15524440000002pt/> is the learning rate.

# Vectorization and Broadcasting in Python

Avoid explicit for-loops whenever possible. Using  the numpy version ()"<img src="/tex/4a9c4f346bb2d551cb597a6c2be1c52b.svg?invert_in_darkmode&sanitize=true" align=middle width=144.2921667pt height=24.65753399999998pt/>") of "<img src="/tex/9ab4a87930a94cc65f504dfe8d3b5b1d.svg?invert_in_darkmode&sanitize=true" align=middle width=94.90634175pt height=27.6567522pt/>" is about 300 times faster than an explicit for loop.

When we use the numpy version, python automatically transform the constant (or 1x1 matrix) "b" and expand to a "1xm" matrix to sum the matrices: <img src="/tex/6210e5f30e37a74c6bcbd15fca43c3c2.svg?invert_in_darkmode&sanitize=true" align=middle width=125.70581759999999pt height=24.65753399999998pt/>. This is called "broadcasting", its also faster way to compute the code.

- Example: Calculating the percentage of calories from carb/protein/fat for each food — without fooloop from the following table

<p align="center">
<img src="images/food.png" width="60%" height="60%">
</p>

In Python would be:

```python
import numpy as np
A = np.array([[56, 0, 4.4,68],
              [1.2,104,52,8],
              [1.8,135,99,0.9]
              ])
print(A)
cal = A.sum(axis=0) # axis=0 is to make python sum vertically, axis=1 would make the sum horizontally.
print(cal)
percentage = 100*A/cal.reshape(1,4) #Taking the 3x4 matrix "A" and diving it by the 1x4 matrix "cal".
print(percentage)
```
Resulting in a 3x4 matrix.

- Python broadcast (or "force") matrices to make the operation match.

As a "General Principle": When you sum, subtract, divide or multiply (m,n) matrix with a (1,n), matrix the (1,n) matrix will be expanded to a (m,n) matrix by copying the row m times, to match the shape.

For example, a 4x1 matrix plus a number would treat the number as a 4x1 matrix with each row the number. A 2x3 matrix plus a 1x3 matrix would treat the last as a 2x3 matrix creating a row with the same numbers and so forth.

```python
a = np.random.randn(2, 3) # a.shape = (2, 3)
b = np.random.randn(2, 1) # b.shape = (2, 1)
c = a + b
c.shape # (2,3)

a = np.random.randn(3, 3)
b = np.random.randn(3, 1)
c = a*b
c.shape # (3,3)

d = np.random.randn(4, 3) # a.shape = (4, 3)
e = np.random.randn(3, 2) # b.shape = (3, 2)
f = d*e # Error: operands could not be broadcast together
```

This allows to write quite a flexible code, but it also allows to start creating product matrices that create bugs difficult to track. Specify always the matrix shape and don't use rank 1 matrices: "np.random.randn(5,1)" instead of "np.random.randn(5)" for a five column vector, for example.

# Neural Networks Overview and Vectorized Implementation

<p align="center">
<img src="images/leyers.png" width="60%" height="60%">
</p>

What a Neural Network does is doing the logistic regression for each neuron. This logistic regression has 2 steps of computation: it's own regression <img src="/tex/284725ac5e6b6f357803a43ef12682ee.svg?invert_in_darkmode&sanitize=true" align=middle width=79.99767599999998pt height=27.6567522pt/> and an activation function <img src="/tex/05c003d60a6e5e2b03dc6e29ddbcf5a8.svg?invert_in_darkmode&sanitize=true" align=middle width=61.74271949999999pt height=24.65753399999998pt/>

<p align="center">
<img src="images/activation.png" width="60%" height="60%">
</p>

So for each neuron <img src="/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/> or node in the layer you have: <img src="/tex/31755903d960cb832e6310355ea7a1dd.svg?invert_in_darkmode&sanitize=true" align=middle width=127.52654969999998pt height=34.337843099999986pt/> and <img src="/tex/36da5c31f64ac5ba1ab6dbab12055de5.svg?invert_in_darkmode&sanitize=true" align=middle width=79.63859475pt height=24.65753399999998pt/> where n is the layer number:

<p align="center">
<img src="images/formula.png" width="70%" height="70%">
</p>

In this Neural Network example with 2 layers and 4 logistic regression, we can stack the vectors together of the entire layers to make a vectorized implementation:

- <img src="/tex/e99a4e1d357aa81fae3ff5e34777a7d3.svg?invert_in_darkmode&sanitize=true" align=middle width=31.80377474999999pt height=29.190975000000005pt/> as a vector (4x3) of <img src="/tex/a7f37b9c641adf6695dd9f803fb4961e.svg?invert_in_darkmode&sanitize=true" align=middle width=161.55281339999996pt height=34.337843099999986pt/>;
- <img src="/tex/39c7d8201e2cadb69c40aa59b2b65d48.svg?invert_in_darkmode&sanitize=true" align=middle width=21.05031389999999pt height=29.190975000000005pt/> as a vector (4x1) of <img src="/tex/ea0a6219d3589a549b215cdb2d5c9801.svg?invert_in_darkmode&sanitize=true" align=middle width=118.53896834999999pt height=34.337843099999986pt/>;
- <img src="/tex/f4fca034d9054d2cee867c6dfd0d21d5.svg?invert_in_darkmode&sanitize=true" align=middle width=26.392786199999986pt height=29.190975000000005pt/> as the vector dot product of <img src="/tex/717159c72c244b22ef8030a0a19390f4.svg?invert_in_darkmode&sanitize=true" align=middle width=83.16217304999998pt height=29.190975000000005pt/>;
- <img src="/tex/b9feec4b9dbf507e05a4da3aee588d61.svg?invert_in_darkmode&sanitize=true" align=middle width=26.324316149999987pt height=29.190975000000005pt/> as a vector (4x1) of <img src="/tex/9adad71e414b40cbbeba1ae7ee4b70ea.svg?invert_in_darkmode&sanitize=true" align=middle width=192.94738485pt height=34.337843099999986pt/>.

# Activation functions

When you build your neural network, one of the choices you get to make is what activation function to use in the hidden layers.

The sigmoid function goes within zero and one. An activation function that almost always works better than the sigmoid function is the tangent function or also called hyperbolic tangent function (Tanh):

<p align="center">
<img src="images/tangent.png" width="70%" height="70%">
</p>

It goes between 1 and - 1. The tanh function is almost always strictly superior. The one exception is for the output layer because if y is either 0 or 1, then it makes sense for y hat to be a number, the one to output that's between 0 and 1 rather than between minus 1 and 1.

One of the downsides of both the sigmoid function and the tanh function is that if z is either very large or very small, then the gradient or the derivative or the slope of this function becomes very small.

Another choice that is very popular in machine learning is what's called the rectify linear unit (ReLU). So the value function looks like:

<p align="center">
<img src="images/relu.png" width="70%" height="70%">
</p>

The derivative is 1 as z is positive. And the derivative or  slope is 0, when z is negative. Is an increasingly default choice of activation function.

In practice, using the ReLU activation function, your neural network will often learn much faster than Tanh or Sigmoid activation function. The main reason is that there is less of these effects of the slope of the function going to 0, which slows down learning.

If you use a linear activation function, or alternatively if you don't have an activation function, then no matter how many layers your neural network has, all it's doing is just computing a linear activation function. A linear hidden layer is more or less useless because the composition of two linear functions is itself a linear function.

# Derivatives of activation functions

* Sigmond Function: <img src="/tex/f77e969353c9a7729e03b69c6c14b698.svg?invert_in_darkmode&sanitize=true" align=middle width=267.81727664999994pt height=24.7161288pt/>
* Tanh Function: <img src="/tex/93d9f210c58d318efad18058bbe48d5f.svg?invert_in_darkmode&sanitize=true" align=middle width=225.1692366pt height=26.76175259999998pt/>

Where <img src="/tex/53ee452a5c1c62644cbbda5f322a11cc.svg?invert_in_darkmode&sanitize=true" align=middle width=51.822576299999994pt height=24.65753399999998pt/> is a nonlinear function.

* ReLU Function: <img src="/tex/34723e8bb818e0f73e261ceafe24b692.svg?invert_in_darkmode&sanitize=true" align=middle width=142.03955865pt height=24.7161288pt/> and <img src="/tex/d879ba2ba361fe33298bf4435ff89222.svg?invert_in_darkmode&sanitize=true" align=middle width=162.13087604999998pt height=24.7161288pt/>
* Leaky ReLU Function: <img src="/tex/20fc333f7572589f986a3031f5d25eea.svg?invert_in_darkmode&sanitize=true" align=middle width=170.35008539999998pt height=24.7161288pt/>
and <img src="/tex/92270b0c287e23e140cb3b8308e4739d.svg?invert_in_darkmode&sanitize=true" align=middle width=154.82499284999997pt height=24.7161288pt/>

# Gradient descent for Neural Networks

The gradient is a numeric calculation allowing us to know how to adjust the parameters (weights, bias) of a network in such a way that its output deviation (cost function) is minimized:

<img src="/tex/1ebaa02d11f35a12286c34747697f370.svg?invert_in_darkmode&sanitize=true" align=middle width=281.5793574pt height=29.190975000000005pt/>

# Random Initialization

When you change your neural network, it's important to initialize the weights randomly. Random initialization of the model is a common practice. The rational behind is that from wherever we start, if we are perseverant enough and through an iterative learning process, we can reach the pseudo-ideal model.

We can initialize the weights randomly and the bias at zero.

If we initialize both weights and bias to zeros, all activations functions (a_{i,j,...}) will be equal. All the neurons units would start off computing the same function and therefore all the hidden neurons completely identical. Again, using hidden layers would be useless.
