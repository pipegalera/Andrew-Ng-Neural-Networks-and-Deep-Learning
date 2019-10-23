# What is a Neural Network?

Think about this function of housing prices as very simple neural network:

<p align="center">
<img src="images/nn.png" width="60%" height="60%">
</p>

The little circle is a single neuron of a neural network, that implements the function (ReLU) that is drawn on the left. All the neuron does is it inputs the size, computes this linear function, takes a max of zero, and then outputs the estimated price.

A larger neural network is formated by taking many of the single neurons and stacking them together. As well, each of the little circles implements the Rectified Linear Unit function (or some other slightly non-linear function). By stacking together a few of the single neurons or simple predictors, we have now a a larger neural network.

<p align="center">
<img src="images/nn2.png" width="60%" height="60%">
</p>

To feed the network, you need to give it just the input x and the output y for a number of examples in your training set and the neural network will figure it out by itself the hidden layer in the middle. The input layer and the hidden layer are density connected: every input feature is connected to every "hidden" feature.

<p align="center">
<img src="images/nn3.png" width="60%" height="60%">
</p>

# Supervised Learning with Neural Networks

There are different types of neural network, for example Convolution Neural Network (CNN) used often for image application and Recurrent Neural Network (RNN) used for one-dimensional sequence data such as translating English to Chinses or a temporal component such as text transcript. As for the autonomous driving, it is a hybrid neural network architecture.

<p align="center">
<img src="images/supervised_types.png" width="60%" height="60%">
</p>

Structured data refers to things that has a defined meaning such as price, age whereas unstructured data refers to thing like pixel, raw audio, text.

# Binary Classification and Logistic Regression

In a binary classification problem, the result is a discrete value output.

The feature matrix shape is made "stacking" the number of features ($n_x$) in different columns, one for every observation ($m$): $X.shape = (n_x, m)$. The output shape is a 1 by $m$ dimensional matrix; $y.shape = (1,m)$

Logistic regression is used for binary classification, when the output labels are either 0 or 1: $\hat{y} = P(y=1|x)$, where $0 \leq \hat{y} \leq 1$.

The parameters used in Logistic regression are:

* The input features vector: $x\in{R^{n_{x}}}$, where $n_{x}$ is the number of features
* The training label: $y\in0,1$
* The weights: $w\in{R^{n_{x}}}$
* The threshold: $b\inR$
* The output: $\hat{y}= \sigma(w^{T}x + b)$
* The Sigmoid function: $\sigma(z)=\frac{1}{1+e^{-z}}$ where $z=w^{T}x + b$

# Logistic Regression Loss and Cost Function

The loss function measures the discrepancy between the prediction ($\hat(y)$) and the desired output ($y$). In other words, the loss function computes the error for a single training example.

The cost function is the average of the loss function of the entire training set. We are going to find the parameters $w$ and $b$ that minimize the overall cost function.

The loss function computes the error for a single training example; the cost function is the average of the loss functions of the entire training set.

\begin{align*}
    J(w,b)=\frac{1}{m}\sum\limits_{i=1}^{m}L(\hat{y}^{(i)}, y^{(i)})=-\frac{1}{m}\sum\limits_{i=1}^{m}(y^{(i)}log(\hat{y}^{(i)})+(1-y^{(i)})log(1-y^{(i)})
\end{align*}

# Logistic Regression Gradient Descent

The gradient descent algorithm is:

\begin{align*}
  w := w - \alpha\frac{dJ(w,b)}{dw}
\end{align*}

\begin{align*}
  b := b - \alpha\frac{dJ(w,b)}{db}
\end{align*}

where: $w$ and $b$ represents the weights and the threshold, $:=$ is the assigment ("update") math symbol and $\alpha$ is the learning rate.

# Vectorization and Broadcasting in Python

Avoid explicit for-loops whenever possible: $z = w^{T}X + b$ in Numpy ($z = np.dot(w, x) + b$  is about 300 times faster than the explicit for loop.

When we use the numpy version, python automatically transform the constant (or 1x1 matrix) $b$ and expand to a $1xm$ matrix to sum the matrices: $b = [b_1, b_2, b_3...b_n]$. This is called "broadcasting", it also faster way to compute the code.

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
Resulting:

[[94.91525424  0.          2.83140283 88.42652796]
 [ 2.03389831 43.51464435 33.46203346 10.40312094]
 [ 3.05084746 56.48535565 63.70656371  1.17035111]]

The 94.92 % of the Apple is Carbs, the 2% is Protein and so on...

Python will broadcast (or "force") matrices to make the operation match.

- General Principle: When you sum, subtract, divide or multiply (m,n) matrix with a (1,n), matrix the (1,n) matrix will be expanded to a (m,n) matrix by copying the row m times, to match the shape.

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

This allows to write quite a flexible code, but it also allows to start creating product matrices that create bugs difficult to track. Specify always the matrix shape and don't use rank 1 matrices: $np.random.randn(5,1)$ instead of $np.random.randn(5)$ for a five column vector for example.
