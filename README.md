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

The loss function measures the discrepancy between the prediction (<img src="/tex/a7d6723ffa9250d5cf6af8dbdc3a9aac.svg?invert_in_darkmode&sanitize=true" align=middle width=22.347882149999997pt height=33.333385800000016pt/>) and the desired output (<img src="/tex/deceeaf6940a8c7a5a02373728002b0f.svg?invert_in_darkmode&sanitize=true" align=middle width=8.649225749999989pt height=14.15524440000002pt/>). In other words, the loss function computes the error for a single training example.

The cost function is the average of the loss function of the entire training set. We are going to find the parameters <img src="/tex/31fae8b8b78ebe01cbfbe2fe53832624.svg?invert_in_darkmode&sanitize=true" align=middle width=12.210846449999991pt height=14.15524440000002pt/> and <img src="/tex/4bdc8d9bcfb35e1c9bfb51fc69687dfc.svg?invert_in_darkmode&sanitize=true" align=middle width=7.054796099999991pt height=22.831056599999986pt/> that minimize the overall cost function.

The loss function computes the error for a single training example; the cost function is the average of the loss functions of the entire training set.

<p align="center"><img src="/tex/38ca16bf77d69de7f6099740f09c4703.svg?invert_in_darkmode&sanitize=true" align=middle width=541.50746925pt height=44.89738935pt/></p>

# Logistic Regression Gradient Descent

The gradient descent algorithm is:

<p align="center"><img src="/tex/43f87f495e438d7c33b25b70a709382d.svg?invert_in_darkmode&sanitize=true" align=middle width=142.1551065pt height=34.7253258pt/></p>

<p align="center"><img src="/tex/e296d438b605ffd614b2be1f3617f1f1.svg?invert_in_darkmode&sanitize=true" align=middle width=131.84300745pt height=34.7253258pt/></p>

where: <img src="/tex/31fae8b8b78ebe01cbfbe2fe53832624.svg?invert_in_darkmode&sanitize=true" align=middle width=12.210846449999991pt height=14.15524440000002pt/> and <img src="/tex/4bdc8d9bcfb35e1c9bfb51fc69687dfc.svg?invert_in_darkmode&sanitize=true" align=middle width=7.054796099999991pt height=22.831056599999986pt/> represents the weights and the threshold, <img src="/tex/5fc6094a9c29537af5f99e0fceb76364.svg?invert_in_darkmode&sanitize=true" align=middle width=17.35165739999999pt height=14.15524440000002pt/> is the assigment ("update") math symbol and <img src="/tex/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode&sanitize=true" align=middle width=10.57650494999999pt height=14.15524440000002pt/> is the learning rate.

# Vectorization and Broadcasting in Python

Avoid explicit for-loops whenever possible: <img src="/tex/9ab4a87930a94cc65f504dfe8d3b5b1d.svg?invert_in_darkmode&sanitize=true" align=middle width=94.90634175pt height=27.6567522pt/> in Numpy (<img src="/tex/4a9c4f346bb2d551cb597a6c2be1c52b.svg?invert_in_darkmode&sanitize=true" align=middle width=144.2921667pt height=24.65753399999998pt/>  is about 300 times faster than the explicit for loop.

When we use the numpy version, python automatically transform the constant (or 1x1 matrix) <img src="/tex/4bdc8d9bcfb35e1c9bfb51fc69687dfc.svg?invert_in_darkmode&sanitize=true" align=middle width=7.054796099999991pt height=22.831056599999986pt/> and expand to a <img src="/tex/10ce1c06f2c58912b31fc827dc6e56fb.svg?invert_in_darkmode&sanitize=true" align=middle width=32.04729824999999pt height=21.18721440000001pt/> matrix to sum the matrices: <img src="/tex/6210e5f30e37a74c6bcbd15fca43c3c2.svg?invert_in_darkmode&sanitize=true" align=middle width=125.70581759999999pt height=24.65753399999998pt/>. This is called "broadcasting", it also faster way to compute the code.

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

This allows to write quite a flexible code, but it also allows to start creating product matrices that create bugs difficult to track. Specify always the matrix shape and don't use rank 1 matrices: <img src="/tex/567d08aae21c7a0094372ce9fe163f33.svg?invert_in_darkmode&sanitize=true" align=middle width=166.03755299999997pt height=24.65753399999998pt/> instead of <img src="/tex/81bbc24d782ec14a14e4c804233f2777.svg?invert_in_darkmode&sanitize=true" align=middle width=150.51246045pt height=24.65753399999998pt/> for a five column vector for example.
