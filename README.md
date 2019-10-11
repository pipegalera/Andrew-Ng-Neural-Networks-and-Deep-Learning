# What is a Neural Network?

Think about this function of housing prices as very simple neural network:

<p align="center">
<img src="images/nn.png">
</p>

The little circle is a single neuron of a neural network, that implements the function (ReLU) that is drawn on the left. All the neuron does is it inputs the size, computes this linear function, takes a max of zero, and then outputs the estimated price.

A larger neural network is formated by taking many of the single neurons and stacking them together. As well, each of the little circles implements the Rectified Linear Unit function (or some other slightly non-linear function). By stacking together a few of the single neurons or simple predictors, we have now a a larger neural network.

<p align="center">
<img src="images/nn2.png">
</p>

To feed the network, you need to give it just the input x and the output y for a number of examples in your training set and the neural network will figure it out by itself the hidden layer in the middle. The input layer and the hidden layer are density connected: every input feature is connected to every "hidden" feature.

<p align="center">
<img src="images/nn3.png">
</p>

# Supervised Learning with Neural Networks

There are different types of neural network, for example Convolution Neural Network (CNN) used often for image application and Recurrent Neural Network (RNN) used for one-dimensional sequence data such as translating English to Chinses or a temporal component such as text transcript. As for the autonomous driving, it is a hybrid neural network architecture.

<p align="center">
<img src="images/supervised_types.png">
</p>

Structured data refers to things that has a defined meaning such as price, age whereas unstructured data refers to thing like pixel, raw audio, text.
