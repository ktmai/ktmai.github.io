---
layout: post
title:  "Transductive SVMs"
date:   2019-11-11 10:00:00
comments: true
---

An overview on the concept of transduction, support vector machines, and a variant -  transductive support vector machines.

<!--more-->

1. TOC
{:toc}


## Primer

### Support Vector Machines

Let's assume we have a dataset $\mathbf{X}$ consisting of $n$ samples, where each item $x^i$ in $\mathbf{X}$ is paired with a class $y^i$ ($y^i$ is correspondingly part of $\mathbf{Y}$). To simplify the problem, we assume that only a positive and a negative class exist, denoted as $1$ and $-1$ respectively, i.e. $y^i \in \\{-1, 1\\}$. A _support vector machine_ ('SVM') is an algorithm that classifies data by dividing the classes using a _hyperplane_. 

What exactly does this mean? We can assume that the items belonging to the negative class appear more similar to each other compared to points belonging to the positive class:

<p align="center"> 
<img src="/assets/blog/tsvm/1.png">
</p>
<h6 align="center">Simple example of a dataset where it can be seen that features of the same class (denoted by colour) are grouped more closely together.</h6>

A hyperplane is essentially a decision boundary that separates the two classes. Mathematically we can describe the hyperplane as a linear function of the points $\mathbf{w}^T\mathbf{x}+b$.  There may be more than one way to divide the classes. Ideally we would like this decision boundary to be as far away from both of the two classes as possible. This would mean that if we encounter more samples, we would have more certainty about what class the new samples belong to.

<p align="center"> 
<img src="/assets/blog/tsvm/2.png">
</p>
<h6 align="center">The dashed line represents potential hyperplanes that can divide the two classes.</h6>


To achieve this, we can examine the borderline points in each class ($\mathbf{x}^+$ and $\mathbf{x}^-$) so that the distance between $\mathbf{x}^+$ and $\mathbf{x}^-$ is maximised through the decision boundary. These borderline points are also known as _support vectors_.  As we are relying on these support vectors, the data can be rescaled such that $\mathbf{x}^+$ lies on $\mathbf{w}^T\mathbf{x}^+ +b = 1$ and $\mathbf{x}^-$ lies on $\mathbf{w}^T\mathbf{x}^-+b = -1$. Again, due to scaling, the distance from the support vectors to the decision boundary can be expressed arbitrarily as $\gamma^+ = \frac{\mathbf{w}^T\mathbf{x}^+ + b}{\lvert\lvert\mathbf{w}\rvert\rvert}$ and $\gamma^- = \frac{\mathbf{w}^T\mathbf{x}^- + b}{\lvert\lvert\mathbf{w}\rvert\rvert}$. Therefore we can express the geometric distance between $\mathbf{x}^+$ and $\mathbf{x}^-$ as:

$$\gamma^+ - \gamma^- = \frac{2}{\lvert\lvert\mathbf{w}\rvert\rvert}$$ 

The above expression suggests that in order to maximise the margin between the support vectors, we should find the smallest value of $\mathbf{w}$. Therefore we can write the SVM objective as follows:

$$    \underset{\textbf{w}}{\textrm{min}}\;\lvert\lvert\mathbf{w}\rvert\rvert^2 \;\; \textrm{s.t.} \;\;\forall i \;\;y^i(\textbf{w}^T\textbf{x}^i + b) \;\geq 1 $$

This expression is nice because it is convex, meaning we can find a global minimum!

If the dataset is not entirely linearly separable, we may not be able to find a solution that satisfies the SVM objective above. In this case, we can still find a useful classifier by relaxing the constraint through introduction of a _slack variable_ $\xi^i$ for each data point. The slack variable enables some points to be within the margin or even misclassified. As we would like to minimise the number of misclassified points, we want to minimise the use of slack variables. Therefore, $\sum_{i=1}^n \xi^i$ can be viewed as the upper bound for the number of errors and the SVM objective changes to:

$$
\begin{align*}    
\underset{\textbf{w}}{\textrm{min}}\;\lvert\lvert\mathbf{w}\rvert\rvert^2 + C\sum_{i=1}^n \xi^i \;\; \textrm{s.t.} \;\;&\forall i \;\;y^i(\textbf{w}^T\textbf{x}^i + b) \;\geq 1 - \xi^i\\ 
& \forall i \;\; \xi^i \geq 0
\end{align*}$$

where $C$ is a hyperparameter that controls the strictness of the constraint.

Finally, combining the two constraints above is equivalent to rewriting the slack variable as $\xi^i = \textrm{max}(0, 1 - y^i((\textbf{w}^T\textbf{x}^i + b))$, which is known as a _hinge loss_. This means we can express the SVM objective in terms of an unconstrained optimisation objective:

$$\underset{\textbf{w}}{\textrm{min}}\lvert\lvert\mathbf{w}\rvert\rvert^2 + C\sum^n_{i=1} \textrm{max}(0, 1 - y^i(\textbf{w}^T\textbf{x}^i + b))$$

### Training Sets and Test Sets

When fitting a model, we would like it to cater to the true distribution of the dataset as closely as possible. In practice however, we only have a sample of the dataset. The sample data may not necessarily represent the population data as it may contain biases, so fitting a model perfectly to the sample data is unlikely to generalise well to unseen data. 

One method to counteract this is to partition the sample data into a training set and a test set. The training set can be used to fit the parameters of the model whilst the test set can be used to provide an indication of the fitted model's performance on unseen data.

Let us consider the dataset $\mathbf{X}$ again, assuming it is only a selected sample of the population data. Instead of considering $\mathbf{X}$ in its entirety, we partition it into a training set of size $m$: $\mathbf{X}\_{train} \subset \mathbf{X}$ with labels $\mathbf{Y}\_{train} \subset \mathbf{Y}$. Correspondingly the test set is the complement of the training set and is of size $l = n - m$: $\mathbf{X}_\{test} = \mathbf{X} \setminus \mathbf{X}\_{train}, \mathbf{Y}\_{test} = \mathbf{Y} \setminus \mathbf{Y}\_{train}$. We fit the model using $\mathbf{X}\_{train}, \mathbf{Y}\_{train}$ and evaluate this on $\mathbf{X}\_{test}$ to generate a set of predictions for the labels which we will denote as $\mathbf{Y}^*\_{test}$.

Note that this can be further extended through partitioning the dataset into three parts to also incorporate a validation set which can be used to select the model's hyperparameters.

### Induction versus Transduction
[Wikipedia](https://en.wikipedia.org/wiki/Transduction_(machine_learning)) attempts to distinguish between the two by stating that _inductive inference_ uses the training samples to learn general characteristics about the dataset whilst _transductive inference_ uses the training samples to learn about the characteristics of the test set only. Vapnik (who introduced transductive inference to the machine learning community) also described this concept as the following: inductive inference results in a two step process (moving from the particular to the general and back to the particular, in this case the test set) whereas transductive inference skips this general step. As a consequence, the inductive setup may be too complex for what we are looking to address.

In contrast to the setting in the previous section where we only use $\mathbf{X}\_{train}$ and $\mathbf{Y}\_{train}$ to fit the model, we will also incorporate $\mathbf{X}\_{test}$ to generate predicted $\mathbf{Y}^*_{test}$. Notably the labels from the test set are not used in model fitting.

Through this method, we can study the location of test samples and consequently encode prior knowledge about the distribution of $\mathbf{Y}$, meaning the number of samples required for training can potentially be reduced.

<p align="center"> 
<img src="/assets/blog/tsvm/3.png">
</p>
<h6 align="center">In this diagram, the coloured datapoints are the elements belonging to the training set with the rest belonging to the test set. The dashed line is the inductive solution whilst the solid line is the transductive solution. We can see in this example the transductive method considers the location of the test samples to make a decision about the classifier that is more appropriate.</h6>

## Model

To extend the standard SVM to the transductive SVM ('TSVM'), we need to add additional constraints. In the setting with no slack variables, this can be expressed as:

$$
\begin{align*}
    \underset{\textbf{w}}{\textrm{min}}\;\lvert\lvert\mathbf{w}\rvert\rvert^2 \;\; \textrm{s.t.} \;\;& \;\;\forall i = 1,...,m  \;\; y^i_{train}(\textbf{w}^T\textbf{x}^i_{train} + b) \;\geq 1 \\
    & \;\; \forall j = 1,...,l \;\; y^{*j}_{test}(\textbf{w}^T
    \textbf{x}^{*j}_{test} + b) \;\geq 1 
\end{align*}$$

Whereas in the setting with slack variables, it can be expressed as:

$$
\begin{align*}
\underset{\textbf{w}}{\textrm{min}}\;\lvert\lvert\mathbf{w}\rvert\rvert^2 + C\sum_{i=1}^m \xi^i + C^*\sum_{j=1}^l 
\xi^{*j} \;\; \textrm{s.t.} \;\;& \forall i = 1,...,m \;\; y^i(\textbf{w}^T\textbf{x}^i_{train} + b) \;\geq 1 - \xi^i\\
& \forall i = 1,...,m \;\; \xi^i \geq 0, \\
& \forall j = 1,...,l \;\; y^{*j}_{test}
(\textbf{w}^T\textbf{x}^{*j}_{test} + b) \;\geq 1 - 
\xi^{*j}\\
& \forall j = 1,...,l \;\; \xi^{*j} \geq 0, \\
\end{align*}
$$

where again $C$ and $C^*$ are hyperparameters.

This can be expressed in an unconstrained form as follows:

$$\underset{\textbf{w}}{\textrm{min}}\;\lvert\lvert\mathbf{w}\rvert\rvert^2 + C\sum_{i=1}^m \textrm{max}(0, 1 - y^i_{train}(\textbf{w}^T\textbf{x}^i_{train} + b)) + C^*\sum_{j=1}^l 
\textrm{max}(0, 1 - |\textbf{w}^T\textbf{x}^j_{test} + b|)$$

The reason for using the absolute function for the hinge loss relating to the unlabelled data is because we assume that the label for the unlabelled sample is $y^{*j}_{test} = \textrm{sign}(\textbf{w}^T\textbf{x}^j\_{test})$.

The objective functions are no longer convex, meaning analytical methods are no longer suitable for finding solutions and other optimisation algorithms are required.


## Implementation

An example notebook illustrating toy implementations of SVM and TSVM can be found [here](https://github.com/ktmai/code-demonstrations/blob/master/TSVM/Transductive_SVM.ipynb).


## References
\[1\] Vapnik, V. Statistical Learning Theory. 1998. Wiley & Sons.

\[2\] Zisserman, A. [Machine Learning Lectures](http://www.robots.ox.ac.uk/~az/lectures/ml/index.html). 2015.

\[3\] Chapelle, O. et al. Semi-Supervised Learning. 2006. MIT Press. 

\[4\] Joachim, T. Transductive Inference for Text Classification using Support Vector Machines. 1999. 

\[5\] Collobert, R. et al. Large Scale Transductive SVMs. 2006. JMLR.
