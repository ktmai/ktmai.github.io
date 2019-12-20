---
layout: post
title:  "Label Propagation"
date:   2019-12-20 23:21:00
comments: true
---

Labelling data for supervised learning can be extremely time consuming. Often, it is more feasible to use predominantly unlabelled data complemented with some labelled data points for guidance. If we assume that data points which are close to each other have similar labels, we can use graph-based methods to classify the unlabelled data. 

Label propagation and label spreading are examples of such methods. Both methods consider the local structure (the distance between neighbouring datapoints) and the overall global shape of the dataset.

<!--more-->

1. TOC
{:toc}

## Graphs

_Graphs_ $G = (V,E)$ are a useful way of representing relationships between different objects. I will cover some aspects very briefly as a primer for notation used in the label propagation algorithm.

In its simplest form, graphs are comprised of _nodes_ $V$ (also known as _vertices_) and _edges_ $E$. The nodes represent the objects whilst the edges illustrate the relation between the different objects. Typically when drawn, circles are used for the nodes and lines are used for the edges.

<p align="center"> 
<img src="/assets/blog/lp/1.png">
</p>

If it is possible to get from one node in the graph to all of the other nodes in the graph using the edges, the graph is _connected_. The graph is _complete_ if there is an edge between every single node in the graph. The particular series of edges required to move from one node to another is the _path_.  If there does not exist a path between two nodes, then the graph is _disconnected_. 

<p align="center"> 
<img src="/assets/blog/lp/2.png">
</p>
<h6 align="center">Example of a complete graph. We can use the term $K_N$ for a complete graph with $N$ nodes, so this example would be a $K_4$ graph.</h6>

Furthermore, edges can have _directions_ (depicted by arrows) and _weights_ associated with them, which supplies further information about the relations between nodes, for example conditional probability relations or cost of travelling between two nodes.
 
<p align="center"> 
<img src="/assets/blog/lp/3.png">
</p>

## Semi-Supervised Learning

_Semi-supervised_ learning refers to a collection of machine learning techniques that use both labelled and unlabelled data for training. 

A semi-supervised problem can be characterised as follows: we have a dataset $X$ consisting of $l + u$ samples, where typically the number of labelled samples is much smaller than the number of unlabelled samples $l \ll u$. The first $l$ samples $X_L = \\{ x_1, x_2, ..., x_l \\}$ have a corresponding label $Y_{L} = \\{ y_1, y_2, ..., y_l \\}$ which can come from $C$ classes, all of which are represented in the labelled dataset. $X_U = \\{x_{l+1}, x_{l+2}, ..., x_{l+u} \\}$ denote the remaining samples and do not have a known label.

## Label Propagation

_Label propagation_ is a graph-based semi-supervised learning algorithm introduced by Zhu and Ghahramani in 2002. It takes a [transductive](https://ktmai.github.io/2019-11/transductive-svms/#induction-versus-transduction) approach and makes two assumptions about the data:
1. **Smoothness Assumption**: If two data points are close to each other, their labels are likely to be the same.
2. **Cluster Assumption**: If two data points are on the same cluster, they are likely to have the same label.

These two assumptions are local and global assumptions with regards to $X$ respectively. Note that this suggests label propagation is unlikely to perform well when dealing with high dimensional data or if the manifolds in which the data lies on is highly curved.

In the context of label propagation, we build a fully connected graph where all of the data points $X$ are nodes (i.e. $V = X$). 

In addition, we change the representation of the labels $Y$ to a soft format of dimension $ (l + u) \times C$. This means $Y_i$ represents the label probability distribution of $x_i$. We treat this as the initial estimates of the labels $\hat{Y}$. Due to the nature of label propagation, the estimates of the unlabelled data $\hat{Y}_U$ is irrelevant and can be set to zeroes. In other words, the initial estimate of the labels is $\hat{Y}^{(0)} = (\hat{Y}_L, \hat{Y}^{(0)}_U) = (y_1,..., y_l, 0, 0, ..., 0)$.

The edges $E$ are weighted to define the similarity between nodes, with higher weights illustrating higher levels of similarity. These edges can be summarised by a weight matrix $\mathbf{W}$, where $\mathbf{W}_{ij}$ denotes the weight between node $x_i$ and $x_j$. There are different ways to construct this weight matrix. Examples of implementations include:
1. K-nearest neighbours: $\mathbf{W}_{ij} = 1$ if and only if $x_i$ is a nearest neighbour of $x_j$ or 0 otherwise, or vice versa. A further constraint can be made so that both of them have to be nearest neighbours of each other.
2. Gaussian kernels: $\mathbf{W}_{ij} = e^{-\frac{\lvert\lvert x_i - x_j \rvert \rvert^2}{2 \sigma^2}}$, where $\sigma$ is a chosen hyperparameter.

As we are dealing with probability distributions in relation to the label estimates, we normalise $\mathbf{W}$ by multiplying it with the inverse of the diagonal degree matrix $\mathbf{D}$ where $\mathbf{D}_{ii} \leftarrow \sum_j W\_{ij}$ to create a probability transition matrix $\mathbf{T}$. Note that $\mathbf{D}$ is a matrix that contains the information about the number of edges attached to each node.

### Method

Using  the above setup, the labelled nodes are used to propagate information to all of the other nodes. Larger weights between nodes enables this propagation to occur more readily. As the label estimates $\hat{Y}$ are in a soft label format, we take the hard label of the unlabelled node to be the one with highest confidence.

The algorithm can be described as follows:
1. Compute $\mathbf{T}$ and initialise $\hat{Y}^{(0)}$ 
2. Until convergence of the label estimates ($\lvert\hat{Y}^{(t)} - \hat{Y}^{(t-1)}\rvert < \delta$ for a chosen $\delta$):
    1. Propagate the labels $\hat{Y}^{(t+1)} \leftarrow \mathbf{T}\hat{Y}^{(t)}$
    2. Row normalise $\hat{Y}$
    3. Clamp the known labelled nodes $\hat{Y}^{(t+1)}_L \leftarrow Y_L$

The clamping step is required as we want the values from the known labelled nodes to be providing consistent information.

As we would like to ensure the cluster and smoothness assumption are adhered to, the cost function can be generalised as follows (Chapelle 2006):

$$C(\hat{Y}) = \lvert\lvert \hat{Y}_L - Y_L \rvert\vert^2 + \mu\hat{Y}^\intercal L \hat{Y} + \mu\epsilon \lvert\lvert \hat{Y} \rvert\rvert ^2$$

Where $L = \mathbf{D} - \mathbf{W}$ is the graph Laplacian. The first term ensures the estimate of the known labels is consistent with the ground truth, the second is transductive and incorporates the smoothness assumption by penalising rapid changes in $\hat{Y}$ that are close, and the final term is a regularisation term used in degenerate situations, for example when the graph $G$ has a connected component without a labelled sample.

An example implementation of label propagation can be found [here](https://github.com/ktmai/code-demonstrations/blob/master/Label%20Propagation/Label_Propagation.ipynb).

### Label Spreading

_Label spreading_ is a similar algorithm introduced by Zhou in 2004. However instead of using $\mathbf{D}^{-1}\mathbf{W}$ for propagation, it uses the normalised graph Laplacian $\mathcal{L} = \mathbf{D}^{-1/2}\mathbf{W}\mathbf{D}^{-1/2}$.

## Label Propagation in Deep Learning

With increasing interest in training deep learning architectures with less labelled data, multiple papers incorporating label propagation in deep learning have emerged. I have briefly summarised some below:

### Neural Graph Machines
The authors observed that the typical objective function used for neural networks is not transductive and does not exploit the nature of unlabelled data. 

As a solution, neural graph machines combine neural networks and label propagation by using an objective function inspired by label propagation to train a neural network. Notabily, the objective function considers both labelled and unlabelled data points. Hidden representations are used instead of network outputs in the objective to ensure datapoints within a graph representation have similar predictions.

### Transductive Propagation Networks
This paper is motivated by the task of _few-shot learning_, which aims to learn a classifier that is trained with a small number of training examples per class. Few-shot learning tasks typically have a support set (known labelled examples used for training the neural network) and a query set (unseen examples used for testing). 

This paper takes a transductive approach by including the query set in the optimisation objective. More specifically, the architecture involves transforming the input data into a feature embedding using a convolutional neural network, constructing a graph using the union of the support and query set, and conducting label propagation on this graph. The loss is computed with respect to the embedding and the graph. Finally, the parameters of the architecture are updated using backpropagation.

### Label Propagation for Deep Semi-Supervised Learning
This paper also takes a transductive approach for a semi-supervised task. Initially, a neural network is used to train the labelled samples. Following this step, an iterative process is initiated. A nearest neighbour graph of the entire training set is computed and label spreading is conducted to generate pseudo-labels for the unlabelled samples. This newly labelled dataset (including both labelled and pseudo-labelled samples) is then used to retrain the neural network. To account for possible imbalanced classes, the pseudo-labels are weighted to provide a measure of uncertainty with regards to the estimated labels.

## References

[1] Zhu, X. and Ghahramani Z. Learning from Labeled and Unlabeled Data with Label Propagation. 2002. 

[2] Zhou, D. et al. Learning with Local and Global Consistency. 2004.

[3] Chapelle, O. et al. Semi-Supervised Learning. 2006. MIT Press.

[4] Bui, T.D. et al. Neural Graph Machines: Learning Neural Networks using Graphs. 2017.

[5] Liu, Y. et al. Learning to Propagate Labels: Transductive Propagation Network for Few Shot Learning. 2019.

[6] Iscen, A. et al. Label Propagation for Deep Semi-Supervised Learning. 2019.