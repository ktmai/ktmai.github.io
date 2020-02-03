---
layout: post
title:  "Co-Training"
date:   2020-02-02 14:00:00
comments: true
---

Co-training is a transductive semi-supervised learning algorithm which relies on multiple 'views' of the data, whereby each view provides different information about the data. A separate classifier is trained on each view. An added benefit of this method is that the classifiers may be able to consider important features that may have been disregarded if the views were pooled together and only one classifier were used.

<!--more-->

Many models used for classification can be categorised as _discriminative models_. If we have a set of samples $x^i \in X$ each paired with a ground truth label $y^i \in Y$, this means that such a model tries to learn the function $f(x): X \rightarrow Y$. The set of all possible functions that the classifier can choose from which fulfil this is known as the _hypothesis space_ $\mathcal{H}$. Ideally, we would like to pick a function that best generalises to unseen test data. However, the set of potential functions may be very large. Co-training aims to reduce the size of $\mathcal{H}$. 

Suppose we have two views of the data. In the context of co-training, we can define the sample space as $X = X_1 \times X_2$, where $X_1$ and $X_2$ correspond to the two different views. This means each datapoint can be written as a pair $x^i = (x^i_1, x^i_2)$. 

Defining a _concept class_ $C$ as the set of true mapping functions, $C_1$ and $C_2$ can be defined as the concept classes over $X_1$ and $X_2$ respectively. If we have target functions corresponding to each view $f_1 \in C_1$ and $f_2 \in C_2$, co-training assumes we have the relation $f(x^i) = f_1(x^i_1) = f_2(x^i_2) = y^i$. In other words, there is no such sample $x^i = (x^i_1, x^i_2)$ such that $f_1(x^i_1) \neq f_2(x^i_2)$. Finally, as we are considering a semi-supervised setting, we assume $X$ comprises of labelled samples $L$ and unlabelled samples $U$. 

The above relation means that co-training assumes that the views agree with each other. Furthermore, to ensure that each view can improve the other, we assume:
1. **Self sufficiency**: Each view used by itself can learn a reasonable classifier.
2. **Conditional independence**: Each view contains complementary information.

In practice, the above assumptions are rather strict and natural views satifying the above may not always exist. However, Nigam and Ghani showed that co-training algorithms which manufactured a reasonable split outperformed algorithms that did not use a split. They speculated that compared to other methods, it may be due to co-training being more robust in instances where the assumptions of the underlying classifiers had been violated. Conversely, there is not a definitive way to decide on the most reasonable split of data and the way to choose these splits are not clear cut.

The co-training algorithm itself is agnostic to the choice of classifier, although the original paper uses a Naive Bayes classifier for their experiments. Given $L$ and $U$, we create a subset of $U$ denoted as $U'$. The following is then iterated until convergence:
1. Use $L$ to learn a classifier $h_1$ using $x_1$ portion of $x$ only.
2. Use $L$ to learn a classifier $h_2$ using $x_2$ portion of $x$ only.
3. Allow $h_1$ and $h_2$ to label a fixed number of positive and negative examples from $U'$.
4. Add the predicted labels from above to $L$.
5. Replenish $U'$ by sampling randomly from $U$.

An implementation of the co-training paper can be found [here](https://github.com/ktmai/code-demonstrations/blob/master/Co-Training/Co_Training.ipynb).

## Label Propagation Perspective

Wang and Zhou showed that the above approach can also be viewed as a type of [label propagation](https://ktmai.github.io/2019-12/label-propagation/). They stated that assigning a label to an unlabelled instance $x^j_v$ (where $v = 1, 2$) 
based on a labelled sample $x^k_v$ can be seen as estimating the conditional probability $p(y(x^j_v) = y(x^k_v)| x^j_v, x^k_v)$. Therefore, if we have two labelled samples $x^q_v, x^w_v$   with the same label, we can set $p(y(x^q_v) = y(x^w_v)| x^q_v, x^w_v) = 1$, otherwise $p(y(x^q_v) = y(x^w_v)| x^q_v, x^w_v) = 0$. 

We then create a weight matrix $\mathbf{W}_v$ corresponding to each view, where $\mathbf{W}_v^{ij}$ corresponds to 
$p(y(x^i_v) = y(x^j_v)| x^i_v, x^j_v)$. The weight matrices are normalised and label propagation can be conducted as standard.


## References
[1] Blum, A. and Mitchell, T. Combining Labeled and Unlabeled Data with Co-Training. 1998.

[2] Chapelle, O. et al. Semi-Supervised Learning. 2006. MIT Press.

[3] Zhu, X. and Goldberg, A.B. Introduction to Semi-Supervised Learning. 2009. Morgan & Claypool.

[4] Nigam, K. and Ghani, R. Analyzing the Effectiveness and Applicability of Co-training. 2000.

[5] Wang, W. and Zhou, Z.H. A New Analysis of Co-Training. 2010.