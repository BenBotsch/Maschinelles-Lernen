# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 09:42:05 2023

@author: botsch
"""

from collections import Counter

import numpy as np


def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])


class Node:
    """
    A class that represents a node in a decision tree.
    """
    def __init__(self, 
        feature=None, 
        threshold=None, 
        left=None, 
        right=None, *, 
        value=None
    ):
        """
        Constructs a Node object.

        Parameters:
        -----------
        feature : int
            The feature is used for splitting at the node.
        threshold : float
            The threshold value used for splitting the 
            feature at the node.
        left : Node object
            The left child node of the current node.
        right : Node object
            The right child node of the current node.
        value : float
            The predicted value for the node. Only 
            available for leaf nodes.
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    """
    A decision tree algorithm for supervised learning tasks.
    """
    def __init__(self, 
                 min_samples_split=2, 
                 max_depth=50, 
                 n_feature=None):
        """
        Initialize a DecisionTree object.
        Parameters:
        ----------
        min_samples_split : int
            The minimum number of samples required 
            to split an internal node.
        max_depth : int
            The maximum depth of the decision tree.
        n_feature : int or None
            The maximum number of features to consider 
            when splitting a node.
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feature
        self.root = None

    def fit(self, X, y):
        """
        Train the decision tree on the input dataset X 
        and target variable y.
        Parameters:
        ----------
        X : array-like of shape (n_samples, n_features)
            The input dataset for training the decision tree.
        y : array-like of shape (n_samples,)
            The target variable for training the decision tree.
        """
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        """
        Predict the target variable for the input dataset 
        X using the trained decision tree.
        Parameters:
        ----------
        X : array-like of shape (n_samples, n_features)
            The input dataset for predicting the target 
            variable using the decision tree.
        Returns:
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted target variable for the 
            input dataset X.
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth=0):
        """
        Build a decision tree by recursively splitting 
        the input data.
        Parameters:
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        y : array-like of shape (n_samples,)
            The target values.
        depth : int
            The current depth of the tree.
        Returns:
        -------
        node : Node
            The root node of the decision tree.
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping criteria
        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        # greedily select the best split according to information gain
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)

        # grow the children that result from the split
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_criteria(self, X, y, feat_idxs):
        """
        Selects the best split criterion for the 
        Decision Tree by greedily selecting the 
        feature and threshold that result in the 
        highest information gain.
        Parameters:
        ----------
            X : array-like of shape (n_samples, n_features)
                The feature matrix.
            y : array-like of shape (n_samples,)
                The target vector.
            feat_idxs : list 
                The indices of the features to consider 
                for splitting.
        Returns:
        ----------
            tuple: 
                The index of the best feature and the 
                best threshold value.
        """
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        """
        Computes the information gain for a given split.
        Parameters:
        ----------
            y : array-like
                The target vector of shape (n_samples,).
            X_column : array-like
                A single column of the feature matrix X.
            split_thresh : float
                The threshold to use for splitting.
        Returns:
        ----------
            float: 
                The information gain for the given split.
        """
        # parent loss
        parent_entropy = entropy(y)

        # generate split
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # compute the weighted avg. of the loss for the children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # information gain is difference in loss before vs. after split
        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common


if __name__ == "__main__":
    # Imports
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    clf = DecisionTree(max_depth=10)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy(y_test, y_pred)

    print("Accuracy:", acc)