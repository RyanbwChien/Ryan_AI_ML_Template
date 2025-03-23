# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 00:59:43 2025

@author: user
"""

from sklearn.datasets import load_iris
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor

iris = load_iris()

iris.data
iris.feature_names
iris.target
iris.target_names

ISO = IsolationForest()

ISO.fit(iris.data, iris.target)
ISO.predict(iris.data)
