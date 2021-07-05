from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.svm import SVC

data_base_train = pd.read_csv("test_data/test_model_1_2d_filter.csv", sep=',')
data_base_train = data_base_train.replace(['covid', 'pneumonia', 'regular'], [0, 1, 2])

X_train=data_base_train.drop(data_base_train.columns[[-3,-1]], axis=1)
Y_train = X_train.iloc[:,-1]

#cross_val_score(RandomForestClassifier(n_estimators=5),X_train, Y_train)

Score = cross_val_score(SVC(gamma='auto'), X_train, Y_train)
print(Score)
np.average(Score)