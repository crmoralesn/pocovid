from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.svm import SVC

data_base = pd.read_csv("test_data/test_model_2_blur_filter.csv", sep=',')
data_base = data_base.replace(['covid', 'pneumonia', 'regular'], [0, 1, 2])

# filtering data
dbtrain = data_base.loc[data_base['66']=='train']
dbtest = data_base.loc[data_base['66']=='test']

X_train = dbtrain.drop(dbtrain.columns[[-3,-1]], axis=1)
Y_train = X_train.iloc[:,-1]


X_test = dbtest.drop(dbtest.columns[[-3,-1]], axis=1)
Y_test = X_test.iloc[:,-1]

#cross_val_score(RandomForestClassifier(n_estimators=5),X_train, Y_train)

svm = SVC(gamma='auto')
svm.fit(X_train, Y_train)
scoresvm =  svm.score(X_test, Y_test)


rf = RandomForestClassifier(n_estimators=40)
rf.fit(X_train, Y_train)
scorerf = rf.score(X_test, Y_test)

lr = LogisticRegression(solver='liblinear',multi_class='ovr')
lr.fit(X_train, Y_train)
scorelg = lr.score(X_test, Y_test)

print(scoresvm,'SVM')
print(scorerf,'RF')
print(scorelg,'LG')
