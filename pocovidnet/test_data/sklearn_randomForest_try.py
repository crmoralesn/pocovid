from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data_base_train = pd.read_csv("test_data/_test_model_general_train.csv", sep=';')
data_base_train = data_base_train.replace(['covid', 'pneumonia', 'regular'], [0, 1, 2])

data_base_test = pd.read_csv("test_data/_test_model_general_test.csv", sep=';')
data_base_test = data_base_test.replace(['covid', 'pneumonia', 'regular'], [0, 1, 2])
#X = X.drop(columns=[449,451])

X_train=data_base_train.drop(data_base_train.columns[[448,450]], axis=1)
Y_train = X_train.iloc[:,-1]

X_test=data_base_test.drop(data_base_test.columns[[448,450]], axis=1)
Y_test = X_test.iloc[:,-1]

#X, y = make_classification(n_samples=1000, n_features=4,n_informative=2, n_redundant=0,random_state=0, shuffle=False)
clf = RandomForestClassifier(max_depth=2, random_state=0, n_estimators=50)
clf.fit(X_train, Y_train)

# Make predictions for the test set
y_pred_test = clf.predict(X_test)

# View accuracy score
val = accuracy_score(Y_test, y_pred_test)
print(val)

# View confusion matrix for test data and predictions
val = confusion_matrix(Y_test, y_pred_test)
print(val)
#random forest clasification
#get metrics
