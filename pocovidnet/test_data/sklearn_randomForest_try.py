import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

ap = argparse.ArgumentParser()

ap.add_argument('-m', '--csv_name', type=str, default='test_model_general.csv')
ap.add_argument('-f', '--filters', type=int, default=1)

args = vars(ap.parse_args())

csv_name = args['csv_name']
filters = args['filters']

path_image_index=filters*64
test_train_type_index=(filters*64)+2

data_base = pd.read_csv("test_data/"+csv_name, sep=';')

data_base_train = data_base[data_base.train == 'train']
data_base_train = data_base_train.replace(['covid', 'pneumonia', 'regular'], [0, 1, 2])
X_train=data_base_train.drop(data_base_train.columns[[path_image_index,test_train_type_index]], axis=1)
Y_train = X_train.iloc[:,-1]

data_base_test = data_base[data_base.train == 'test']
data_base_test = data_base_test.replace(['covid', 'pneumonia', 'regular'], [0, 1, 2])
X_test=data_base_test.drop(data_base_test.columns[[path_image_index,test_train_type_index]], axis=1)
Y_test = X_test.iloc[:,-1]

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

