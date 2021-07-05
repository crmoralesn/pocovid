import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

#obtenido desde
#https://www.datacamp.com/community/tutorials/naive-bayes-scikit-learn


def naive(name):
    fileName = "test_data/"+name
    data_base = pd.read_csv(fileName, sep=',')
    data_base = data_base.replace(['covid', 'pneumonia', 'regular'], [0, 1, 2])

    # filtering data
    dbtrain = data_base.loc[data_base['66']=='train']
    dbtest = data_base.loc[data_base['66']=='test']

    X_train = dbtrain.drop(dbtrain.columns[[-3,-1]], axis=1)
    Y_train = X_train.iloc[:,-1]


    X_test = dbtest.drop(dbtest.columns[[-3,-1]], axis=1)
    Y_test = X_test.iloc[:,-1]

    #Create a Gaussian Classifier
    gnb = GaussianNB()

    #Train the model using the training sets
    gnb.fit(X_train, Y_train)

    #Predict the response for test dataset
    y_pred = gnb.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    print(name)
    print("Accuracy:" ,metrics.accuracy_score(Y_test, y_pred))
    print(metrics.classification_report(Y_test, y_pred))

    print(metrics.confusion_matrix(Y_test, y_pred))

naive("test_model_0_no_filter.csv")
naive("test_model_1_2d_filter.csv")
naive("test_model_2_blur_filter.csv")
naive("test_model_3_mediablur_filter.csv")
naive("test_model_4_laplacian_filter.csv")
naive("test_model_5_sobel_filter.csv")
naive("test_model_6_equalizeHist_filter.csv")


