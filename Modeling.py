import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Starting from the advancement realised by EAD / PreProcessing
# This is the optimal shape of our dataset that we could get
# Every step of this cell was detailed in the other two scripts

dataset = pd.read_excel("dataset.xlsx")
feature_presence = dataset.isna().sum() / dataset.shape[0]
feature_continuos_category = list(dataset.columns[(feature_presence < 0.9) & (feature_presence > 0.88)])
feature_categorial_category = list(dataset.columns[(feature_presence < 0.8) & (feature_presence > 0.7)])
dataset = dataset[
    feature_continuos_category + ['Patient age quantile', "SARS-Cov-2 exam result"] + feature_categorial_category]
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=0)

convert = {
    'negative': 0,
    'not_detected': 0,
    'positive': 1,
    'detected': 1
}


def encodage(dataset):
    for col in dataset.select_dtypes("object"):
        dataset[col] = dataset[col].map(convert)
    return dataset


def feature_engineering(dataset):
    dataset['infected_by_any_type'] = dataset[feature_categorial_category].sum(axis=1) >= 1
    dataset = dataset.drop(feature_categorial_category, axis=1)
    return dataset


def imputation(dataset):
    dataset = dataset.dropna()
    return dataset


def preprocessing(dataset):
    dataset = encodage(dataset)
    dataset = feature_engineering(dataset)
    dataset = imputation(dataset)
    target = dataset["SARS-Cov-2 exam result"]
    dataset = dataset.drop("SARS-Cov-2 exam result", axis=1)
    return dataset, target


x_train, y_train = preprocessing(train_set)
x_test, y_test = preprocessing(test_set)


def evaluation(model):
    model.fit(x_train, y_train)
    y_predicted = model.predict(x_test)
    print(confusion_matrix(y_true=y_test, y_pred=y_predicted), classification_report(y_true=y_test, y_pred=y_predicted))
    N, train_score, val_score = learning_curve(model, x_train, y_train, cv=5, scoring='f1', random_state=0,
                                               train_sizes=np.linspace(0.1, 1, 10))
    plt.figure()
    plt.plot(N, train_score.mean(axis=1), label="Train Score")
    plt.plot(N, val_score.mean(axis=1), label="Validation Score")
    plt.legend()
