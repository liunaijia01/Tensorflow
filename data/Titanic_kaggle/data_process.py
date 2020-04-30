import os
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline


def load_data():
    cur_path = os.path.abspath(os.path.dirname(__file__))
    train_file_path = os.path.join(cur_path, "train.csv")
    features = pd.read_csv(train_file_path)
    labels = features.pop("Survived")
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=1)

    col_cat = ['Sex', 'Embarked']
    col_num = ['Age', 'SibSp', 'Parch', 'Fare']
    x_train_cat = x_train[col_cat]
    x_train_num = x_train[col_num]
    x_test_cat = x_test[col_cat]
    x_test_num = x_test[col_num]

    scaler_cat = make_pipeline(SimpleImputer(strategy='constant'), OneHotEncoder())
    x_train_cat_enc = scaler_cat.fit_transform(x_train_cat)
    x_test_cat_enc = scaler_cat.transform(x_test_cat)

    scaler_num = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler())
    x_train_num_scaled = scaler_num.fit_transform(x_train_num)
    x_test_num_scaled = scaler_num.transform(x_test_num)

    x_train_scaled = sparse.hstack((x_train_cat_enc, sparse.csr_matrix(x_train_num_scaled)))
    x_test_scaled = sparse.hstack((x_test_cat_enc, sparse.csr_matrix(x_test_num_scaled)))

    x_train = x_train_scaled.toarray()
    x_test = x_test_scaled.toarray()
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    load_data()