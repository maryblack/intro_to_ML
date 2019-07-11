import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge

def one_hot_enc(data_train, data_test):
    enc = DictVectorizer()
    X_train_categ = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
    X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))
    return X_train_categ, X_test_categ

def preprocessing_text(data):
    description = data['FullDescription']
    description = description.replace('[^a-zA-Z0-9]', ' ', regex = True)
    description = [text.lower() for text in description]
    return description

def del_nan(data):
    loc = data['LocationNormalized']
    time = data['ContractTime']
    loc.fillna('nan', inplace=True)
    time.fillna('nan', inplace=True)

def preprocessing(data_train, data_test):
    train_descr = preprocessing_text(data_train)
    test_descr = preprocessing_text(data_test)
    vectorizer = TfidfVectorizer(min_df=5)
    D_train = vectorizer.fit_transform(train_descr)
    D_test = vectorizer.transform(test_descr)
    del_nan(data_train)
    del_nan(data_test)
    X_train_categ, X_test_categ = one_hot_enc(data_train, data_test)
    X_train = hstack([D_train,X_train_categ]).toarray()
    X_test = hstack([D_test,X_test_categ]).toarray()
    y_train = data_train['SalaryNormalized']
    return X_train, y_train, X_test


def main():
    train = pd.read_csv('salary-train.csv')
    test = pd.read_csv('salary-test-mini.csv')
    X_train, y_train, X_test = preprocessing(train, test)
    model = Ridge(alpha=1)
    model.fit(X_train, y_train)
    print(model.predict(X_test))


if __name__ == '__main__':
    main()
