import pandas as pd
from sklearn.linear_model import LinearRegression

# from __future__ import print_function
print(pd.__version__)

train = pd.read_csv('train.tsv', delimiter='\t', header=None)
test = pd.read_csv('test.tsv', delimiter='\t', header=None)

print(train.head())
# print(test.head())
print(len(train.columns))
x = train.iloc[:, 0:(len(train.columns) - 1)]
last = train.iloc[:, -1]
y_train = last.to_numpy()
# print (y_train)
# print ('dffsaaaa')
x_train = x.to_numpy()
# print (x_train)
x_test = test.to_numpy()

model = LinearRegression()
model.fit(x_train, y_train)
# print(model)
predicted = model.predict(x_test)
# print(predicted)


with open("output.tsv", "w") as f:
    for i in range(0, len(predicted)):
        print(predicted[i], file=f)
