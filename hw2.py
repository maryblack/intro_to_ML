import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('titanic.csv', index_col = 'PassengerId')
#f = data['Survived']
s = {'male' : 1, 'female' : 0}
data['Sex'] = data['Sex'].map(s)
df = data[['Pclass', 'Fare', 'Age', 'Sex','Survived']]

df = df.to_numpy()
splitted = df.astype(float)

cleared = [x for x in splitted if True not in np.isnan(x)]
print(len(splitted),len(cleared))

y = [row[-1] for row in cleared]
X = np.delete(cleared,np.s_[-1],1)
#print (X)
clf = DecisionTreeClassifier()
clf.fit(X,y)
importances = clf.feature_importances_
print (importances)
