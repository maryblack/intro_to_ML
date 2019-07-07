import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr

data = pd.read_csv('titanic.csv', index_col='PassengerId')
#задание нумерации строк - index_col
'''print(data.head())
print( data.columns.values)
print (data [ 'Sex' ] . value_counts ())
print (data [ 'Pclass' ] . value_counts ('f'))
print (data [ 'Age' ] . mean ())
print (data [ 'Age' ] . median ())
print (pearsonr(data['SibSp'], data['Parch']))'''
Name = data.loc[data['Sex'] == 'female'] ['Name']
Name = Name.to_numpy()
#print(Name)

splitted_data = [elem.split(' ') for elem in Name]
print (splitted_data)
select = [elem[i+1] for elem in splitted_data for i in range (0, len(elem)) if ('Miss' in elem[i] or 'Mrs' in elem[i])
          and 'Mr' not in elem[i]]
print (len(select))
print (max(select, key = select.count))

