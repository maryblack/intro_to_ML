import pandas as pd

sap = pd.read_csv('sap.csv', delimiter='\t', encoding='utf-8')
print(sap.head())
