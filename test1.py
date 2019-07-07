import numpy as np

X = np.random.normal(loc=1, scale=10, size=(1000, 50))#матрица с элементами нормального распределениян
#scale - стандартное отклонение нормального распределения
#loc - среднее нормального распределения
m = np.mean(X, axis=0)#axis=1 по строкам, axis=0 по столбцам
std = np.std(X, axis=0)
X_norm = ((X - m)  / std)
#print (X_norm)

Z = np.array([[4, 5, 0],
             [1, 9, 3],
             [5, 1, 1],
             [3, 3, 3],
             [9, 9, 9],
             [4, 7, 1]])
Z_sum = np.sum(Z, axis = 1)
result = np.nonzero(Z_sum>10)

#Функция для генерации единичной матрицы: np.eye
#Функция для вертикальной стыковки матриц: np.vstack((A, B))
A = np.eye(3)
B = np.eye(3)
AB = np.vstack((A, B))
print (AB)
