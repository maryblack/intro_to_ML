import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.svm import SVC

newsgroups = datasets.fetch_20newsgroups(
    subset='all',
    categories=['alt.atheism', 'sci.space']
)

y = newsgroups.target
list_text = newsgroups.data

tf_vectorizer = TfidfVectorizer()
X = tf_vectorizer.fit_transform(list_text)
feature_names = tf_vectorizer.get_feature_names()
# print (len(feature_names), feature_names[-10:])

kf = KFold(n_splits=5, random_state=241, shuffle=True)
kf.get_n_splits(X, y)
'''
C_array = []
h = 0.00001
for i in range(0,11):
    C_array.append(h)
    h *=10

max_a = 0
max_c = 0
for c_v in C_array:
    clf = SVC(random_state=241, C=c_v, kernel='linear')
    clf.fit(X, y)
    y_pred = clf.predict(X)
    m_a = cross_val_score(clf, X, y, cv=5)
    a = m_a.mean()
    print(a)
    if a>max_a:
        max_a = a
        max_c = c_v
        
print(max_a, max_c)


grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(X, y)

for a in gs.grid_scores_:
    print (a.mean_validation_score, a.parameters)'''

clf = SVC(random_state=241, C=10, kernel='linear')
clf.fit(X, y)
y_pred = clf.predict(X)
svm_coef = clf.coef_.toarray()
svm_abs = [abs(el) for el in svm_coef]
i = np.argsort(svm_abs)
feature_sorted = i[0]
top10 = feature_sorted[-10:]
answer = [feature_names[i] for i in top10]
answer.sort()
print("%s" % ",".join(answer))

print('done')
