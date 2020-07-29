import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn import utils

df = pd.read_csv('/home/mitch/Documents/sc_winners/matchups.csv')

df2 = df.loc[:, 'Outcome':]
df2['Series W%'] = (df2['Series W%'] * 100000).round()

dels = ['Diff T', 'Diff OL', 'Diff W', 'Diff L', 'Diff PTS', 'Diff TG/G', 'Diff PP', 'Diff PPO', 'Diff PPA', 'Diff PPOA', 'Diff SO']

for col in dels:
	del df2[col]


X = df2.loc[:, 'Diff AvAge':].values
y = df2.loc[:, 'Outcome'].values


# le = LabelEncoder()
# y2 = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


lr = linear_model.LogisticRegression()
model = lr.fit(X_train, y_train)
pred = model.predict(X_test)
acc = accuracy_score(pred, y_test)

adj = []
adj_test = []
'''
for i in pred:
	if i > 50000:
		adj.append(1)
	else:
		adj.append(0)

for j in y_test:
	if j > 50000:
		adj_test.append(1)
	else:
		adj_test.append(0)
'''
adj_acc = accuracy_score(adj, adj_test)

print(adj)
print(adj_test)
print(acc)
print(adj_acc)
print(pred)
print(y_test)



