import pandas as pd
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score



fulldata = pd.read_csv("data/indian_pines.csv").to_numpy()
X = fulldata[:,1:]
y = fulldata[:,0]

acc = []
k = []
for i in range(20):
    train_x, test_x, train_y, test_y = model_selection.train_test_split(X,y, test_size=0.95, shuffle=True)
    svc = SVC(C=1e5, kernel='rbf', gamma=1.)
    svc.fit(train_x, train_y)
    y_pred = svc.predict(test_x)
    accuracy = accuracy_score(test_y, y_pred)
    kappa = cohen_kappa_score(test_y, y_pred)
    print(accuracy, kappa)
    acc.append(accuracy)
    k.append(kappa)

print(sum(acc)/len(acc))
print(sum(k)/len(k))


