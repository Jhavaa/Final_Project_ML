import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from mpl_toolkits.mplot3d import Axes3D
from credit_data import small_X_train_scaled_std, small_y_train_scaled , X_train,y_train , X_test,y_test ,X_train_std


cov = np.cov(small_X_train_scaled_std.T)
eigenvals, eigenvectorss = np.linalg.eig(cov)
model = LogisticRegression(C=1, random_state=1, solver='liblinear', multi_class='ovr')
model.fit(small_X_train_scaled_std,small_y_train_scaled)
y_predit = model.predict(small_X_train_scaled_std)
print(accuracy_score(small_y_train_scaled,y_predit))


tot = sum(eigenvals)
var_exp = [(i / tot) for i in sorted(eigenvals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
plt.bar(range(1, len(var_exp)+1), var_exp, alpha=0.5, align='center',label='Individual explained variance')
plt.step(range(1, len(var_exp)+1), cum_var_exp, where='mid',label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
# plt.savefig('images/05_02.png', dpi=300)
plt.show()

# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigenvals[i]), eigenvectorss[:, i]) for i in range(len(eigenvals))]
# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda k: k[0], reverse=True)
w = np.hstack((eigen_pairs[0][1][:, np.newaxis],eigen_pairs[1][1][:, np.newaxis],eigen_pairs[2][1][:, np.newaxis]))
#print('Matrix W:\n', w)


#print(X_train_std[0].dot(w))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X_train_pca = small_X_train_scaled_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

for l, c, m in zip(np.unique(y_train), colors, markers):
    ax.scatter(X_train_pca[y_train == l, 0],
                X_train_pca[y_train == l, 1],
               X_train_pca[y_train == l, 2],
                c=c, label=l, marker=m)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('images/05_03.png', dpi=300)
plt.show()


modelSVM  = SVC(kernel='rbf', C=1.0, random_state=1)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(small_X_train_scaled_std)
X_test_pca = pca.transform()
modelSVM.fit(X_train_pca,y_train)
print(modelSVM.score(X_test_pca,y_test))





#-----------




for l, c, m in zip(np.unique(y_train), colors, markers):
    ax.scatter(X_train[y_train == l, 0],
                X_train[y_train == l, 1],X_train[y_train == l, 2],
                c=c, label=l, marker=m)

plt.xlabel('1')
plt.ylabel('2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('images/05_03.png', dpi=300)
plt.show()
