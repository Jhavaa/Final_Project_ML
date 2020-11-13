import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from credit_card_data import small_X_train_std, small_y_train
from sbs import SBS

## Declare the learning algorithms that will be used
lreg = LogisticRegression(max_iter=10000)
svm = SVC()
rf = RandomForestClassifier()

### selecting features
## Logistic Regression
sbs = SBS(lreg, k_features=1)
sbs.fit(small_X_train_std, small_y_train)

# plot
k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.show()

## SVM
sbs = SBS(svm, k_features=1)
sbs.fit(small_X_train_std, small_y_train)

# plot
k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.show()

## Random Forest
sbs = SBS(rf, k_features=1)
sbs.fit(small_X_train_std, small_y_train)

# plot
k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.show()

# print(small_y)
# print(small_X)