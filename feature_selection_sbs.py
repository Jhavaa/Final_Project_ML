import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from credit_data import small_X_train_scaled_std, small_y_train_scaled ,df_credit#Æ
from sbs import SBS

## Declare the learning algorithms that will be used
lreg = LogisticRegression(max_iter=10000)
svm = SVC()
rf = RandomForestClassifier()

### selecting features
## Logistic Regression
sbs = SBS(lreg, k_features=1)
sbs.fit(small_X_train_scaled_std, small_y_train_scaled)

# plot
k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.title("Logistic Regression")
plt.show()

## SVM
sbs = SBS(svm, k_features=1)
sbs.fit(small_X_train_scaled_std, small_y_train_scaled)

# plot
k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.title("SVM")
plt.show()

## Random Forest
sbs = SBS(rf, k_features=1)
sbs.fit(small_X_train_scaled_std, small_y_train_scaled)

# plot
k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.title("Random Forest")
plt.show()
  


# print(small_y)
# print(small_X)



#Æ
feat_labels = df_credit.columns[1:]
forest = RandomForestClassifier(n_estimators=10000,
                                random_state=0,
                                n_jobs=-1)

forest.fit(small_X_train_scaled_std, small_y_train_scaled)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(small_X_train_scaled_std.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                            feat_labels[indices[f]],
                            importances[indices[f]]))

plt.title('Feature Importances')
plt.bar(range(small_X_train_scaled_std.shape[1]),
        importances[indices],
        color='lightblue',
        align='center')

plt.xticks(range(small_X_train_scaled_std.shape[1]),
           feat_labels[indices], rotation=90)
plt.xlim([-1, small_X_train_scaled_std.shape[1]])
plt.tight_layout()
plt.show()