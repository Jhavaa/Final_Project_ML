from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.svm import SVC
from credit_data import small_X_train_scaled_std, small_y_train_scaled,X_train_scaled,y_train_scaled,X_test_scaled,y_test_scaled
from sklearn.model_selection import learning_curve

def Validation(estimator):
    pipe_lr = Pipeline([('scl', StandardScaler()),('pca', PCA(n_components=2)),('clf',estimator)])
    pipe_lr.fit(X_train_scaled, y_train_scaled)
    print('Test Accuracy: %.3f' % pipe_lr.score(X_test_scaled, y_test_scaled))
    y_pred = pipe_lr.predict(X_test_scaled)

    kfold = StratifiedKFold(n_splits=10,random_state=1).split(X_train_scaled, y_train_scaled)

    scores = []
    for k,(train, test) in enumerate(kfold):
        pipe_lr.fit(X_train_scaled[train], y_train_scaled[train])
        score = pipe_lr.score(X_train_scaled[test], y_train_scaled[test])
        scores.append(score)

    print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

    scores = cross_val_score(estimator=pipe_lr,
                            X=X_train_scaled,
                            y=y_train_scaled,
                            cv=10,
                            n_jobs=1)
    print('CV accuracy scores: %s' % scores)
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


    train_sizes, train_scores, test_scores =\
                    learning_curve(estimator=pipe_lr,
                                X=X_train_scaled,
                                y=y_train_scaled,
                                train_sizes=np.linspace(0.1, 1.0, 10),
                                cv=10,
                                n_jobs=1)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean,
            color='blue', marker='o',
            markersize=5, label='training accuracy')

    plt.fill_between(train_sizes,
                    train_mean + train_std,
                    train_mean - train_std,
                    alpha=0.15, color='blue')

    plt.plot(train_sizes, test_mean,
            color='green', linestyle='--',
            marker='s', markersize=5,
            label='validation accuracy')

    plt.fill_between(train_sizes,
                    test_mean + test_std,
                    test_mean - test_std,
                    alpha=0.15, color='green')

    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.8, 1.0])
    plt.tight_layout()
    plt.show()



    from sklearn.metrics import confusion_matrix

    pipe_lr.fit(X_train_scaled, y_train_scaled)
    y_pred = pipe_lr.predict(X_test_scaled)
    confmat = confusion_matrix(y_true=y_test_scaled, y_pred=y_pred)
    print(confmat)
    
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

    plt.xlabel('predicted label')
    plt.ylabel('true label')

    plt.tight_layout()
    plt.show()

    recall = confmat[1,1]/(confmat[1,1]+confmat[1,0])
    precision = confmat[1,1]/(confmat[1,1]+confmat[0,1])
    F1 = 2*precision*recall/(recall+precision)
    print("recall: {} \n precision{} \n F1{}".format(recall,precision,F1))
    
    from sklearn.metrics import roc_curve, auc
    from numpy import interp

    #pipe_lr = Pipeline([('scl', StandardScaler()),('pca', PCA(n_components=2)),('clf', LogisticRegression(penalty='l2', random_state=0, C=100.0))])

    X_train2 = X_train_scaled[:, [4, 14]]
    cv = list(StratifiedKFold(n_splits=3, random_state=1).split(X_train_scaled, y_train_scaled))

    fig = plt.figure(figsize=(7, 5))

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    for i, (train, test) in enumerate(cv):
        probas = pipe_lr.fit(X_train2[train],y_train_scaled[train]).predict_proba(X_train2[test])

        fpr, tpr, thresholds = roc_curve(y_train_scaled[test],
                                        probas[:, 1],
                                        pos_label=1)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr,
                tpr,
                lw=1,
                label='ROC fold %d (area = %0.2f)'
                    % (i+1, roc_auc))

    plt.plot([0, 1],
            [0, 1],
            linestyle='--',
            color=(0.6, 0.6, 0.6),
            label='random guessing')

    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
            label='mean ROC (area = %0.2f)' % mean_auc, lw=2)
    plt.plot([0, 0, 1],
            [0, 1, 1],
            lw=2,
            linestyle=':',
            color='black',
            label='perfect performance')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('Receiver Operator Characteristic')
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.show()