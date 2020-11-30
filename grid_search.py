from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

from credit_data import small_X_train_scaled_std, small_y_train_scaled

from validation import Validation
### Setup pipes

pipe_lr = make_pipeline(StandardScaler(),
                         LogisticRegression(random_state=1, max_iter=10000))
pipe_svm = make_pipeline(StandardScaler(),
                         SVC(random_state=1))
pipe_rf = make_pipeline(StandardScaler(),
                        RandomForestClassifier())

# Grid search feature extraction

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

# Grid Search on lr
param_grid = [{'logisticregression__C': param_range, 
               'logisticregression__solver': ['lbfgs'],
               'logisticregression__penalty':['l2']},
              {'logisticregression__C': param_range,
               'logisticregression__solver': ['liblinear'],
               'logisticregression__penalty':['l1','l2']},
              {'logisticregression__C': param_range,
               'logisticregression__solver': ['sag'],
               'logisticregression__penalty':['l2']}]

gs = GridSearchCV(pipe_lr, 
                  param_grid,
                  n_jobs=-1)
gs = gs.fit(small_X_train_scaled_std, small_y_train_scaled)
best_params_LR = gs.best_params_
print('Grid Search best score (LR): %f' % gs.best_score_)
print('Grid Search best params(LR): ', best_params_LR)

# Grid Search on svm
param_grid = [{'svc__C': param_range, 
               'svc__kernel': ['linear']},
              {'svc__C': param_range, 
               'svc__gamma': param_range, 
               'svc__kernel': ['rbf']}]

gs = GridSearchCV(pipe_svm, 
                  param_grid, 
                  n_jobs=-1)
gs = gs.fit(small_X_train_scaled_std, small_y_train_scaled)
best_params_SVM =gs.best_params_
print('Grid Search best score (SVM): %f' % gs.best_score_)
print('Grid Search best params (SVM): ', best_params_SVM)

# Grid Search on rf
param_grid = [{'randomforestclassifier__n_estimators': [100, 120, 140, 160, 180, 200], 
               'randomforestclassifier__max_depth': [5, 10, 20, 30, None],
               'randomforestclassifier__class_weight': ['balanced', 'balanced_subsample', {0: 1, 1: 1}, {0: 1, 1: 5}]}]

gs = GridSearchCV(pipe_rf,
                  param_grid,
                  n_jobs=-1)
gs = gs.fit(small_X_train_scaled_std, small_y_train_scaled)
best_params_RF = gs.best_params_
print('Grid Search best score (RF): %f' % gs.best_score_)
print('Grid Search best params (RF): ', best_params_RF)



#LR
c_LR=best_params_LR['logisticregression__C']
penalty=best_params_LR['logisticregression__penalty']
solver=best_params_LR['logisticregression__solver']
Validation(LogisticRegression(penalty=penalty, random_state=0, C=c_LR,solver=solver))

#SVM
c_SVC=best_params_SVM['svc__C']
gamma=best_params_SVM['svc__gamma']
kernel=best_params_SVM['svc__kernel']
Validation(SVC(C=c_SVC,gamma=gamma,kernel=kernel,probability=True))
#RF
class_weight=best_params_RF['randomforestclassifier__class_weight']
max_depth=best_params_RF['randomforestclassifier__max_depth']
n_estimators=best_params_RF['randomforestclassifier__n_estimators']
Validation(RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,class_weight=class_weight))