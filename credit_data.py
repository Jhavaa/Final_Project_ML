import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

### Prepare this dataset to be used for further analysis
## Import dataframe
df_credit = pd.read_csv("c:/Users/Jhanava/Desktop/2020 FALL/CAP5610 - Introduction to Machine Learning/Final project/Final_Project/creditcard.csv")

# find column names and save them (convenience)
columns = df_credit.columns[:]

## Obtain values from dataframe columns and assign them appropriately
# iloc is a purely integer-location based indexing for selction by position.
# iloc is used to select data that is to be stored in X and y.
# X gets the whole row between the first column and second to last column.
# y gets the whole last column.
X, y = df_credit.iloc[:, :-1].values, df_credit.iloc[:, -1].values

## Split between training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 1, stratify = y)

# standardize training feature data
sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# take 10% of the data to perform functions faster
small_credit = df_credit.sample(frac=0.1)
small_X, small_y = small_credit.iloc[:, :-1].values, small_credit.iloc[:, -1].values

small_X_train, small_X_test, small_y_train, small_y_test = train_test_split(small_X, small_y, train_size = 0.8, test_size = 0.2, random_state = 1, stratify = small_y)

sc = StandardScaler()

small_X_train_std = sc.fit_transform(small_X_train)
small_X_test_std = sc.transform(small_X_test)