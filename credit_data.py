import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

############### A more detailed explanation to why we transformed the data the way we did can be found in credit_analysis.py ###############

### Prepare this dataset to be used for further analysis
## Import dataframe
# df_credit = pd.read_csv("c:/Users/Jhanava/Desktop/2020 FALL/CAP5610 - Introduction to Machine Learning/Final project/data/creditcard.csv")
#df_credit = pd.read_csv("../data/creditcard.csv")
df_credit = pd.read_csv("C:\\Users\\Adnrew\\Dropbox\\dataSets\\creditcard.csv")
#df_credit = pd.read_csv("https://www.kaggle.com/mlg-ulb/creditcardfraud/download")


# Shuffle data entries
df_credit = df_credit.sample(frac=1)

# find column names and save them (convenience)
columns = df_credit.columns[:]

# Split entries evenly by class
df_fraud = df_credit.loc[df_credit['Class'] == 1]
df_nonfraud = df_credit.loc[df_credit['Class'] == 0][:len(df_fraud.index)]

# Combine evenly split entries
df_credit_even = pd.concat([df_fraud, df_nonfraud])

# Shuffle again
df_credit_even = df_credit_even.sample(frac=1)

## Encode data

# Display data type of columns in the data frame
# print(df_credit.dtypes)

# The data types have already been assigned the correct data types.
# No need for further changes in regards to encoding.

## Perform appropriate scaling
sc = StandardScaler()

# Scale Time and Amount values (this data will use df_credit_even)
df_credit_scaled = df_credit_even.assign(Time=sc.fit_transform(df_credit_even['Time'].values.reshape(-1, 1)),
                                    Amount=sc.fit_transform(df_credit_even['Amount'].values.reshape(-1, 1)))

## Obtain values from dataframe columns and assign them appropriately
# iloc is a purely integer-location based indexing for selction by position.
# iloc is used to select data that is to be stored in X and y.
# X gets the whole row between the first column and second to last column.
# y gets the whole last column.
X, y = df_credit.iloc[:, :-1].values, df_credit.iloc[:, -1].values

#scaled
X_scaled, y_scaled = df_credit_scaled.iloc[:, :-1].values, df_credit_scaled.iloc[:, -1].values

## Split between training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 1, stratify = y)

#scaled
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y_scaled, train_size = 0.8, test_size = 0.2, random_state = 1, stratify = y_scaled)

# standardize training feature data
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

#scaled
X_train_scaled_std = sc.fit_transform(X_train_scaled)
X_test_scaled_std = sc.transform(X_test_scaled)

# take 10% of the data to perform functions faster
small_credit = df_credit.sample(frac=0.1)
small_X, small_y = small_credit.iloc[:, :-1].values, small_credit.iloc[:, -1].values

small_X_train, small_X_test, small_y_train, small_y_test = train_test_split(small_X, small_y, train_size = 0.8, test_size = 0.2, random_state = 1, stratify = small_y)

small_X_train_std = sc.fit_transform(small_X_train)
small_X_test_std = sc.transform(small_X_test)


#scaled
small_credit_scaled = df_credit_scaled.sample(frac=0.1)
small_X_scaled, small_y_scaled = small_credit_scaled.iloc[:, :-1].values, small_credit_scaled.iloc[:, -1].values

small_X_train_scaled, small_X_test_scaled, small_y_train_scaled, small_y_test_scaled = train_test_split(small_X_scaled, small_y_scaled, train_size = 0.8, test_size = 0.2, random_state = 1, stratify = small_y_scaled)

small_X_train_scaled_std = sc.fit_transform(small_X_train_scaled)
small_X_test_scaled_std = sc.transform(small_X_test_scaled)