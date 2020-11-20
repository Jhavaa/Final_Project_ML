import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

from credit_data import df_credit

### The dataframe was said to be imbalanced on the website,
### but how imbalanced?

# Here, we can print how many entries have a class of 0 and
# how many entries have a class of 1.
num_of_nofraud = df_credit['Class'].value_counts()[0]
num_of_fraud = df_credit['Class'].value_counts()[1]
print('amount of class 0 (no fraud): ', num_of_nofraud)
print('amount of class 1 (fraud): ', num_of_fraud)

## Percentage calculation:
print(round(num_of_fraud/len(df_credit) * 100, 3), '% of the data are fraudulent credit charges')

# This percentage is incredibly low! over 99% of the data
# in this set is of class 0.

# **the imbalanced nature of this data means we need to create a sample where the number of
# fraudulent charges and number of non-fraudulent charges are equal (50/50)**

### The classes need to be evenly distributed in order to get
### a less imbalanced model and avoid overfitting.

## First let's randomize the entries of the dataframe:
# Using sample(frac=1) on the credit dataframe shuffles the data
# randomly, but keeps all the entries intact. We do this to keep
# the data we work on random and independent from our influence.
df_credit = df_credit.sample(frac=1)

## Second we should split the data evenly between fraud and
## non-fraud.
df_fraud = df_credit.loc[df_credit['Class'] == 1]

# We know that there are 492 fraud entries, so we should limit the
# number of non fraudulent entries to 492 to keep it even.
df_nonfraud = df_credit.loc[df_credit['Class'] == 0][:len(df_fraud.index)]

## Third, we combine the data frames we created into one evenly
## distributed dataframe.
df_credit_even = pd.concat([df_fraud, df_nonfraud])

# Shuffle again
df_credit_even = df_credit_even.sample(frac=1)

## Finally, we can prove to ourselves that the Class column is now
## evenly distributed: equal 0's and 1's.
num_of_nofraud = df_credit_even['Class'].value_counts()[0]
num_of_fraud = df_credit_even['Class'].value_counts()[1]
print('amount of class 0 (no fraud): ', num_of_nofraud)
print('amount of class 1 (fraud): ', num_of_fraud)

# **This distribution will be done in the credit_data.py file.
#   The new dataframe is called "df_credit_even".**

### The only two values that have a description are time and
### and Amount, so we can use these two features to see how
### fraudulant charges and non-fraudulant charges differ in
### these values.

fraud = df_credit[df_credit['Class'] == 1]
nonfraud = df_credit[df_credit['Class'] == 0]

## How the amount spent relates to fraudulent charges
print(pd.concat([fraud.Amount.describe(), nonfraud.Amount.describe()], axis=1))

## How the time frame between purchases relates to fraudulent charges
print(pd.concat([fraud.Time.describe(), nonfraud.Time.describe()], axis=1))


### "Features V1, V2, … V28 are the principal components
### obtained with PCA, the only features which have not
### been transformed with PCA are 'Time' and 'Amount'."

## lets see what amount and time look like
amount_val = df_credit['Amount'].values
time_val = df_credit['Time'].values
n_bins = 20

fig, axs = plt.subplots(1, 2, tight_layout=True)

# a histogram will be used to see the general distribution
# of values through a histogram.
axs[0].set_title('Distribution of Transaction Amount', fontsize=14)
axs[0].hist(amount_val, bins=n_bins)

axs[1].set_title('Distribution of Transaction Time', fontsize=14)
axs[1].hist(time_val, bins=n_bins)

# V1_val = df_credit['V1'].values
# axs[1].set_title('Distribution of V1', fontsize=14)
# axs[1].hist(V1_val, bins=n_bins)

# **through these histograms, we can see how skewed these features are.
#   The values range so greatly compared to the already scaled features
#   (V1 - V28) **

plt.show()

### We need to scale the Time and Amount column so that it can match the same
### range and distribution pattern like the other feature values.

sc = StandardScaler()

## Before the standard scaler fitting and transformation
print("Before StandardScaler fitting and transformation: \n", df_credit.head())

## Time and Amount will be transformed with StandardScaler
df_credit = df_credit.assign(Time=sc.fit_transform(df_credit['Time'].values.reshape(-1, 1)),
                             Amount=sc.fit_transform(df_credit['Amount'].values.reshape(-1, 1)))

## See what the new transformed columns look like
print("After StandardScaler fitting and transformation: \n", df_credit.head())

# Create a histogram with these new values
amount_val = df_credit['Amount'].values
time_val = df_credit['Time'].values

print(time_val)
n_bins = 20

fig, axs = plt.subplots(1, 2, tight_layout=True)

axs[0].set_title('Distribution of Transaction Amount', fontsize=14)
axs[0].hist(amount_val, bins=n_bins)

axs[1].set_title('Distribution of Transaction Time', fontsize=14)
axs[1].hist(time_val, bins=n_bins)

plt.show()

# **This scaling will be done in the credit_data.py file.
#   The new dataframe is called "df_credit_scaled".**


# print("PATH: ", os.path)