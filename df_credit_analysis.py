import matplotlib.pyplot as plt
from credit_data import df_credit

# The dataframe was said to be imbalanced on the website,
# but how imbalanced?

# Here, we can print how many entries have a class of 0 and
# how many entries have a class of 1.
num_of_nofraud = df_credit['Class'].value_counts()[0]
num_of_fraud = df_credit['Class'].value_counts()[1]
print('amount of class 0 (no fraud): ', num_of_nofraud)
print('amount of class 1 (fraud): ', num_of_fraud)

# Percentage calculation:
print(round(num_of_fraud/len(df_credit) * 100, 3), '% of the data are fraudulent credit charges')

# This percentage is incredibly low! over 99% of the data
# in this set is of class 0.

# "Features V1, V2, … V28 are the principal components
# obtained with PCA, the only features which have not
# been transformed with PCA are 'Time' and 'Amount'."

# lets see what amount and time look like
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

# **through these histograms, we can see how skewed these features are.**

plt.show()