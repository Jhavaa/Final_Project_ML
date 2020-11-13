import matplotlib.pyplot as plt
from credit_card_data import df_credit

# Features V1, V2, … V28 are the principal components
# obtained with PCA, the only features which have not
# been transformed with PCA are 'Time' and 'Amount'.

# fig, ax = plt.subplots(1, 2, figsize=(18,4))

amount_val = df_credit['Amount'].values
time_val = df_credit['Time'].values


plt.plot(range(1, len(amount_val) + 1), amount_val, marker='o')
plt.xlim([0, max(amount_val)])
plt.ylabel('amount_val')
plt.xlabel('Number of entries')
plt.grid()
plt.tight_layout()
plt.show()


# sns.distplot(amount_val, ax=ax[0], color='r')
# ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
# ax[0].set_xlim([min(amount_val), max(amount_val)])

# sns.distplot(time_val, ax=ax[1], color='b')
# ax[1].set_title('Distribution of Transaction Time', fontsize=14)
# ax[1].set_xlim([min(time_val), max(time_val)])



plt.show()