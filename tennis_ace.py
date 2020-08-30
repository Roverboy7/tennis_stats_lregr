# Import necessary packages and libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
tennis_stats = pd.read_csv('tennis_stats.csv')
print(tennis_stats.head(10))
# X = tennis_stats['BreakPointsOpportunities']
# y = tennis_stats['Wins']
# plt.scatter(X,y)
# plt.xlabel('BreakPointsOpportunities ')
# plt.ylabel('Wins')
# plt.show()
# Select your features and outcomes here:
features = tennis_stats[['BreakPointsOpportunities',
'FirstServeReturnPointsWon']]
outcome = tennis_stats[['Winnings']]
features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size = 0.8)
# print(features_train)
# print(outcome_train)
# Run Linear Regression:
regr1 = LinearRegression()
regr1.fit(features_train,outcome_train)
prediction = regr1.predict(features_test)
plt.scatter(outcome_test,prediction, alpha=0.4)
plt.show()
