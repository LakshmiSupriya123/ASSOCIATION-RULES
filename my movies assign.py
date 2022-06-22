"""
Created on Fri May  6 23:10:13 2022

@author: Supriya
"""
#import pandas
import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules  # importing apriori , association rules 
from mlxtend.preprocessing import TransactionEncoder

titanic = pd.read_csv("C://Users/NAVEEN REDDY/Downloads/my_movies.csv")
titanic
df=pd.get_dummies(titanic) # deleting unwanted null or nan in data set
df.head()
list(titanic)
titanic.drop(['V1','V2','V3','V4','V5'],axis=1,inplace=True)
frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
frequent_itemsets

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.7)
rules
rules.sort_values('lift',ascending = False)

rules.sort_values('lift',ascending = False)[0:20]

rules[rules.lift>1]

rules[['support','confidence']].hist()

rules[['support','confidence','lift']].hist()

import matplotlib.pyplot as plt
x = [5,7,8,7,2,17,2]
y = [99,86,87,88,111]

plt.scatter(rules['support'], rules['confidence'])
plt.show()
import seaborn as sns
sns.scatterplot('support', 'confidence', data=rules, hue='antecedents')
plt.show()
''' conclusion:
  As per Lower the Confidence level Higher the no. of rules.
Higher the Support, lower the no. of rules.'''