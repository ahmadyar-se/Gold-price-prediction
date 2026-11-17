import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# Reading the file through pandas.read_csv 

gold_data = pd.read_csv('C:\\Users\\ahmad\\Desktop\\Gold prediction\\gld_price_data.csv')

#Printing first 5 rows of dataset

#print(gold_data.head())

#Prinitng last 5 rows of dataset

#print(gold_data.tail())

#Printing statisical values of dataset 

#print(gold_data.describe())

#correlation

correlation = gold_data.corr(numeric_only=True)

#constructing heatmap to find correlation with matplotlib 

plt.figure(figsize= (8,8))
sns.heatmap(correlation, cbar = True, square = True, fmt= '.1f', annot= True, annot_kws= {'size': 8},  cmap= 'Blues')

#showing gold correlation 

#print(correlation['GLD'])

#for showing density of the value
#sns.displot(gold_data['GLD'])


#To show correlation
plt.show()

#splitting the features and target
x= gold_data.drop(['Date','GLD'], axis = 1)
y= gold_data['GLD']
print(x,y)



#splitting training data and test data 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 2)

#model training using Random forest regressor 

regressor = RandomForestRegressor(n_estimators=100)    #created a variable 

regressor.fit(x_train, y_train)

#Model evaluation (Prediction on test data)

test_data_prediction = regressor.predict(x_test)

print(test_data_prediction)


#finding error thrugh Rsquare 
error_score = metrics.r2_score(y_test,test_data_prediction)
print('R Squared Error:', error_score )


#making it list by list()
y_test = list(y_test)                                                          

#comparing Actual value vs Predicted value through matplotlib
plt.plot(y_test, color= 'Blue', label= 'Actual Value')
plt.plot(test_data_prediction, color = 'Green', label = 'Predicted Value')
plt.title('Actual Value vs Predicted Value')
plt.xlabel('Number of values')
plt.ylabel('GLD price')
plt.legend()
plt.show()