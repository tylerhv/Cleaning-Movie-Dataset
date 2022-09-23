import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

#from sklearn.linear_model import PolynomialFeatures


df = pd.read_csv(r'C:\Users\Tyler\OneDrive\Desktop\Fall 2022\Coursera Pandas\movies.csv')

#we need to see the data types for all of the columns, and see if we need to change anything
print(df.dtypes)

#we want to get ONLY movies. This data set has TV shows in it. We will drop all the 'movies' with less than 80 minutes of run time
df.dropna(subset=['RunTime'], axis=0, inplace=True)
df.drop(axis=0, index=df[df['RunTime']<85].index, inplace=True )
df.dropna(axis=0,subset=['Gross'],inplace=True)

# the Gross value such that it is a float value. I remove the $ and M sign from the column
for i in range(len(df['Gross'])):
    df['Gross'].iloc[i] = ((df['Gross'].iloc[i])[1:-1])

#Here, I change the dtype of the column to float
df['Gross'] = df['Gross'].astype('float')


#binning gross into low, mid, and high
bins = np.linspace(min(df['Gross']), max(df['Gross']), num=4)

group_names = ['Low','Mid','High']

#Made a new column with the string type
df['Gross_Binned'] = pd.Series(dtype='str')
df['Gross_Binned'] = df['Gross_Binned'].astype('str')
#Applying everything to this next line
df['Gross_Binned'] = pd.cut(df['Gross'], bins ,labels=group_names, include_lowest=True)

#'one-hot encoding'
#here I used concat. concat doesn't change the inplace, it just returns another dataframe. you gotta save it to a dataframe

df_dummies = pd.get_dummies(df['Gross_Binned'])
df = pd.concat([df, df_dummies],axis=1)

#replacing the commas found in VOTES, then changing it to a int

#Removing the commas from the 'VOTES' column. This way we can change 'VOTES' to an integer
df['VOTES'] = df.VOTES.str.replace(',','',regex=True)
df['VOTES'] = df['VOTES'].astype('float')


#linear regression using seaborn
sns.regplot(x='VOTES', y='Gross', data=df)
plt.ylim(0,)
plt.show()

#we can get a residual plot, to see if the errors in the data matches with the linear function: in other words, did we use the right thing?
sns.residplot(df['VOTES'],df['Gross'])
plt.show()

#Finding the R^2 score to this linear regression
x = df[['VOTES']]
y = df['Gross']
lm = LinearRegression()
lm.fit(x,y)
Yhat = lm.predict(x)
print(lm.score(x,y))
#this comes out to be .4048, so about 41% of the data points can be explained by this model. Pretty low correlation, although it is slightly positive
#It appears that more votes for a movie correlates to having a higher gross for a movie

#distribution plots
axl = sns.distplot(df['Gross'],hist=False, color='r', label="Actual Value")
sns.distplot(Yhat, hist=False, color='b', label="Fitted values", ax=axl)
plt.show()

#Here, we want to see how good our data set is at predicting points. We have a training set and a testing set. The testing set is 30% of the total data values
x_train, x_test, y_train, y_test=train_test_split(df['VOTES'],df['Gross'], test_size=.3, random_state = 0)
    #x_data: features or independent variables
    #y_data: dataset target: df['Gross]
    #x/y_test = parts of available data as testing sets
    #test size: percentage of data available for testing



#Cross Validation/out of sample testing
    #uses different portion of the data to test and train the model on different itterations

from sklearn.model_selection import cross_val_score
scores = cross_val_score(lm, x, y, cv=3)
    #returns an array of scores one for each partition that was chosen as a testing set
    #we can average the scores out to estimate out of sample r squared using the mean function Numpy
        #the first input is the type of model
        #cv is the number of partions 

print(scores)
#[-.014, .229, -9.884] 
#we see that this is not a great model for this data set.. The values are no where close to 1.0

df.to_csv('movies_without_tv.csv')


 

