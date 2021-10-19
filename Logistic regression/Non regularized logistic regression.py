@author: Danial Arab
"""
# The same data was used to generate a logistic regression model but with Python using SKlearn library 
import pandas as pd

df = pd.read_csv ('ex2data1.txt')
df.head()

# renaming the columns' names
df = df.rename (columns = {'34.62365962451697' : 'Exam one score', '78.0246928153624': 'Exam two score', '0': 'Admission decision'})
df.head()

# adding the first row which was replaced while renaming
df.loc[-1] = ['34.62365962451697', '78.0246928153624', '0']
df.index = df.index + 1
df = df.sort_index()
df.head()

# slicing 
X = df[['Exam one score', 'Exam two score']].astype('float')
y = df[['Admission decision']]
y = df.iloc[:, -1].values.astype('float')

# building and training logistic regression model

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression( random_state = 0)
classifier.fit(X, y)


import numpy as np

x_test = [[45. , 85.]]
x_test = np.array(x_test)

predictions = classifier.predict(x_test)
