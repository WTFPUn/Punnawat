import pandas as pd
import numpy as np
df = pd.read_csv('titanic.csv')
#print(df.head())

age = df['Age'].values
age = np.reshape(age,(-1,1))
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values= np.nan,strategy='most_frequent')
imp.fit(age)
df['Age'] = imp.transform(age)

x = df[['Sex','Pclass','Age','Fare']]
y=  df['Survived']



x = pd.concat([x,pd.get_dummies(x['Sex'],prefix = 'Sex',dummy_na=False)],axis=1).drop(['Sex'],axis=1)
print(x.head())
#print(y.head())
#print(df.shape)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size = 0.2)
print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(xtrain,ytrain)

ypre = LR.predict(xtest)

from sklearn.metrics import accuracy_score

dfafterpredict = pd.DataFrame({'ypredict' : ypre ,'ytest'  : ytest})

print(dfafterpredict.head(3))
print('accuracy = ',accuracy_score(ytest,ypre)*100,'%')