import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df=pd.read_csv("taxi.csv")

features=df.iloc[:,0:-1].values
target=df.iloc[:,-1].values

X_train,X_test,y_train,y_test=train_test_split(features,target,test_size=0.3,random_state=42)

c=LinearRegression()
c.fit(X_train,y_train)


model=pickle.dump(c,open('taxi.pkl','wb'))

#test the model
model=pickle.load(open('taxi.pkl','rb'))

