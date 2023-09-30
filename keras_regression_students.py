import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from keras.layers import Layer
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import keras
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import Normalizer
import sklearn
from sklearn.metrics import mean_squared_error
from math import sqrt



df=pd.read_csv(r'C:\Users\Ariya Rayaneh\Desktop\StudentsPerformance.csv')

df.gender=df.gender.replace(['male','female'],[0,1])

df['race/ethnicity']=df['race/ethnicity'].replace(['group A','group B','group C','group D','group E'],[0,1,2,3,4])

df['parental level of education']=df['parental level of education'].replace(['some college','associate\'s degree','high school','some high school','bachelor\'s degree','master\'s degree'],[0,1,2,3,4,5])

df['lunch']=df['lunch'].replace(['standard','free/reduced'],[0,1])

x=df.iloc[:,:4]

# x1=Normalizer().fit_transform(x)

y=df['writing score']

# x=pd.DataFrame(x1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

print(x_train.iloc[3,:])
print(y_train.iloc[3])

# Define model
model = Sequential()
model.add(Dense(500, input_dim=4, activation= "relu"))
model.add(Dense(100, activation= "relu"))
model.add(Dense(50, activation= "relu"))
model.add(Dense(1))
#model.summary()

model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])
model.fit(x_train, y_train, epochs=300)

print(model.predict([[1,3,0,1]]))

pred_train=model.predict(x_test)

print(np.sqrt(mean_squared_error(y_test,pred_train)))