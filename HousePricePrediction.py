#1.collect data ---- csv,json dummy data --pandas--house price pridiction
#2.Dataframe 
#3.distribute dara on x and y axis --price , area  y= depeandant value(price) ,x=area
#4.Spilt data into train and test --- 100%--train,train, --test,test--80%data--train, 20%data -- to test
#5.train the model on training data 
#6.train the model with training data
#7.user input -- area 
#8.Test the model on user input / testing data 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

flat_data ={
    "Size":[500,700,1000,1200,1500],
    "Price":[100000,150000,200000,250000,300000]
    }
df=pd.DataFrame(flat_data)
print(df)
x=df[["Size"]]
y=df[["Price"]]

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)
model=LinearRegression()
model.fit(x_train,y_train)
user_input=float(input("Enter the size of flat :- "))
predicted_price=model.predict([[user_input]])
print(f"Prediced Price:{predicted_price[0][0]:.2f}")