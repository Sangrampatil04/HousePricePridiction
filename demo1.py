import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data={"Hours_study":[1,2,3,4,5,6,7,8,9,10],"Exam_score":[50,60,70,80,90,100,120,130,140,150]}
df = pd.DataFrame(data)
print(df)
x=df[["Hours_study"]]
y=df[["Exam_score"]]

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2)
model=LinearRegression()
model.fit(x_train,y_train)


user_input=float(input("Enter number hours you  study "))
predicted_result=model.predict([[user_input]])

print("Predicted Result:-",predicted_result)