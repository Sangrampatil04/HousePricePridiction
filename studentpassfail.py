import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#example dataset
data={
    "hours_studied":[1,2,3,4,5,6,7,8,9,10],
    "passed_exam":[0,0,0,0,1,1,1,1,1,1]
}

df=pd.DataFrame(data)

#features and lables
x=df[["hours_studied"]]
y=df[["passed_exam"]]

#train=test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#Logistic Regression model
model=LogisticRegression()
model.fit(x_train,y_train)

#model accurancy o test set
y_test_pred=model.predict(x_test)
print("Model Accurancy = ",accuracy_score(y_test,y_test_pred))

#user input prediction
user_input=float(input("Enter hours studied :"))
y_pred=model.predict([[user_input]])
print(y_pred)

if y_pred[0] == 1:
    print(f"If you study {user_input} hours , you are likely to PASS the exam")
else:
    print(f"If you study {user_input} hours , you are likely to FAIL the exam")