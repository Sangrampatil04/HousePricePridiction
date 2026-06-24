import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

#Sample email dataset
data={
    "text":[
        "Win a free iphone now",
        "Your account has been suspended",
        "Meeting tomorrrow",
        "Congratulation , you won a lottery",
        "Lets catch up over coffe",
        "Limited offer just for you ",
        "Update your password immediately",
        "Get cheap meds online",
        "Lunch with team at 1pm",
        "buy now and get 50 off"
        ],
    "lable": [1, 1, 0, 1, 0, 1, 1, 1, 0, 1] #1= spam 0=not spam

}


#convert to DataFrame
df=pd.DataFrame(data)
x=df["text"]
y=df["lable"]

#convert text to numeric features
vectorizer=CountVectorizer() # create object of countVectorizer
X=vectorizer.fit_transform(x)
print(x)

#target lables
y=df["lable"]

#train , test , spilt
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

#train logistic regression
model=LogisticRegression()
model.fit(X_train,y_train)
#Exampleuser input
user_input=input("Enter an email message : ")
newu = vectorizer.transform([user_input]) #transforms raw text into a numeric features vector

print(type(newu))

#predict spam/not spam
prediction=model.predict(newu)[0]

if prediction == 0:
    print("Prediction : Not Spam")
else:
    print("Prediction: Spam")

