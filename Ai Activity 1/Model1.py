import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import joblib


df = pd.read_csv("restaurent_reviews.csv")
df.columns = df.columns.str.strip()
x = df['Review'].values
y = df['Liked'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)


vect = CountVectorizer(stop_words='english')
x_train_vect = vect.fit_transform(x_train)
x_test_vect = vect.transform(x_test)


model = SVC()
model.fit(x_train_vect, y_train)

y_pred = model.predict(x_test_vect)


accuracy_score(y_pred, y_test)


# -------------------pipeline-----------------------
text_model = make_pipeline(CountVectorizer(), SVC())

text_model.fit(x_train, y_train)
y_pred = text_model.predict(x_test)

# print(accuracy_score(y_pred,y_test))
# saving the trained model

trained_model=joblib.dump(text_model,"Feedback_model")

feedback_mod=joblib.load("Feedback_model")


print("Thank you , I hope you liked it !")

print("Kindly Provide us some feedback : ")
msg = input()

res_ponse = feedback_mod.predict([msg])
set_code = ''

print(res_ponse)

if res_ponse > 0:
    set_code = "Positive"
else:
    set_code = "Negative"


if set_code == "Positive":
    print("Thank you For your Feedback ! hope You vist us Again ....")
else:
    print("Oops ! We saw you Didn't like something, Please Help us to improve : ")
    issue = input("Write something here ....")
    print(" Thank you for your Feedback we will improve it accordingly , Hope you visit us again....  ")
