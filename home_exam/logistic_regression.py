#importing the neccesary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
import random as rnd


#loading the data sets
train_set = pd.read_csv("home_exam/train.csv")
test_set = pd.read_csv("home_exam/test.csv")

print(train_set.shape) #230 keys
#Reducing the data sets
for i in train_set.keys():
    unique = train_set[f"{i}"].unique()
    if len(unique) <= 1:
        train_set.drop(f"{i}", axis=1, inplace=True)
        test_set.drop(f"{i}", axis=1, inplace=True)
    last_value = i #to remember the last value, the one we want to classify
print(train_set.shape) #220 keys, down 10 features


   
accuracy = 0
features = []
for i in range(220):
    print(features)
    print(f"accuracy is {accuracy}")
    acc = np.array([])
    for key in test_set.keys(): #using test set keys to not include lipophilicity
        lr = LogisticRegression(max_iter=1000000)
        split = int(len(train_set)*0.2)

        randy = rnd.randint(0, len(train_set))

        if randy + split > len(train_set):
            validation_set = pd.concat([train_set.iloc[randy : ], train_set.iloc[ : randy + split - len(train_set)]])
            train_set_forward = train_set.iloc[ - len(train_set) + randy + split : randy]
        else:
            validation_set = train_set.iloc[randy:randy+split]
            train_set_forward = pd.concat([train_set.iloc[:randy], train_set.iloc[randy+split:]])
            
        X_train = train_set_forward.loc[:,features + [f"{key}"]]
        Y_train = train_set_forward[f"{last_value}"]
        x_vali = validation_set.loc[:,features + [f"{key}"]]
        y_vali = validation_set[f"{last_value}"]


        lr.fit(X_train, Y_train)
        y_pred = lr.predict(x_vali)
        acc = np.append(acc, metrics.f1_score(y_vali, y_pred))
    
    if acc.max() > accuracy:
        accuracy = acc.max()
        indx = np.where(acc == acc.max())[0][0]
        features.append(validation_set.columns[indx])
    else:
        break
    
    

# Implementing a logistic regression classifier
lr = LogisticRegression(max_iter=1000000)
X_train = train_set.loc[:, features]
Y_train = train_set[f"{last_value}"]
X_test = test_set.loc[:, features]


lr.fit(X_train, Y_train)
y_pred = lr.predict(X_test)

# #writing a csv file to out in the competition
out_file = pd.DataFrame({"Id": test_set["Id"], f"{last_value}": y_pred})
out_file.to_csv("home_exam/submission1/submission.csv", index=False)

print(accuracy)
print(features)
