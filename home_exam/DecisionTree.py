#importing the neccesary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
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



all_features = []
t = 0
for i in range(5):
    accuracy = 0
    features = []
    for i in range(220):
        print(features)
        print(f"accuracy is {accuracy}")
        acc = np.array([])
        
        split = int(len(train_set)*0.2)

        randy = rnd.randint(0, len(train_set))

        if randy + split > len(train_set):
            validation_set = pd.concat([train_set.iloc[randy : ], train_set.iloc[ : randy + split - len(train_set)]])
            train_set_forward = train_set.iloc[ - len(train_set) + randy + split : randy]
        else:
            validation_set = train_set.iloc[randy:randy+split]
            train_set_forward = pd.concat([train_set.iloc[:randy], train_set.iloc[randy+split:]])
            
        for key in test_set.keys(): #using test set keys to not include lipophilicity
            dtc = DecisionTreeClassifier()
                
            X_train = train_set_forward.loc[:,features + [f"{key}"]]
            Y_train = train_set_forward[f"{last_value}"]
            x_vali = validation_set.loc[:,features + [f"{key}"]]
            y_vali = validation_set[f"{last_value}"]


            dtc.fit(X_train, Y_train)
            y_pred = dtc.predict(x_vali)
            acc = np.append(acc, metrics.f1_score(y_vali, y_pred))
        
        if acc.max() > accuracy:
            accuracy = acc.max()
            indx = np.where(acc == acc.max())[0][0]
            features.append(validation_set.columns[indx])
        else:
            break
    all_features += features
    t += 1
   

all_features = np.array(all_features)
print(len(all_features))
features = np.unique(all_features.flatten())


def select_training_sets(k, dataframe):
    frames = []
    for _ in range(10):
        frames.append(dataframe.iloc[int(k*len(dataframe)/10) : int((k+1)*len(dataframe)/10), :])
    
    t = frames.pop(k)
    frames = pd.concat(frames)
    
    return frames, t
        

y_pred_all = []
for k in range(10):
    dtc = DecisionTreeClassifier()
    
    frames, t = select_training_sets(k, train_set)
    
    X_train = frames.loc[:, features]
    Y_train = frames[f"{last_value}"]
    Y_act = t[f"{last_value}"]
    X_test_acc = t.loc[:, features]
    X_test = test_set.loc[:, features]
    
    dtc.fit(X_train, Y_train)
    y_pred1 = dtc.predict(X_test_acc)
    print(metrics.f1_score(Y_act, y_pred1))
    y_pred_all.append(dtc.predict(X_test))

y_pred_all = np.array(y_pred_all)

y_pred = np.mean(y_pred_all, axis=0)
y_pred = np.where(y_pred >= 0.5, 1, 0)



# Implementing a Decision tree classifier
dtc = DecisionTreeClassifier()
X_train = train_set.loc[:, features]
Y_train = train_set[f"{last_value}"]
X_test = test_set.loc[:, features]


dtc.fit(X_train, Y_train)
y_pred = dtc.predict(X_test)

# #writing a csv file to out in the competition
out_file = pd.DataFrame({"Id": test_set["Id"], f"{last_value}": y_pred})
out_file.to_csv("home_exam/DecisionTree/submission.csv", index=False)
   

   

#getting the most important features
feature_importance = pd.Series(dtc.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print(features)
print(feature_importance)
feature_importance = feature_importance.iloc[0:4]

a = feature_importance.index

dtc = DecisionTreeClassifier()
X_train = train_set.loc[:, a]
Y_train = train_set[f"{last_value}"]
X_test = test_set.loc[:, a]

dtc.fit(X_train, Y_train)
y_pred = dtc.predict(X_test)

feature_importance = pd.Series(dtc.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print(feature_importance.index)

out_file = pd.DataFrame({"Id": test_set["Id"], f"{last_value}": y_pred})
out_file.to_csv("home_exam/DecisionTree/submission.csv", index=False)

# #writing a csv file to out in the competition
out_file = pd.DataFrame({"Id": test_set["Id"], f"{last_value}": y_pred})
out_file.to_csv("home_exam/randomForestTree/submission.csv", index=False)
feature_importance.plot.bar()
plt.show()