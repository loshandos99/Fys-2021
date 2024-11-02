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



all_features = []


#making the forward selection method
all_features = []
for i in range(5):   
    accuracy = 0
    features = []
    #looping through all the features in case all is needed
    for i in range(220):
        print(features)
        print(f"accuracy is {accuracy}") #keeping tabs
        acc = np.array([])
        
        #making the split before we test the features
        split = int(len(train_set)*0.2)

        randy = rnd.randint(0, len(train_set)) #random int

        #rulse to get 80/20 split completely random
        if randy + split > len(train_set):
            validation_set = pd.concat([train_set.iloc[randy : ], train_set.iloc[ : randy + split - len(train_set)]])
            train_set_forward = train_set.iloc[ - len(train_set) + randy + split : randy]
        else:
            validation_set = train_set.iloc[randy:randy+split]
            train_set_forward = pd.concat([train_set.iloc[:randy], train_set.iloc[randy+split:]])
            
        #looping through is feature
        for key in test_set.keys(): #using test set keys to not include lipophilicity
            rfc = RandomForestClassifier() #setting up classifier
            
            #creating training and validation sets
            X_train = train_set_forward.loc[:,features + [f"{key}"]]
            Y_train = train_set_forward[f"{last_value}"]
            x_vali = validation_set.loc[:,features + [f"{key}"]]
            y_vali = validation_set[f"{last_value}"]

            #training and predicting
            rfc.fit(X_train, Y_train)
            y_pred = rfc.predict(x_vali)
            acc = np.append(acc, metrics.f1_score(y_vali, y_pred))
        
        #finding accuracy
        if acc.max() > accuracy:
            accuracy = acc.max()
            indx = np.where(acc == acc.max())[0][0]
            features.append(validation_set.columns[indx])
        else:
            break #break if we have worse accuracy
        
    all_features += features #updating features
   
all_features = np.array(all_features)
print(len(all_features))
features = np.unique(all_features.flatten())#only want unique features




#creating a select traing sets function to randomly choose 90 prosent of the training data to train on, and 10 prosent to validate
def select_training_sets(k, dataframe):
    frames = []
    for _ in range(10):
        frames.append(dataframe.iloc[int(k*len(dataframe)/10) : int((k+1)*len(dataframe)/10), :])
    
    t = frames.pop(k)
    frames = pd.concat(frames)
    
    return frames, t
        
#cross validation
y_pred_all = []
for k in range(10):
    rfc = RandomForestClassifier()
    
    frames, t = select_training_sets(k, train_set)
    
    #making the training and testing functiom
    X_train = frames.loc[:, features]
    Y_train = frames[f"{last_value}"]
    Y_act = t[f"{last_value}"]
    X_test_acc = t.loc[:, features]
    X_test = test_set.loc[:, features]
    
    rfc.fit(X_train, Y_train)
    y_pred1 = rfc.predict(X_test_acc)
    print(metrics.f1_score(Y_act, y_pred1)) #keeping tabs on the accuracy
    y_pred_all.append(rfc.predict(X_test))

y_pred_all = np.array(y_pred_all) #finding the real predicted values


y_pred = np.mean(y_pred_all, axis=0)
y_pred = np.where(y_pred >= 0.5, 1, 0)

# #writing a csv file to out in the competition
out_file = pd.DataFrame({"Id": test_set["Id"], f"{last_value}": y_pred})
out_file.to_csv("home_exam/randomForestTree/submission.csv", index=False)
   

