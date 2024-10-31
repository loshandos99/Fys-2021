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

split = int(len(train_set)*0.8)

train_set_forward = train_set.iloc[:split]
validation_set = train_set.iloc[split:]

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
            lr = LogisticRegression(max_iter=1000000)
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
        
    all_features += features
    t += 1


all_features = np.array(all_features)
print(len(all_features))
features = np.unique(all_features.flatten())

# print(len(features))
# features = ['MolLogP', 'fr_COO', 'SlogP_VSA5', 'fr_piperdine', 'VSA_EState5', 'NumHDonors', 'NumSaturatedRings', 'fr_HOCCN', 'fr_halogen', 'fr_sulfide', 'Chi2v', 'fr_Ar_NH', 'fr_aniline', 'PEOE_VSA7', 'PEOE_VSA4', 'Number_of_Rotatable_Bonds', 'fr_unbrch_alkane', 'fr_ketone', 'fr_hdrzine']
    
# features = ['MolLogP', 'TPSA', 'MinEStateIndex', 'VSA_EState5']
y_pred_all = []
for k in range(10):
    lr = LogisticRegression(max_iter=1000000)
    X_train = train_set.iloc[int(k*len(train_set)/10) : int((k+1)*len(train_set)/10), :]
    Y_train = train_set.iloc[int(k*len(train_set)/10) : int((k+1)*len(train_set)/10), :]
    X_train = X_train.loc[:, features]
    Y_train = Y_train[f"{last_value}"]
    X_test = test_set.loc[:, features]
    
    lr.fit(X_train, Y_train)
    y_pred_all.append(lr.predict(X_test))

y_pred_all = np.array(y_pred_all)

y_pred = np.mean(y_pred_all, axis=0)
y_pred = np.where(y_pred >= 0.5, 1, 0)


# Implementing a logistic regression classifier
# lr = LogisticRegression(max_iter=1000000)
# X_train = train_set.loc[:, features]
# Y_train = train_set[f"{last_value}"]
# X_test = test_set.loc[:, features]


# lr.fit(X_train, Y_train)
# y_pred = lr.predict(X_test)

# #writing a csv file to out in the competition
out_file = pd.DataFrame({"Id": test_set["Id"], f"{last_value}": y_pred})
out_file.to_csv("home_exam/logistic_regression/submission.csv", index=False)

# print(accuracy)
print(features)
# print(t)


# features = ['MolLogP', 'fr_COO', 'SlogP_VSA5', 'fr_piperdine', 'VSA_EState5', 'NumHDonors', 'NumSaturatedRings', 'fr_HOCCN', 'fr_halogen', 'fr_sulfide', 'Chi2v', 'fr_Ar_NH', 'fr_aniline', 'PEOE_VSA7', 'PEOE_VSA4', 'Number_of_Rotatable_Bonds', 'fr_unbrch_alkane', 'fr_ketone', 'fr_hdrzine']

# print(len(features))