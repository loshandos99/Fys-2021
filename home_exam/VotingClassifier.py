#importing neccesary packages
import pandas as pd
import numpy as np


#loading the data sets
lr = pd.read_csv("home_exam/logistic_regression/submission.csv")
rfc = pd.read_csv("home_exam/randomForestTree/submission.csv")
dtc = pd.read_csv("home_exam/DecisionTree/submission.csv")

#transferring them into numpy arrays
lr_pred = lr["lipophilicity"].to_numpy()
rfc_pred = rfc["lipophilicity"].to_numpy()
dtc_pred = dtc["lipophilicity"].to_numpy()

#combining them
y_pred_all = np.array([lr_pred, rfc_pred, dtc_pred])

#findin the mean value and classifing them
y_pred = np.mean(y_pred_all, axis=0)
y_pred = np.where(y_pred >= 0.5, 1, 0)


#write csv file
out_file = pd.DataFrame({"Id": lr["Id"], "lipophilicity": y_pred})
out_file.to_csv("home_exam/VotingClassifier/submission.csv", index=False)