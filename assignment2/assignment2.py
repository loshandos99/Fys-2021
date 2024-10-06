import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("assignment2/data_problem2.csv", header=None)

features = df.loc[0].to_numpy()
classification = df.loc[1].to_numpy()

#Plotting the data as histograms to show the features for each class
counts, bins = np.histogram(features[np.where(classification<1)])
plt.stairs(counts, bins, label="Class 0")
counts, bins = np.histogram(features[np.where(classification>0)])
plt.stairs(counts, bins, label="Class 1")
plt.legend()
plt.show()

# print(len(features))

split = int(len(features)*0.8)

training_features = features[:split]
training_classification = classification[:split]

test_features = features[split:]
test_classification = classification[split:]

def beta_estimater(x, alfa):
    first = 1/(len(x) * alfa)
    second = np.sum(x)
    return first*second

def mu_estimater(x):
    return np.sum(x)/len(x)

def sigma_sq_estimater(x, mu):
    sum = np.sum((x-mu)**2)
    return sum/len(x)


def gamma_distribution(x, beta, alfa, gamma):
    exp = x**(alfa-1)*np.exp(-x/beta)
    div = beta**alfa * gamma
    return exp/div

def gaussian_distribution(x, sigma, mu):
    exp = np.exp(-0.5*((x-mu)/sigma)**2)
    div = sigma*np.sqrt(2*np.pi)
    return exp/div



beta_est = beta_estimater(training_features, 2)
mu_est = mu_estimater(training_features)
sigma_est = np.sqrt(sigma_sq_estimater(training_features, mu_est))

prob_class_C0 = gamma_distribution(test_features, beta_est, 2, 1)
prob_class_C1 = gaussian_distribution(test_features, sigma_est, mu_est)

def classifier(C0, C1):
    classific = np.zeros(len(C0))
    classific[C0<C1] = 1
    return classific

prob_classification = classifier(prob_class_C0, prob_class_C1)

accuracy = np.mean(test_classification == prob_classification)*100

print(accuracy)


