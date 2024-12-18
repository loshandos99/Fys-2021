import pandas as p
import numpy as n 
import matplotlib.pyplot as plt
import sklearn.metrics as met

#Reading the csv file into a dataframe
DF = p.read_csv("SpotifyFeatures.csv",  index_col=0)




#Getting the dimensions of the data
#print(DF.shape)

#18 song properties, 232 725 songs


#Retrieving the pop and classical part of the dataframes in their own dataframe
Pop_DF = DF.loc["Pop"][["liveness", "loudness"]]
Classical_DF = DF.loc["Classical"][["liveness", "loudness"]]

#Finding how many songs are in each class
Pop_songs_length = Pop_DF.shape[0] #-> 9386 songs
Classical_songs_length = Classical_DF.shape[0] #-> 9256 song

#creating training and testing sets with 80/20 split per class
training_DF = p.concat([Pop_DF.iloc[:int(Pop_songs_length*0.8)], Classical_DF.iloc[:int(Classical_songs_length*0.8)]])
test_DF = p.concat([Pop_DF.iloc[int(Pop_songs_length*0.8):], Classical_DF.iloc[int(Classical_songs_length*0.8):]])

#making the matrixes
training_matrix = training_DF.to_numpy()
test_matrix = test_DF.to_numpy()



#making the genres vectors
training_genres = training_DF.index.to_numpy()
test_genres = test_DF.index.to_numpy()

print(test_genres.shape)
print(test_matrix.shape)

#making Pop=1 and Classical=0
training_genres = n.where(training_genres == "Pop", 1, 0)
test_genres = n.where(test_genres == "Pop", 1, 0)




fig, ax = plt.subplots()
ax.scatter(Classical_DF["loudness"], Classical_DF["liveness"], color="green", label="Classical")
ax.scatter(Pop_DF["loudness"], Pop_DF["liveness"], color="red", label="Pop")
ax.set_ylabel("liveness")
ax.set_xlabel("loudness")
ax.legend()
plt.show()



#logistic function
def sigmoid(x):
    """ 
    The function for returning the predicted value between 0 and 1 to identify the class fo the song
    """
    return n.exp(x)/(1+n.exp(x))

#Prediction function
def predict(features, weigths):
    """
    features = (x songs, 2 features)
    weigths = (2, 1)
    return => (x songs, 1 value between 0 and 1)
    
    function for returning the predicted value between 0 and 1 given the songs feature and the weitghs. This 
    is to determen the given class for each song
    """
    return sigmoid(n.dot(features, weigths))

def cost_function(features, weigths, classification):
    """ 
    features = (x songs, 2 features)
    weigths = (2, 1)
    classification = (x songs, 1 value)
    return => float
    
    Function for calculating the error of the function. We want this to be as low as possible
    """
    pred = predict(features, weigths)
    tmp = n.sum(-classification*n.log(pred) + (1-classification)*n.log(1-pred))
    return tmp / len(classification)
  
  
    
def gradient_descend(features, weigths, classification, learning_rate, epochs):
    """ 
    features = (x songs, 2 features)
    weigths = (2, 1)
    classification = (x songs, 1 value)
    learning_rate = float
    epochs = int

    return => (weigths (2, 1), cost history list(floats))
    
    function for calculating the weigths and returning the cost history, that is what the error value is for each epoch
    """
    cost_history = []
    for _ in range(epochs):
        
        pred = predict(features, weigths)
        
        weigths -= n.dot(features.T, pred-classification)*learning_rate/len(classification)

        cost_history.append(cost_function(features, weigths, classification))
    
    
    return weigths, cost_history


#Creating learning rates values, and starting weigth values, and a epoch nr
learnings = [0.005, 0.01, 0.05]
weigth = [0, 0]
epochs = 10000

#Running through the training set with each learning rate
res = [gradient_descend(training_matrix, weigth, training_genres, learn, epochs) for learn in learnings]


#Plotting the error of each epoch for each learning rate

fig, ax = plt.subplots()
x = n.arange(epochs)
for i in range(len(res)):
    ax.plot(x, res[i][1], label=f"Learning rate {learnings[i]}")
ax.set_ylabel("Cost")
ax.set_xlabel("Epoch")
ax.legend()
plt.show()



def classify(results, features, classification):
    """ 
    Results = List[return of gradient descent fucntion, (weigth, cost history)]
    features = features (2, x songs)
    classification = values of 0 or 1 depending on pop or classical music
    
    return => The accuracy of the model given in percent
    
    Function for calculating the accuracy of the classification model
    """
    prob = sigmoid(n.dot(features, results[0]))
    prob = n.where(prob >= 0.5, 1, 0)
    print(n.sum(prob))
    accuracy = n.mean(classification == prob)

    return accuracy*100


#Finding the accuracy for the different learning rates
for i in range(len(learnings)):

    print(f"The accuracy for learning rate {learnings[i]} is {classify(res[i], training_matrix, training_genres):.2f}")

print(res[2][0])

#Finding the accuracy for the test set for each learning rates
for i in range(len(learnings)):
    test_result = gradient_descend(test_matrix, res[i][0], test_genres, learnings[i], 1)
    print(f"The accuracy for the test results with learning rate {learnings[i]} is {classify(test_result, test_matrix, test_genres):.2f}")
    
#Creating the confusion matrixes
pred = sigmoid(n.dot(test_matrix, res[2][0]))
pred = n.where(pred >= 0.5, 1, 0)

conf_mat = met.confusion_matrix(test_genres, pred)
dis = met.ConfusionMatrixDisplay(conf_mat)
dis.plot()
plt.show()
