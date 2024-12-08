import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rnd



#reading file
frey_face = pd.read_csv("home_exam/frey-faces.csv", skiprows=4, delimiter=" ", header=None)

frey_face = frey_face.to_numpy()

#width and heigth of images
width = 20
height = 28


#plotting the first few images
# fig, axes = plt.subplots(5, 5, figsize=(15, 5))

# count = 0
# for y in range(len(axes[0])):
#     for x in range(len(axes[1])):
#         axes[y, x].imshow(frey_face[count].reshape((height, width)), cmap='gray')
#         axes[y, x].set_title(f"{count}")
#         count += 1

# plt.tight_layout()
# plt.show()

#defining my k

k_list = [3, 6, 8]

#looping through k
for k in k_list:
    max_iteration = 500 #max iteration

    #function fro calculating distance
    def distance(center_image, now, axis=1):
        s = np.sum((center_image - now)**2, axis=axis)
        return np.sqrt(s)



    #finding the centroids
    centroids = []
    for _ in range(k):
        center_ini = rnd.randint(0, len(frey_face))
        centroids.append(frey_face[center_ini, :])

    centroids = np.array(centroids)


    #doing the repetitions
    for i in range(max_iteration):
        
        clusters = [[] for _ in range(k)] #creating empty list for each cluster

        #looping through each image
        for j in range(len(frey_face)):
            dis = distance(centroids, frey_face[j,:]) #calcualting diatnce
            cluster = np.argmin(dis) #findin the correct cluster
            clusters[cluster].append(frey_face[j, :]) #adding the data point to the correct cluster
        
        #finding new centorid
        new_avg_centroid = np.array([np.mean(clus, axis=0) for clus in clusters])
        
        #check if we have reahced comvergence
        if np.all(centroids == new_avg_centroid):
            print("Done by convergence") #checking what stopts the loop
            break
        
        centroids = new_avg_centroid


    #Finding the images
    closest_images = []
    count = 0
    #looping through the clusters
    for cluster in clusters:
        centroid = centroids[count] #finding the centorid now
        distances = [distance(image, centroid, axis=0) for image in cluster] #calculating distance
        closest_indices = np.argsort(distances)[:5] #finding the 5 shortest dinstances
        #adding the closest distances to closet images 
        closest_images_in_cluster = [cluster[i] for i in closest_indices] 
        closest_images.append(closest_images_in_cluster)
        count += 1
    
    
    
    closest_images = np.array(closest_images)

    
    #plotting the results  
    fig, axes = plt.subplots(k, 6, figsize=(15, 10))
    count = 0

    #plotting the three centroid images
    for t in range(k):
        axes[t, 0].imshow(centroids[count].reshape((height, width)), cmap='gray')
        axes[t, 0].set_title(f"{count + 1}")
        count += 1
        
    count = 0
    for x in range(1, len(axes[1])):
        for y in range(0, k):
            # print("y = ", y)
            # print("x = ", x)
            axes[y, x].imshow(closest_images[y, x-1].reshape((height, width)), cmap='gray')
            axes[y, x].set_title(f"{count + 4}")
            count += 1

    plt.tight_layout()
    plt.title(f"Grey faces with k number = {k}")
    plt.savefig(f"home_exam/task2/grey_face_k{k}")
            

    



    
    
        



    
