# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 15:18:25 2023

@author:Kfir Avioz , On Avioz

This code script is to make measures for N_grams_3,4,5 csv matrixes to gather averaged distance matrix of clustered SARS 
distance matrix, and output analyzis. 
"""


from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.ensemble import IsolationForest
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

data_path =os.getcwd()+"\\final_calculations averaged"

if not os.path.exists(data_path):
    os.makedirs(data_path)
else:
     shutil.rmtree(data_path, ignore_errors=True)
     os.makedirs(data_path)

#since it is known that we need to have the next files it is OK to hardcode those names of the files.
mat1 = np.loadtxt("dist_mat_3.csv" , delimiter=',')
mat2 = np.loadtxt("dist_mat_4.csv" , delimiter=',')
mat3 = np.loadtxt("dist_mat_5.csv" , delimiter=',')
avg_row=[]
matrix=[]

#  producing averaged matrix 
for i in range(len(mat1[0])):
    for j in range (len(mat1[0])):
        avg_row.append((mat1[i][j]+mat2[i][j]+mat3[i][j])/3)
    matrix.append(avg_row)
    avg_row = []
matrix = np.asarray(matrix)
del avg_row ,mat1,mat2,mat3,i,j

# Collecting the names of the SARS into a list
file = open(os.getcwd()+"\\SARS_list.txt" , 'r')
lines = file.readlines()
SARS_names=[]
for i in range(len(lines)):
    if i%2 == 0 :
        SARS_names.append(lines[i])


#Performing MDS and than clustering and ploting the results.
#MDS:(multidimensional scaling - to 2D and plotting . its PCA on distance matrix
mds = MDS(random_state=0)
X_transform = mds.fit_transform(matrix)
kmeans = KMeans(n_clusters = 2).fit(X_transform)
predictedLabels = kmeans.labels_+1      
centroids = kmeans.cluster_centers_
plt.title("The Covid Dataset predicted labels")
plt.scatter(matrix[:,0], matrix[:,1], c = predictedLabels, s=50)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker = "x", s=80)
#plt.show()
plt.savefig(data_path+"\Dataset_predicted_lbls.png")
plt.clf()

plt.title("The covid predicted labels")
plt.xlabel("covid sample number")
plt.ylabel("pridicted label")
plt.bar(range(len(predictedLabels)),predictedLabels)
#plt.show()
plt.savefig(data_path+"\\Predicted_labels.png")
plt.clf()
#show the silhouette values for k clusters 
silhouetteAvg = silhouette_score(matrix, predictedLabels)
sample_silhouette_values_ = silhouette_samples(matrix, predictedLabels)  
plt.plot(sample_silhouette_values_) 
plt.plot(silhouetteAvg, 'r--')
plt.title("The silhouette plot for the various clusters.")
plt.xlabel("The silhouette coefficient values")
plt.ylabel("Cluster label")
xmin=0
xmax=len(predictedLabels)
# The vertical line for average silhouette score of all the values
plt.hlines(silhouetteAvg, xmin, xmax, colors='red', linestyles="--") 
#plt.show()
plt.savefig(data_path+ "\\silhouette.png")
plt.clf() 
dists = squareform(np.array(matrix))
linkage = hierarchy.linkage(dists, method="single")
hierarchy.dendrogram(linkage, color_threshold=0.3)
plt.xlabel("SARS number")
plt.ylabel("Dissimilarity")
plt.savefig(data_path+"\\Dendogram.png")
plt.clf() 

#predicting anomals behaviour via the distance matrix (MDS) with isolationforest algorithm
clf = IsolationForest(random_state=0).fit(matrix) 
anomal_list_score = clf.predict(matrix)
anomal_sars = []
for i in range( len (SARS_names)):
    if anomal_list_score[i] == -1:
        anomal_sars.append(SARS_names[i])
txt =""
for i in anomal_sars:
    txt+= i +"\n"
f= open(data_path+"\\anomals.txt","w+")
f.write(txt)
f.close()
tagged_clusters = kmeans.labels_

#grouping clustered SARS and outputing txt file with information.
group1=[]
group2=[]

count = 1
for cov in SARS_names:  
    if tagged_clusters[count-1] == 0:
        group1.append(cov)
    else:
        group2.append(cov)
    count+=1
txt3 = "----------------------------------\n Cluster groups:\n#######################\n"
txt4 = "\n\ngroup 1:\n------------\n"
txt5="" 
for cov in group1:
    txt5+=f'{cov}\n'
txt6 = "\n\n\ngroup 2:\n------------\n"
txt7="" 
for cov in group2:
    txt7+=f'{cov}\n'
sum_text = txt3+txt4+txt5+txt6+txt7
f= open(data_path+"\info.txt","w+")
f.write(sum_text)
f.close()
          
          
   
