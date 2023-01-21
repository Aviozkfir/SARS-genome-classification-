# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from dtaidistance import dtw
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.ensemble import IsolationForest
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from datetime import datetime
from pathlib import Path
import numpy as np
#from google.colab import drive
import nltk
import random
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import warnings
import os
import shutil
import sys



warnings.filterwarnings(action = 'ignore') 
date = datetime.today().strftime('%Y-%m-%d %H:%M:%S').replace("-","_").replace(":","_").replace(" ","_")   ## yyyy , mm , dd , HH , MM , SS
data_path = os.getcwd()+"\\Data_"+date 
recovery_count_path = os.getcwd() +"\\recovery_count.txt"
recovery_clusterd_path = os.getcwd() +"\\recovery_clustered.txt"
recovery_anomals_path = os.getcwd() +"\\recovery_anomals.txt"
#######################################################################################
# mode : 0 - full sequence , 2- from pair_num to end , 1 - single CNN for pair_num
mode = 2  
pair_num = 274
Matrix_err_threshold = 100

#data for CNN 
chunks = 50
data_batch_size = 200
emb_dim = 100
nb_epoch = 5
N_grams = 4
kernel_size_ = 3
data_x6 = True

#######################################################################################
#recovery if fullseq fails: 
    
already_predicted_num = 0 
recovered_clusters=[]

# name - name of the virus 
# data - data of the virus. (can be N_grams)
# anomal - count number of times the current virus tagged with anomal behavior.
# clustered - list with size of imposter pairs , presenting the clusting index the current virus received in every pair sequence.
class Virus:
    def __init__(self, name, data):
        self.name = name
        self.data = data
        self.anomal = 0
        self.clusterd = []
        
    def __len__(self):
         return len(self.data)
 
    def __add__(self, other):
        data = self.data + other.data
        return Virus(self.name , data)
    
    def get_data_by_index (self,efresh):
        return Virus(self.name ,self.data[0:efresh])
    
        
# parsing the virus Data  into 2 lists - imposters and SARS family.
def ReadFile():
    imposterList = [] # this is a list to hold all imposters
    covidList = [] # a list to hold all covid genes
    
    f = open("data.txt")
    lines = f.readlines()
    is_new_seq = False
    corona_virus =False
    seq = ''
    for line in lines:
        if line[0] == '>':
            if "coronavirus" in line or "Betacoronavirus" in line:
                corona_virus = True
            else:
                corona_virus = False
            
            is_new_seq = not is_new_seq
            if not is_new_seq:
                if corona_virus:
                    covidList.append(Virus(line ,seq))
                else:
                    imposterList.append(Virus(line ,seq))
                seq = ""
                is_new_seq = True
        else:
           # if is_new_seq:
           seq += line.replace("\n" , '').replace(" ", '')
                     
    del seq , lines , f , corona_virus , is_new_seq , line
    return covidList , imposterList

# Create N-grams of the inputted sequence by the size (N_grams).
def Create_N_Grams(sequence,  N_grams ):
    words = []  
    size_n = len(sequence) - N_grams + 1
    for i in range(0, size_n , 3):
        word = sequence[i:i + N_grams]
        words.append(word)
    res = ' '.join(words)
    data = []
   
# iterate through each sentence in the file
    for i in sent_tokenize(res):
        temp = []
        # tokenize the sentence into words
        for j in word_tokenize(i):
            temp.append(j.lower())
     
        data += temp
    return data

#adding data to the sequence by concating the data as much as possible without changing 
#the order of the genome until the two pairs have the same size (max (pair1,pair2))
def Pad_imposters_size(imposters):
    new_list = []
    efresh = 0
    selected = 0 
    for pair in imposters:
        if len(pair[0]) > len(pair[1]):
            selected = 1
        else:
            selected = 0
        efresh = abs(len(pair[0]) - len(pair[1]))
        list_pair = list(pair)
        if efresh > len(list_pair[selected]):
              counter = int(efresh/len(list_pair[selected]))
              for i in range(counter):
                 list_pair[selected] = (list_pair[selected]) + (pair[selected])
        efresh = abs(len(list_pair[0]) - len(list_pair[1])) 
        list_pair[selected] = (list_pair[selected]) + (pair[selected].get_data_by_index(efresh))
        new_pair = tuple(list_pair)
        new_list.append(new_pair)
    return new_list 

#adjust the list shape to be valid as input for the word2vec keras model.
def CreateDataForW2v(N_gram_imposters , chunks):
    model_data =  [] 
    for  imposter in N_gram_imposters:
        res = [imposter.data[i * chunks:(i + 1) * chunks] for i in range((len(imposter) + chunks - 1) // chunks )]
        model_data += res
    return model_data

# Main function for PreProcessing: getting the data , arranging the SARS data as N_grams, reciving tupple of all pairs of imposters
# and using word2vec preprocessor model to learn the vocab of the N_grams 

def PreProcess():
    global emb_dim , chunks , N_grams
    print("Preprocessing: 0%") 
    covidList , imposterList = ReadFile()
    print("Preprocessing: 20%") 
    N_gram_covid = []
    N_gram_imposters = [] 
    
    for cov in  covidList:
        N_gram_covid.append(Virus(cov.name ,Create_N_Grams(cov.data,N_grams)))
    for imposter in  imposterList:
        N_gram_imposters.append(Virus(imposter.name ,Create_N_Grams(imposter.data , N_grams)))
    
    print("Preprocessing: 40%") 
    # Create CBOW model 
    model_data = CreateDataForW2v(N_gram_imposters , chunks) 
    w2v_model = Word2Vec(sentences = model_data, vector_size= emb_dim, window=5, min_count=1, workers=4)
    w2v_model.save("w2v.model")
    print("Preprocessing: 60%") 
    #creating all possible pairs of imposters and fit the size
    imposter_pairs = [(a, b) for idx, a in enumerate(N_gram_imposters) for b in N_gram_imposters[idx + 1:]]
    imposter_pairs = Pad_imposters_size(imposter_pairs)
    print("Preprocessing: 80%") 
    del N_gram_imposters , imposter , cov , covidList , imposterList
    return N_gram_covid , imposter_pairs , w2v_model
     

#data manipulation for the CNN model.
def emm0W(impost_1, t_model, data_batch_size):
    impost_1_np_3D = []
    for token in impost_1:
        if token in t_model.wv:
            impost_1_np_3D.append(t_model.wv[token])

    impost_1_np_3D = np.asarray(impost_1_np_3D)
    bloks = len(impost_1_np_3D)//data_batch_size
    X_1 = []
    for kk in range(bloks):
        X_1.append(impost_1_np_3D[kk*data_batch_size:(kk+1)*data_batch_size ,  :])
    X_1 = np.asarray(X_1)
    del impost_1_np_3D
    return X_1
# returns collection of the data adjusted for CNN model.
def emm0W_to_collection(collection, model, data_batch_size):
    res_list = []
    try:
        asdf = emm0W(collection, model, data_batch_size)
        for a in asdf:
            res_list.append(a)
    except Exception as e:
        res_list = []
        print(e)
    finally:
        return res_list
#this function is the main function for adjusting the data for the CNN model.
#in this function we also concat the data x6 to make more data for training the model.      
def create_XY(imp_1, imp_2):
    imp1 = [imp_1, imp_2]
    del imp_2
    l0 = [len(imp1[0]), len(imp1[1])]
    i1 = l0.index(max(l0))
    i2 = l0.index(min(l0))
    imp0 = []

    for kk in range(l0[i1] // l0[i2]):
        imp0 = imp0 + imp1[i2]

    imp1[i2] = imp0 + random.sample(imp1[i2], l0[i1] - len(imp0))
    
    if(data_x6):
      imp1[0]=imp1[0]+imp1[0]+imp1[0]+imp1[0]+imp1[0]+imp1[0]
      imp1[1]=imp1[1]+imp1[1]+imp1[1]+imp1[1]+imp1[1]+imp1[1]
    else:
      imp1[0]=imp1[0]+imp1[0]+imp1[0]
      imp1[1]=imp1[1]+imp1[1]+imp1[1]
  
  
    Y = [1] * len(imp1[0]) + [2] * len(imp1[1])
    X = [y for x in [imp1[0], imp1[1]] for y in x]

    X = np.asarray(X)
    Y = np.asarray(Y)
    return X, Y, imp1  

#In this function, we build the model layers, and training the model with the imposter pairs adjusted data. also plotting the data into directories
def CNN(X, Y, nb_filter, pool_size ,test_size , cnn_index, DropoutP=0.25):
    global data_batch_size , emb_dim , nb_epoch
    X_train, Y_train = X, Y
    # Creating a model
    model = Sequential()
   
    model.add(Conv1D(filters=nb_filter, kernel_size=3, padding='valid', activation='relu',
                         input_shape=(data_batch_size, emb_dim)))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Conv1D(filters=nb_filter, kernel_size=3, padding='valid', activation='relu',
                         input_shape=(data_batch_size, emb_dim)))
    model.add(MaxPooling1D(pool_size=pool_size))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())
    print('Fit model...')


    print(X.shape)
    print(Y.shape)
    Y_tmp = np.zeros([Y_train.shape[0], 2])
    Y_tmp[:, 0] = 2 - Y_train
    Y_tmp[:, 1] = Y_train - 1
    Y_train = Y_tmp
    history = model.fit(X_train, Y_train, validation_split=test_size, epochs = nb_epoch, verbose=1)
    
     
    print("history['accuracy'] = ", history.history['accuracy'][-1])
    print("history['val_accuracy'] = ", history.history['val_accuracy'][-1])
    print("Finish CNN")
    print("---------------------------")
    
    #Print figures
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left') 
    #plt.show()
    plt.savefig(data_path+"\pairs_"+ str(cnn_index)+"\CNN\\"+"accuracy.png")
    plt.clf()
      # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    plt.savefig(data_path+"\pairs_"+ str(cnn_index)+"\CNN\\"+"loss.png")
    plt.clf()
    return model , history

# moving average func- ment to create signal from the output prediction of the CNN model reffers to predicted SARS
def moving_average(sars, n=3) :
    ret = np.cumsum(sars, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# The input SARS is predicted by the trained CNN model and produce an binary model (0-closer to imposter1 , 1- close to imposter 2)
#also after that the predicted array turns into signal, and the plots are saved in directories.
def predict_SARS_Make_Signal(N_gram_covid , cnn_model ,pair_index ,w2v_model):
    pre_proc_cov =[] 
    classified_cov = []
    signals_list = []
    count = 1 
    for cov in N_gram_covid:
        temp = emm0W(cov.data ,w2v_model , data_batch_size)
        pre_proc_cov.append(temp)
    
    for cov in pre_proc_cov:
        predict =cnn_model.predict(np.asarray(cov))
        predict = np.argmax(predict,axis=1)
        classified_cov.append(predict)
        signal = moving_average(predict) 
        signals_list.append(signal)
        plt.plot(signal)
        #plt.show()
        plt.savefig(data_path+"\pairs_"+ str(pair_index)+"\covids\\"+ "signal_"+str(count)+".png")
        plt.clf()
        count+=1
        
    del  pre_proc_cov
    return classified_cov, signals_list

# function to cluster the SARS by the distance matrix received by dinamic time wrapping (on the SARS signals that was produced)
# first the distance matrix shape get changed to 2 dimension to visualize better the results by MDS(multi dimensional scaling )
# and afterwards, clustered by Kmeans algorithm. Moreover, the distance matrix analyzed for anomals by the Isolation forest model 
# and the plots saves in directories.
def Cluster_SARS(matrix,pair_index, mode =0 , k = 2):
    global N_gram_covid
    
    if np.asarray(matrix).any():

        #MDS:(multidimensional scaling - to 2D and plotting . its PCA on distance matrix
        mds = MDS(random_state=0)
        X_transform = mds.fit_transform(matrix)
        kmeans = KMeans(n_clusters = k).fit(X_transform)
        predictedLabels = kmeans.labels_+1      
        centroids = kmeans.cluster_centers_
        cov_names = []
        for cov in N_gram_covid:
            cov_names.append(cov.name)
        #~~~~~~~~
        
        if mode == 0:
            plt.title("The Covid Dataset predicted labels")
            plt.scatter(matrix[:,0], matrix[:,1], c = predictedLabels, s=50)
            plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker = "x", s=80)
            #plt.show()
            plt.savefig(data_path+"\pairs_"+ str(pair_index) +"\Clusters\\"+ "Dataset_predicted_lbls.png")
            plt.clf()
        
            plt.title("The covid predicted labels")
            plt.xlabel("covid sample number")
            plt.ylabel("pridicted label")
            plt.bar(range(len(predictedLabels)),predictedLabels)
            #plt.show()
            plt.savefig(data_path+"\pairs_"+ str(pair_index) +"\Clusters\\"+ "Predicted_labels.png")
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
            plt.savefig(data_path+"\pairs_"+ str(pair_index) +"\Clusters\\"+ "silhouette.png")
            plt.clf() 
            dists = squareform(np.array(matrix))
            linkage = hierarchy.linkage(dists, method="single")
            hierarchy.dendrogram(linkage, color_threshold=0.3)
            plt.xlabel("SARS number")
            plt.ylabel("Dissimilarity")
            plt.savefig(data_path+"\pairs_"+ str(pair_index) +"\Clusters\\"+"Dendogram.png")
            plt.clf() 
            clf = IsolationForest(random_state=0).fit(matrix)
            anomal_list_score = clf.predict(matrix)
            anomal_sars = []
            for i in range( len (cov_names)):
                if anomal_list_score[i] == -1:
                    anomal_sars.append(cov_names[i])
                    N_gram_covid[i].anomal = N_gram_covid[i].anomal + 1
            txt =""
            for i in anomal_sars:
                txt+= i +"\n"
            f= open(data_path+"\pairs_"+ str(pair_index) +"\Clusters\\"+"anomals.txt","w+")
            f.write(txt)
            f.close()
            return kmeans.labels_
        else:
            plt.title("The Covid Dataset predicted labels")
            plt.scatter(matrix[:,0], matrix[:,1], c = predictedLabels, s=50)
            plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker = "x", s=80)
            #plt.show()
            plt.savefig(data_path+"\Post_calculations\\"+ "Dataset_predicted_lbls.png")
            plt.clf()
        
            plt.title("The covid predicted labels")
            plt.xlabel("covid sample number")
            plt.ylabel("pridicted label")
            plt.bar(range(len(predictedLabels)),predictedLabels)
            #plt.show()
            plt.savefig(data_path+"\Post_calculations\\"+ "Predicted_labels.png")
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
            plt.savefig(data_path+"\Post_calculations\\"+ "silhouette.png")
            plt.clf()
            dists = squareform(np.array(matrix))
            linkage = hierarchy.linkage(dists, method="single")
            hierarchy.dendrogram(linkage, color_threshold=0.3)
            plt.xlabel("SARS number")
            plt.ylabel("Dissimilarity")
            plt.savefig(data_path+"\Post_calculations\\"+ "Dendogram_"+str(N_grams)+".png")
            plt.clf()
            clf = IsolationForest(random_state=0).fit(matrix)
            anomal_list_score = clf.predict(matrix)
            anomal_sars = []
            for i in range( len (cov_names)):
                if anomal_list_score[i] == -1:
                    anomal_sars.append(cov_names[i])
                    N_gram_covid[i].anomal = N_gram_covid[i].anomal + 1
            txt =""
            for i in anomal_sars:
                txt+= i +"\n"
            f= open(data_path+"\Post_calculations\\"+ "Anomals_"+str(N_grams)+".txt","w+")
            f.write(txt)
            f.close()
            return kmeans.labels_
        
#removing directory                
def RemoveDataDir(dir_path):
    shutil.rmtree(dir_path, ignore_errors=True)

#creating the Directories structure for the project.
# taking care of each mode (signal,from pair, full sequence)
def CreateDataDirs(num_imposters , mode , pair_num = None):
    
    
        if not os.path.exists(data_path):
            os.makedirs(data_path)
            os.makedirs(data_path+"\Post_calculations")
        if mode == 0 :
            for i in range(1,num_imposters+1):
                os.makedirs(data_path+"\pairs_"+str(i))
                os.makedirs(data_path+"\pairs_"+ str(i)+"\CNN")
                os.makedirs(data_path+"\pairs_"+ str(i)+"\Clusters")
                os.makedirs(data_path+"\pairs_"+ str(i) +"\covids")
        elif mode ==1 :
            os.makedirs(data_path+"\pairs_"+str(pair_num))
            os.makedirs(data_path+"\pairs_"+ str(pair_num)+"\CNN")
            os.makedirs(data_path+"\pairs_"+ str(pair_num)+"\Clusters")
            os.makedirs(data_path+"\pairs_"+ str(pair_num) +"\covids")
        else:
            for i in range(pair_num , num_imposters+1):
                os.makedirs(data_path+"\pairs_"+str(i))
                os.makedirs(data_path+"\pairs_"+ str(i)+"\CNN")
                os.makedirs(data_path+"\pairs_"+ str(i)+"\Clusters")
                os.makedirs(data_path+"\pairs_"+ str(i) +"\covids")
           
# if number of zeros is lower than threshold for zeros in distance matrix return true else false.
def Pair_quality_check(matrix , num_covid ):
    global Matrix_err_threshold
    count = np.count_nonzero(matrix == 0)
    threshold =(num_covid + Matrix_err_threshold)
    return (count <= threshold)  

#Creating info text for final results of the current N_gram run.
def Create_info_text (pair , pair_num ):
    global chunks , data_batch_size, emb_dim , N_grams , Matrix_err_threshold , kernel_size_ , nb_epoch ,data_path
    txt1 = "#######################\nGeneral Information:\n---------------------\n" +  "imposter_1: " +pair[0].name +"\n" + "imposter_2: " +pair[1].name +"\n#######################\n\n"
    txt2 = "CNN Information:\n---------------------\n"+ "num of epcohs: " +str(nb_epoch) +"\n" +"kernel_size: " +str(kernel_size_) +"\n" + "data_batch: " +str(data_batch_size) +"\n" + "embedding_dimension: " +str(emb_dim)+"\n#######################\n\n" 
    txt3 = "More Information:\n---------------------\n" + "N_grams: " +str(N_grams)+"\n" +"distance_matrix_err_threshold: " +str(Matrix_err_threshold)+"\n" +"data_x6: " +str(data_x6)+"\n#######################\n"
    sum_text = txt1+txt2+txt3
    f= open(data_path+"\pairs_"+str(pair_num)+"\info.txt","w+")
    f.write(sum_text)
    f.close()
    
#appending the clust tag into the virus class for later analyzing (postAnalyze sars)    
def SaveClustTag(cluster_list , N_gram_covid):
    count =0
    for tag in cluster_list:
        N_gram_covid[count].clusterd.append(tag)
        count += 1

# After running and analyzing the SARS in each pair of imposters run , we post analyze all the data together 
# by creating distance matrix by the clustering index each SARS recived in each run and produce results with plots in directories.
# here we take care also for recovered data if the run failed in the middle for some reason (RAM choked for example)    
def PostAnalyzeSARS():
    global N_grams
    curr_res = 0
    mat =[]
    row = []
    if has_recovered:
        new_list_oflists = []
        transpose_list = np.array(recovered_clusters).T.tolist()
        for tagged in transpose_list:
            new_list_oflists.append(tagged)
        for z in range(len(N_gram_covid)):
            new_list_oflists[z] += N_gram_covid[z].clusterd
            N_gram_covid[z].clusterd = new_list_oflists[z]
        RecoverAnomals()
    size = len(N_gram_covid[0].clusterd)        
    for i in range(len(N_gram_covid)):
        for j in range(len(N_gram_covid)):
            for k in range(size):
                if N_gram_covid[i].clusterd[k] != N_gram_covid[j].clusterd[k]:
                    curr_res += 1
            row.append(curr_res/size)
            curr_res = 0
        mat.append(row)
        row = []
    np.savetxt("dist_mat_"+str(N_grams)+".csv",np.asarray(mat),delimiter= ',' , fmt='%1.3f')
    tagged_clusters = Cluster_SARS(np.asarray(mat) , pair_num, mode = 1)
    group1=[]
    group2=[]
    txt1 = "#######################\nSARS_LIST_ANOMALS_COUNTER:\n#######################\n"
    txt2 =""
    count = 1
    for cov in N_gram_covid:  
        txt2+= "Cov num: "+str(count)+" Cov Name: "+cov.name+".....how many time was anomal (0-275): "+str(cov.anomal)+"\n"
        if tagged_clusters[count-1] == 0:
            group1.append(cov.name)
        else:
            group2.append(cov.name)
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
    sum_text = txt1+txt2+txt3+txt4+txt5+txt6+txt7
    f= open(data_path+"\Post_calculations\info.txt","w+")
    f.write(sum_text)
    f.close()

# receiving the data saved for the clustered SARS from previous run that failed for some reason ( electric out,RAM choked ..etc)    
def recoverClusters():
        recovery_clustered_file = Path(recovery_clusterd_path)
        if recovery_clustered_file.is_file():
            f = open("recovery_clustered.txt", "r")
            while True:
                str_line_list = list(f.readline())
                if not str_line_list:
                        break  
                del str_line_list[-1]
                int_line_list = [int(i) for i in str_line_list] 
                recovered_clusters.append(int_line_list)
            f.close()
            return True
        else:
            return False


#save anomal count for recovering reasons.        
def SaveAnomals():
    global N_gram_covid
    f = open("recovery_anomals.txt", "w")
    txt =""
    for cov in N_gram_covid:
        txt += str(cov.anomal)+"\n"
    f.write(txt);
    f.close()   

# receiving the data saved for the anomal SARS count from previous run that failed for some reason ( electric out,RAM choked ..etc)   
def RecoverAnomals():
    global N_gram_covid
    recovery_anomals_file = Path(recovery_anomals_path)
    if recovery_anomals_file.is_file():
        f = open("recovery_anomals.txt", "r")
        for i in range(len(N_gram_covid)):
            N_gram_covid[i].anomal += int(f.readline())    
            
# this mode runs full seqence of all pairs of imposters 
# for each pair : we build and traing CNN , predicting the SARS and analyzing the data
# while running the sequence we maintain the data to be recovered incase the run fails for some reason     
def Full_sequence():
    global mode , N_gram_covid , imposter_pairs , w2v_model
    CNN_list=[]
    All_predicted_SARS = []
    All_SARS_Signals = []
    All_distance_matrixes = []
    pair_num = 1  
    num_covid = len(N_gram_covid)
    num_imposters = len(imposter_pairs)
    
    CreateDataDirs(num_imposters, mode ,None )
    
    recovery_clustered_file = Path(recovery_clusterd_path)
    if recovery_clustered_file.is_file():
         recovery_cluster_file = open("recovery_clustered.txt", "a")
    else:
        recovery_cluster_file = open("recovery_clustered.txt", "w")     
    try: 
        for pair in imposter_pairs:
            flag = False
            imp_1 = emm0W_to_collection(pair[0].data, w2v_model, data_batch_size)
            imp_2 = emm0W_to_collection(pair[1].data, w2v_model, data_batch_size) 
            X, Y, imp1 = create_XY(imp_1, imp_2)
            print("Preprocessing: 100%") 
            cnn_model , cnn_history = CNN(X, Y ,nb_filter = 500,pool_size = 1,test_size = 0.25,cnn_index = pair_num , DropoutP=0.25)
            CNN_list.append(cnn_model)
            predicted_sars , signal_sars = predict_SARS_Make_Signal(N_gram_covid , cnn_model , pair_num , w2v_model)
            All_predicted_SARS.append(predicted_sars)
            
        
            distance_matrix = dtw.distance_matrix_fast(signal_sars)
            if(Pair_quality_check(distance_matrix, num_covid)):
                All_SARS_Signals.append(signal_sars)
                All_distance_matrixes.append(distance_matrix)
                print("succesfully Building and predicting with CNN number : "+ str(pair_num))
                clustered_list = Cluster_SARS(distance_matrix , pair_num)
                list_string =list( map(str, clustered_list))
                listToStr = ''.join([str(elem) for elem in list_string])
                flag = True
                SaveClustTag(clustered_list , N_gram_covid)
                Create_info_text(pair , pair_num)
            else:
                print("Pair " +str(pair_num) +"has been removed \n" )
                RemoveDataDir(data_path+"\pairs_"+str(pair_num))
            pair_num +=1
            f = open("recovery_count.txt", "w")
            f.write(str(pair_num))
            f.close()
            SaveAnomals()
            if flag:
                recovery_cluster_file.write(listToStr + '\n')
        recovery_cluster_file.close()
    except :
        print("error with Building and predicting on CNN number : "+ str(pair_num))
        recovery_cluster_file.close()
        sys.exit()
    del X,Y,imp1
    
# this mode give the option to run only 1 pair selected for further research purporses. 
# creating directories adjusted only for 1 run , and running only one pair analyzing sequence.
# while running the sequence we maintain the data to be recovered incase the run fails for some reason       
def Single_Sequence(pair_num):
        global mode, data_path , N_gram_covid , imposter_pairs , w2v_model
        CNN_list=[]
        All_predicted_SARS = []
        All_SARS_Signals = []
        All_distance_matrixes = []
        num_covid = len(N_gram_covid)
        CreateDataDirs( 1 ,  mode , pair_num  )
        
        pair_num -=1
        imp_1 = emm0W_to_collection(imposter_pairs[pair_num][0].data, w2v_model, data_batch_size)
        imp_2 = emm0W_to_collection(imposter_pairs[pair_num][1].data, w2v_model, data_batch_size) 
        X, Y, imp1 = create_XY(imp_1, imp_2)
        print("Preprocessing: 100%") 
        cnn_model , cnn_history = CNN(X, Y ,nb_filter = 500,pool_size = 1,test_size = 0.25,cnn_index = pair_num , DropoutP=0.25)
        CNN_list.append(cnn_model)
        predicted_sars , signal_sars = predict_SARS_Make_Signal(N_gram_covid , cnn_model , pair_num, w2v_model)
        All_predicted_SARS.append(predicted_sars)
        distance_matrix = dtw.distance_matrix_fast(signal_sars)
        if(Pair_quality_check(distance_matrix, num_covid)):
            All_SARS_Signals.append(signal_sars)
            All_distance_matrixes.append(distance_matrix)
            print("succesfully Building and predicting with CNN number : "+ str(pair_num))
            Cluster_SARS(distance_matrix , pair_num)
            Create_info_text(imposter_pairs[pair_num] , pair_num)
        else:
            print("Pair " +str(pair_num) +"has been removed \n" )
            RemoveDataDir(data_path+"\pairs_"+str(pair_num))
        pair_num +=1
        #just for now!
        ### TBD: Algorithm for anomality in each signal.

        del X,Y,imp1
       
# this mode runs from chosen pair until the last pair of imposters , this mode is mainly for recovering reasons. 
# for each pair : we build and traing CNN , predicting the SARS and analyzing the data
# while running the sequence we maintain the data to be recovered incase the run fails for some reason        
def  Sequence_from_mid(pair_num):
        global mode, data_path , N_gram_covid , imposter_pairs , w2v_model    
        CNN_list=[]
        All_predicted_SARS = []
        All_SARS_Signals = []
        All_distance_matrixes = []
        num_covid = len(N_gram_covid)
        num_imposters = len(imposter_pairs)
        
        recovery_count_file = Path(recovery_count_path)
        if recovery_count_file.is_file():
            with open('recovery_count.txt', 'r') as file:
                already_predicted_num =int(file.read().replace('\n', ''))
                pair_num = already_predicted_num
        recovery_clustered_file = Path(recovery_clusterd_path)
        if recovery_clustered_file.is_file():
             recovery_cluster_file = open("recovery_clustered.txt", "a")
        else:
            recovery_cluster_file = open("recovery_clustered.txt", "w")  
        
       

        CreateDataDirs(num_imposters, mode ,pair_num )
      
        try: 
            for index  in range(pair_num -1 ,len(imposter_pairs)):
                flag = False
                imp_1 = emm0W_to_collection(imposter_pairs[index][0].data, w2v_model, data_batch_size)
                imp_2 = emm0W_to_collection(imposter_pairs[index][1].data, w2v_model, data_batch_size) 
                X, Y, imp1 = create_XY(imp_1, imp_2)
                print("Preprocessing: 100%") 
                cnn_model , cnn_history = CNN(X, Y ,nb_filter = 500,pool_size = 1,test_size = 0.25,cnn_index = pair_num , DropoutP=0.25)
                CNN_list.append(cnn_model)
                predicted_sars , signal_sars = predict_SARS_Make_Signal(N_gram_covid , cnn_model , pair_num , w2v_model)
                All_predicted_SARS.append(predicted_sars)
                
            
                distance_matrix = dtw.distance_matrix_fast(signal_sars)
                if(Pair_quality_check(distance_matrix, num_covid)):
                    All_SARS_Signals.append(signal_sars)
                    All_distance_matrixes.append(distance_matrix)
                    print("succesfully Building and predicting with CNN number : "+ str(pair_num))
                    clustered_list = Cluster_SARS(distance_matrix , pair_num)
                    list_string = map(str, clustered_list)
                    listToStr = ''.join([str(elem) for elem in list_string])
                    flag = True
                    SaveClustTag(clustered_list , N_gram_covid)
                    Create_info_text(imposter_pairs[index], pair_num)
                else:
                    print("Pair " +str(pair_num) +"has been removed \n" )
                    RemoveDataDir(data_path+"\pairs_"+str(pair_num))
                pair_num +=1

                f = open("recovery_count.txt", "w")
                f.write(str(pair_num));
                f.close()
                SaveAnomals()
                if flag:
                    recovery_cluster_file.write(listToStr + '\n')
            recovery_cluster_file.close()
                
                ### TBD: Algorithm for anomality in each signal.
        except :
            print("error with Building and predicting on CNN number : "+ str(pair_num))
            recovery_cluster_file.close()
            sys.exit()
        del X,Y,imp1
        
################## MAIN: ####################
    
N_gram_covid , imposter_pairs , w2v_model = PreProcess()

has_recovered = recoverClusters()

if mode == 2:
    Sequence_from_mid(pair_num)
    
elif mode == 1:  
    Single_Sequence(pair_num)
else:
    
    recovery_count_path_file = Path(recovery_count_path)
    if recovery_count_path_file.is_file():
        os.remove("recovery_count.txt")
        os.remove("recovery_clustered.txt")
        os.remove("recovery_anomals.txt")
    Full_sequence()
    
if mode != 1:
    PostAnalyzeSARS()
    os.remove("recovery_count.txt")
    os.remove("recovery_clustered.txt")
    os.remove("recovery_anomals.txt")
    