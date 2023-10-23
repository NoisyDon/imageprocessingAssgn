# import all the required packages
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import os
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import pickle

Categories=['Healthy','Herniated'] #Categories of data frame

flat_data_arr=[]
target_arr=[]
datadir='DataSet' #Data directory of data set

for i in Categories: #Locating folder containing data set
  print(f'loading... category : {i}') #Progress message
  path=os.path.join(datadir,i)
  
  for img in os.listdir(path): #Reading and resizing of data
    img_array=imread(os.path.join(path,img)) #Reading
    img_resized=resize(img_array,(150,150)) #Resizing
    flat_data_arr.append(img_resized.flatten())
    target_arr.append(Categories.index(i))
    
  print(f'loaded category:{i} successfully') #Progress messsage
  
flat_data=np.array(flat_data_arr)
target=np.array(target_arr)
df=pd.DataFrame(flat_data)
df['Target']=target #Name of data frame
print(df) #Display data frame


healthy=df.iloc[:,:-1] #Splittnig of data set into train and tes
herniated=df.iloc[:,-1]
healthy_train,healthy_test,herniated_train,herniated_test=train_test_split(healthy,herniated,test_size=0.20,random_state=0,stratify=herniated)
print('Splitted Successfully') #Progress message


param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']} #Parameters of SVM training


svc=svm.SVC(probability=True) #Use of 5-fold cross validation

print("The training of the model is started, please wait for while as it may take few minutes to complete") #Progress message

model=GridSearchCV(svc,param_grid) #Parameters of models used for training later

model.fit(healthy_train,herniated_train) #Training of model
print('The Model is trained well with the given images') #Progress message
 
print(model.best_params_) #Display best model

herniated_pred=model.predict(healthy_test) #Perform prediction
print("The predicted Data is :\n" + str(herniated_pred)) #Display predicted data

print("The actual data is:") #Display correct data
print(np.array(herniated_test)) 

print(f"The model is {accuracy_score(herniated_pred,herniated_test)*100}% accurate") #Display accuracy of prediction

# save model as pickle
pickle.dump(model,open('model_new.p','wb')) #Perform pickling process
print("Pickle is dumped successfully") #Progress message

con_mat = confusion_matrix(herniated_true=herniated_test,herniated_pred=herniated_pred)
print(con_mat) #Display confusion matrix

report = classification_report(herniated_test,herniated_pred,target_names=Categories)
print(report) #Display classification report