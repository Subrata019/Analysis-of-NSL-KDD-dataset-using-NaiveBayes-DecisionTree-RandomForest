# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 19:31:50 2019

@author: Puchu
"""

#importing libraries

import pandas as pd 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
 


#importing dataset
#dataset = pd.read_csv('C:\\Users\\Puchu\\.spyder-py3\\kddtrain+_20percent.csv') 
file_handler = open("C:\\Users\\Puchu\\.spyder-py3\\KDDTrain+.csv", "r") 
  
# creating a Pandas DataFrame 
# using read_csv function  
# that reads from a csv file. 
dataset = pd.read_csv(file_handler, sep = ",") 
  
# closing the file handler 
file_handler.close() 
  
# creating a dict file  
protocol_type = {'tcp': 1,'udp': 2,'icmp':3} 
flag = { 'OTH':1,'REJ':2,'RSTO':3,'RSTOS0':4,'RSTR':5,'S0':6,'S1':7,'S2':8,'S3':9,'SF':10,'SH':11}
service = {'aol':1,'auth':2,'bgp':3,'courier':4,'csnet_ns':5,'ctf':6,'daytime':7,'discard':8,'domain':9,'domain_u':10,'echo':11,'eco_i':12,'ecr_i':13,'efs':14,'exec':15,'finger':16,'ftp':17,'ftp_data':18,'gopher':19,'harvest':20,'hostnames':21,'http':22,'http_2784':23,'http_443':24,'http_8001':25,'imap4':26,'IRC':27,'iso_tsap':28,'klogin':29,'kshell':30,'ldap':31,'link':32,'login':33,'mtp':34,'name':35,'netbios_dgm':36,'netbios_ns':37,'netbios_ssn':38,'netstat':39,'nnsp':40,'nntp':41,'ntp_u':42,'other':43,'pm_dump':44,'pop_2':45,'pop_3':46,'printer':47,'private':48,'red_i':49,'remote_job':50,'rje':51,'shell':52,'smtp':53,'sql_net':54,'ssh':55,'sunrpc':56,'supdup':57,'systat':58,'telnet':59,'tftp_u':60,'tim_i':61,'time':62,'urh_i':63,'urp_i':64,'uucp':65,'uucp_path':66,'vmnet':67,'whois':68,'X11':69,'Z39_50':70}

# traversing through dataframe 
# protocol_type,flag, service column and writing 
# values where key matches 
dataset.protocol_type = [protocol_type[item] for item in dataset.protocol_type] 
dataset.flag = [flag[item] for item in dataset.flag] 
dataset.service = [service[item] for item in dataset.service] 
print(dataset) 

#printing the head of the dataset
dataset.head() 


#splitting dataset into features and class
X = dataset.iloc[:, 0:41].values  
y = dataset.iloc[:, 41].values  


#splitting dataset into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


#training the algorithm
# Create the RandomForest model
classifier1 = RandomForestClassifier(n_estimators=200, bootstrap = True, max_features = 'sqrt')
#creating the Decision Tree model
classifier2 = DecisionTreeClassifier(criterion = "entropy", max_depth=10)
#creating the Naive Bayes
classifier3 = GaussianNB(priors=None, var_smoothing=1e-09)


# Fit on training data
# training RandomForest
clf1 = classifier1.fit(X_train, y_train)
#training decision tree
clf2 = classifier2.fit(X_train, y_train)
#training Naive Bayes
clf3 = classifier3.fit(X_train, y_train)



#prediction
#for RandomForest
pred1 = clf1.predict(X_test)
#for decision tree
pred2 = clf2.predict(X_test)
#for Naive Bayes
pred3 = clf3.predict(X_test)


#confusion matrix.
#for RandomForest
cm1=confusion_matrix(y_test,pred1)
cm1
#for decision tree
cm2=confusion_matrix(y_test,pred2)
cm2
#for Naive Bayes
cm3=confusion_matrix(y_test,pred3)
cm3


#accuracy matrix
#for RandomForest
ac1=100*accuracy_score(y_test,pred1)
ac1
#for decision tree
ac2=100*accuracy_score(y_test,pred2)
ac2
#for Naive Bayes
ac3=100*accuracy_score(y_test,pred3)
ac3

from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(clf2, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

objects = ('NaiveBayes','Decision Tree', 'RandomForest',)
y_pos = np.arange(len(objects))
performance = [ac3,ac2,ac1]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Usage')
plt.title('Classifier Accuracy')
 
plt.show()
