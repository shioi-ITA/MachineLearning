#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#########################################################

features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]

from sklearn import svm

# linear kernel
#clf = svm.SVC(kernel="linear")

# rbf kernel and various C parameters
#clf = svm.SVC(kernel="rbf")
#clf = svm.SVC(C=10.0, kernel="rbf")
#clf = svm.SVC(C=100.0, kernel="rbf")
#clf = svm.SVC(C=1000.0, kernel="rbf")
clf = svm.SVC(C=10000.0, kernel="rbf")

t0=time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t0=time()
predictions = clf.predict(features_test)
print "test time:", round(time()-t0, 3), "s"

#print(clf.score(features_test, labels_test))
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, predictions)
print "accuracy: ", accuracy

print "10th element: ", predictions[10], \
      "\n26th element: ", predictions[26], \
      "\n50th element: ", predictions[50]

count=0
for pred in predictions:
	if pred==1:
		count +=1

print """The number  predicted to be in the "Chris": """, count
