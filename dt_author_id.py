
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()



clf = DecisionTreeClassifier(min_samples_split= 40)

clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

accuracy = accuracy_score(labels_test, pred)

print accuracy

