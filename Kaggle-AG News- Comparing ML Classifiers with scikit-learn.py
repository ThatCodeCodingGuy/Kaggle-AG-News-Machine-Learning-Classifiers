#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Loading the necessary modules 
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn import metrics


# In[2]:


# Loading datasets by pandas
dataset = pd.read_csv(r"C:\Users\User\Downloads\train.csv")
testset = pd.read_csv(r"C:\Users\User\Downloads\test.csv")


# In[3]:


# Looking at the first five rows to gain insight
dataset.head()


# In[4]:


dataset.shape
# Output: (120000, 3)
testset.shape
# Output:  (7600, 3)
dataset.columns
# Output: Index(['Class Index', 'Title', 'Description'], dtype='object')
testset.columns
# Output: Index(['Class Index', 'Title', 'Description'], dtype='object')


# In[5]:


# Forming our tranining and test sets
X_train_normal  =  dataset['Title']+' '+dataset['Description'] # Joining the contents of "Title" and "Description" together
X_test_normal   =  testset['Title']+'  '+testset['Description']
y_train  =   dataset['Class Index'].apply(lambda x: x-1)
y_test =    testset['Class Index'].apply(lambda x: x-1)


# In[6]:


# Using TF-IDF Vectorizer to convert text into numerical vector

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
X_train = vectorizer.fit_transform(X_train_normal)
    
X_test = vectorizer.transform(X_test_normal)


# Defining a function consisting of models to be used in classification

def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)


        
    print("classification report:")
    print(metrics.classification_report(y_test, pred))

    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))

    
    return score

results = []
for clf, name in (
        (Perceptron(max_iter=50), "Perceptron"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Training Liblinear model
    results.append(benchmark(LinearSVC(penalty=penalty, dual=False,
                                       tol=1e-3)))

    # Training SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=50,
                                           penalty=penalty)))

# Training SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=50,
                                       penalty="elasticnet")))

# Training sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01)))
results.append(benchmark(BernoulliNB(alpha=.01)))
results.append(benchmark(ComplementNB(alpha=.1)))

# Training Linear SVC with L1-based feature selection
print('=' * 80)
print("LinearSVC with L1-based feature selection")
results.append(benchmark(Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
                                                  tol=1e-3))),
  ('classification', LinearSVC(penalty="l2"))])))

# Training Nearest Centroid 
print("="* 80)
print("Nearest Centroid")
results.append(benchmark(NearestCentroid()))


# Training Decision Tree Classifier 
print("="* 80)
print("Decision Tree Classifier")
results.append(benchmark(DecisionTreeClassifier()))

# Training Gradient Boosting Classifier 
print("=" * 80)
print("Gradient Boosting Classifier")
results.append(benchmark(GradientBoostingClassifier(n_estimators = 100, learning_rate=0.03, max_depth=3)))

# Training Extra Tree Classifier 
extra_tree = ExtraTreeClassifier()
print("="* 80)
print("Extra Tree Classifier")
results.append(benchmark(BaggingClassifier(extra_tree)))

