# **Kaggle-AG-News-Machine-Learning-Classifiers**
This project uses scikit-learn's built-in different Machine Learning (ML) classifiers and the "Perceptron" classifier to categorize the text data of the third version of AG News dataset on Kaggle.
This project uses scikit-learn's built-in different Machine Learning (ML) classifiers and "Perceptron" classifier to categorize the text data of the third version of AG News dataset on Kaggle.

I use simple TF-IDF Vectorizer to convert text data into vectors that is to be trained on 10 ML classifiers and on a Perceptron that are built into scikit-learn. The ML classifiers I use are:

1. K-Nearest Neighbor
2. Linear Support Vector Machine (LinearSVC)
3. Stochastic Gradient Descent Classifier (SGD)
4. Multinomial Naive Bayes (MultinomialNB)
5. Bernoulli Naive Bayes (BernoulliNB)
6. Complement Naive Bayes (ComplementNB)
7. Nearest Centroid (Also known as "Rocchio Classifier" when used for text classification with TF-IDF vectorization)
8. Decision Tree
9. Gradient Boosting
10. Extra Tree

## **Results:**
The accuracies of tested ML classifiers which can be looked at after running the codes as well:

- Perceptron(max_iter=50): 0.890
- KNeighborsClassifier(n_neighbors=10): 0.911
- LinearSVC(dual=False, tol=0.001) (with "l2" penalty): 0.919
- SGDClassifier(max_iter=50) (with "l2" penalty): 0.909
- LinearSVC(dual=False, penalty='l1', tol=0.001): 0.917
- SGDClassifier(max_iter=50, penalty='l1'): 0.886
- SGDClassifier(max_iter=50, penalty='elasticnet'): 0.902
- Naive Bayes:
   1. MultinomialNB(alpha=0.01): 0.903
   2. BernoulliNB(alpha=0.01): 0.901
   3. ComplementNB(alpha=0.1): 0.908
- LinearSVC with L1-based feature selection: 0.919
- Nearest Centroid: 0.857
- Decision Tree Classifier: 0.819
- GradientBoostingClassifier(learning_rate=0.03): 0.736
- Extra Tree Classifier: 0.88

## **What have I learned?**
1. For this classification, nearly all of the classifiers seem to be sufficient, perhaps, with the exception of Gradient Boosting and Decision Tree Classifers. However, it needs to be stressed that from only one dataset, it would be a huge misstep to generalize which ML classifiers to use in text classification. Much more datasets should be used at the same time to achieve such a goal.
2. Despite not being available in the codes, by doing some other tests before forming the last version of this project, I realized that three models (GaussianNB, Quadratic Discriminant Analysis, Linear Discriminant Analysis) didn't work well because of sparse matrices. Thanks to this situation, I indeed managed to check the truthness of the general assumption that GaussianNB, Quadratic Discriminant Analysis, and Linear Discriminant Analysis are not well suited models to be used for text classification (when they involve sparse matrices).

The link to the data: https://www.kaggle.com/jmq19950824/agnews
