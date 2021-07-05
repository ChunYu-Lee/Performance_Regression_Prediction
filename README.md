# Performance_Regression_Prediction
This is a model to predict the performance regression in commit level dataset.  
Installed Packages: dtreeviz, shapash, imbalanced-learn, scipy, sklearn.  
  
In this dataset, there are 49 features with 1 label(Hit/Dismiss).
The goal is to find out the potential commits would cause the performance regression when the developer push them into the system, and FN(False Negative) is the metric we try to improve.  
The features selection was implemented using shapash before the model training.  
There are two model used to train. One is KNN(K-Nearest Neighbors), and the other is RFC(Random Forest Classifier).  

