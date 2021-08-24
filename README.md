# Performance_Regression_Prediction
This is a model to predict the performance regression in Git commit level dataset.
Installed Packages: Dtreeviz, Shapash, Imbalanced-learn, SciPy, Scikit-learn.
In this dataset, there are 49 features with 1 label(Hit/Dismiss). The goal is to determine the potential commits that would cause the performance regression when the developer pushes them into the system, and FN(False Negative) is the metric we try to improve.
The features selection was implemented using Shapash before the model training.
There are two models used to train. One is KNN(K-Nearest Neighbors), and the other is RFC(Random Forest Classifier).

