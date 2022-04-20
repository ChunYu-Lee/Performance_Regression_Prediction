# Performance_Regression_Prediction
Used RFC, KNN to build models to predict the potential commit would cause performance regression. <br />
This model can predict the potential performance regression commit so the testing team could test the commit before pushing into the system. <br /> 
Tool used: Python, Imbalanced-learn, SciPy, Scikit-learn. <br />
In this dataset, there are 49 features with 1 label(Hit/Dismiss). The goal is to determine the potential commits that would cause the performance regression when the developer pushes them into the system, and FN(False Negative) is the metric we try to improve.
The features selection was implemented using Shapash before the model training.
There are two models used to train. One is KNN(K-Nearest Neighbors), and the other is RFC(Random Forest Classifier).  <br />
Please run the Performance_Regression_Prediction.py to get the result.

