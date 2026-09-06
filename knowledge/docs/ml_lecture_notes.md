# Machine Learning - Lecture Notes

## 1. Supervised vs Unsupervised Learning
Supervised learning uses labelled data: the model learns a mapping from inputs X to
known outputs y. Examples: linear regression, logistic regression, decision trees.
Unsupervised learning works on unlabelled data to find structure. Examples:
k-means clustering, PCA, hierarchical clustering.

## 2. Bias-Variance Tradeoff
- High bias (underfitting): model too simple, high error on train and test.
- High variance (overfitting): model memorises training data, low train error but
  high test error.
- Regularisation (L1/Lasso, L2/Ridge), more data, and cross-validation reduce variance.

## 3. Gradient Descent
An optimisation algorithm that iteratively updates parameters in the direction of the
negative gradient of the loss function. Learning rate controls the step size. Variants:
batch, stochastic (SGD), and mini-batch gradient descent. Adam combines momentum and
adaptive learning rates.

## 4. Evaluation Metrics
- Classification: accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix.
- Regression: MAE, MSE, RMSE, R-squared.
- Always evaluate on a held-out test set; use a validation set for hyperparameter tuning.

## 5. Train/Validation/Test Split
A common split is 70/15/15. K-fold cross-validation (k=5 or k=10) gives a more robust
estimate of generalisation performance when data is limited.

## 6. Exam Focus Areas
The mid-term exam covers: bias-variance tradeoff, gradient descent maths, precision vs
recall, and when to use L1 vs L2 regularisation. The exam is scheduled for 15 October.
