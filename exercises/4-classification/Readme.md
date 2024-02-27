# Classification on `emnist`

## 1. Create `Readme.md` to document your work

Name: Marlene Lin

## 2. Classify ~~all symbols~~ letters a -> g

### Subset the data

Select only the lowercase letters (a, b, ..., g) for classification using str.isupper(). The label are recoded to 1 - 7.

### Choose a model

Convolutional Neural Network because it's suitable for implementing image classification tasks. 

The simple convolution neural network consists of two convolution layers, each followed by a max pooling layer, where the convolution layers perform feature extractions through convolutions in 2-dimension of the input images and the max pooling layer summarize the extraction and reduce spatial dimensions through taking regional max values. Then, a flattening layer converts the 2D tensor in to 1D, before the two dense layers which provide weighted summations to scale the previous layer size, ultimately reaching the number of labels.

### Train away!

The train and test split is stratified, 9:1. The data is in a dataframe and should be converted to tensors as input to the tensorflow neural network. 

### Evaluate the model

Evaluate the models on the test set, analyze the confusion matrix to see where the model performs well and where it struggles. The display_metrics method is modified to work with the cases in this ps. Per the confusion matrix and the accuracy plot near 100% accuracy is reached on the test set, with slight error on misclasifying e as d. The macro-averaged accuracy on the test set is 98.4%, precision 97.6%, recall 97.2%, f1  score 97.4%

### Investigate subsets

I don't think there is anything particularly confusin in this case. 

### Improve performance

Intuitively, doing data augmentation of flipping the a-g letters could help with classification between confusing letters, like up and down for d and e, because d has more of a tail？But could make things worse for case like distinguishing b and d, if you are flipping left and right. Adding more layers and using smaller convolution windows might be helpful for picking up the more nuisance.

## 3. Model showdown: upper vs lowercase on abcXYZ

### Subset the data

Select out the set of upper- and lowercase (a, b, c, x, y z, A, B, C, X, Y, Z). Note that some of these classes can be confusing (e.g., x and y).

### Train and tune models

Perform a full model training and hyperparameter tuning.

1. Select candidate models, hyperparameter options, and evaluation metric

Candidate models include gradient boost classifier, extreme gradient boost classifier, and random forest classifier (objective binary classificaiton). The evaluation metric is accuracy.

2. Set aside a validation hold-out dataset
10% is set asside. 

3. Train models over K splits (use k-fold or train/test split)
    1. Split train using k-fold with the number of folds equal to the number of parameter combinations
    2. Train on k-fold split
    3. Record performance of each set of parameters
    4. Use winning set of parameters to train model on full training set
    5. Record each model's performance on that split's test set

The set of hyperparameters examine include max_depth (3, 7), and number of base estimators for the ensemble models (50, 100). So the tuning is based on 4-Stratified fold cross validation.

4. Evaluate model performance and promote one model as the winner

The best param combination for random_forest is n_estimator = 100 and max_depth = 7, same for xgboost, although the latter achieves a higher average cv accuracy of 84.3%.


5. Train winning model on both train + test
6. Check model performance on the validation hold-out
Using the XGBoost with max_depth = 7 and 100 base learners the accuracy achieved on the validation set is 84.5%
