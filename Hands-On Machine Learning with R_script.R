### R scripts based on the book by Boehmke&Greenwell

#Libraries ----
library(tidyverse)
library(ggplot2)
library(dplyr)
library(rsample) #for resampling procedures
library(caret) #resampling and model training
library(h2o) # resampling and model training

# 1.4 Data sets ----
## Used in supervised regression
ames <- AmesHousing::make_ames() # Sale_Price of house as response, want to predict based on attributes of the house

##Used in supervised binomial classification
data("attrition", package = "modeldata") #Attrition (yes/no) as response. use employee attributes to predict if they will attrit (leave the company)

##Used in supervised multinomial classification
mnist<-dslabs::read_mnist() # V785 (0,1,...,9). use attributes about "darkness" of pixels in images to predict if the handwritten number is 0,1,...,9 (60000 images for train / 10000 test)
head(mnist$train$labels) #response variable

## Used in unsupervised basket analysis
my_basket <- readr::read_csv("https://koalaverse.github.io/homlr/data/my_basket.csv") #no response variable. Use attributes of each basket to identify common groupings of items purchased together


# 2 Model process----
## Multiple package examples given in the book. I will mostly ignore the base R unless very clear

## H2O is a cloud service that can be used for a lot of ML tasks. Will here show both with and without
### h2o.no_progress() #turn off h2o progress bar
### h2o.init() # launch h2o
### as.h2o(<my-data-frame>) # used to convert any data frame to a H2O object (i.e. import it to the H2O cloud)

ames.h2o <- as.h2o(ames) #make the ames data into an h2o object
churn <- attrition %>%
  mutate_if(is.ordered, .funs = factor, ordered = FALSE) #have to convert ordered factors into unordered to use the attrition dataset in h2o
churn.h2o <- as.h2o(churn) #attrition data as h2o object

## 2.2 Data splitting ----
### Training set: used to develop feature sets, train algorithms, tune hyperparameters, compare models, etc needed to choose final model
### Test set: using the final model, use these data to estimate an unbiased assesment of model performance (refered to as "generalization error")
### 60%-40%, 70-30 and 80-20 typically recommended (training-test). Too much training can lead to overfitting (i.e. not generalizable), while too little gives weaker final model

### Simple random sampling: useful when very big dataset
#### Using caret package
set.seed(123) # for reproducibility
index_2 <- createDataPartition(ames$Sale_Price, p = 0.7,
                               list = FALSE)
train_2 <- ames[index_2, ]
test_2 <- ames[-index_2, ]
#### Using rsample package
set.seed(123) # for reproducibility
split_1 <- initial_split(ames, prop = 0.7)
train_3 <- training(split_1)
test_3 <- testing(split_1)
#### Using h2o package
split_2 <- h2o.splitFrame(ames.h2o, ratios = 0.7,
                          seed = 123)
train_4 <- split_2[[1]]
test_4 <- split_2[[2]]

### Stratified sampling: usually by Y distribution to make sure the split is even
#### More common with classification problems where the response is very imbalanced (e.g. 90% yes, 10 no), or in small dataset, or skewed distribution of Y
#### With a continuous response variable, stratified sampling will segment 𝑌 into quantiles and randomly sample from each

#### orginal response distribution
table(attrition$Attrition) %>% prop.table() #unbalanced
#### stratified sampling with the rsample package
set.seed(123)
split_strat <- initial_split(attrition, prop = 0.7,
                             strata = "Attrition") #strata specifies the variable to stratify by
train_strat <- training(split_strat)
test_strat <- testing(split_strat)

### Class imbalances: up-sampling and down-sampling
#### Down-sampling: Down-sampling balances the dataset by reducing the size of the abundant class(es) to match the frequencies in the least prevalent class. Need big N for this
##### keeps all data from rare class, and randomly selects similar number from the other classes
#### Up-sampling: used when low sample sizes. Creates new cases of the rarer samples, using repetition/bootstrapping

## 2.3 Creating models in R ----
### Standard Y~X formula method (Y~x1+x2+x1:x2 etc)
### OR: XY interface (x= c(...,...), y=...)
### OR: variable name specification (x=c("...","...",...), y="...") #used in h2o package

### Meta engines (e.g. caret): allows a more consistent way of modelling, but often less flexible than using direct engines/packages (e.g. lm() vs glm(fam="gaussian") give same model, but different specification)

## 2.4 Resampling methods ----
### Training data should be further split into a training set and a validation/holdout set (separate from the untouched test set) - that way you can validate the models prior to finding the final model
### Resampling (e.g. k-fold cross validation / bootstrapping) of the training data allows us to repeatedly fit a model of interest to parts of the training data and test its performance on other parts

### k-fold cross validation (CV)
#### Make k splits, train on k-1 and test on the last one. Then repeat this k times, each time testing/validating the data on a different set
#### Can often be performed directly in ML functions (e.g. h2o.glm), or externally (but then you need a process to apply the ML model to each resample)
#### example of external:
rsample::vfold_cv(ames, v = 10)

### Bootstrapping
#### train model using selected data in the bootstrap and then test on the out-of-bag (OOB) samples (i.e. the ones not included in the bootstrap sample). Repeat several times
#### example external
bootstraps(ames, times = 10)
#### Often internally build in ML algorithms (e.g. in random forest)

## 2.5 Bias variance trade-off ----
### prediction error comes from either bias or due to variance, often a tradeoff
### Hyperparameter tuning
#### k-nearest neightbor model: k is the hyperparameter, determines the predicted value based on the k nearest observations in the training data
#### Manually change hyperparameters and test with f.ex. k-fold CV OR use a grid search to do it for you
#### Grid search helps determining the optimal parameter, where neither model variance or model bias is too large

## 2.6 Model evaluation ----
### Regression models for loss functions
#### Most common: MSE (mean squared error, more weight on large errors), RMSE (root of MSE, error in same unit as response variable)
#### Other useful: deviance (often more useful if on-gaussian distribution of response), RMSLE (similar to RMSE, but log transformed first. Good when large range in response values)

### Classification models: 
#### Main types: misclassification (overal error, how many (in percentage) cases very wrongly classified), mean per class error (similar to misclass, but better when unbalanced classes)
##### MSE (takes predicted probability of the correct class into account, so higher error if the correct class was predicted as less likely)
##### cross-entropy/deviance (similar to MSE, but punishes high confidence in wrong answer even more), gini index (for tree-based methods)

#### Confusion matrix: matrix with predicted vs actual category. 
##### If binary classes: True positive (predicted correct), false positive (predicted x to happen, but it didnt), false negative (did not predict x, but it happened), true negative
###### Accuracy (how often correct, opposite of misclassification); precision (how many (in %) true pos compared to false pos)
###### Sensitivity (true pos compared to false neg, how many of the events that occured did we predict); specificity (true neg compared to false pos)
###### AUC: area under the curve. Want the line to be in the upper left-hand corner

## 2.7 Simple test ----

### Stratified sampling with the rsample package
set.seed(123)
split <- initial_split(ames, prop = 0.7,
                       strata = "Sale_Price")
ames_train <- training(split)
ames_test <- testing(split)

### Specify resampling strategy, 10-fold, repeated 5 times
cv <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 5
)
### Create grid of hyperparameter values
hyper_grid <- expand.grid(k = seq(2, 25, by = 1)) # k from 2 to 25
### Tune a k-nearest neighbor (knn) model using grid search (uses caret package)
# knn_fit <- train( 
  Sale_Price ~ ., #model, ~. means all predictor variables
  data = ames_train,
  method = "knn",
  trControl = cv,
  tuneGrid = hyper_grid,
  metric = "RMSE" #loss function
) #OBS: takes some time to run! Remove # from knn_fit to run

knn_fit # gives the CV result. model with k=6 is best here
ggplot(knn_fit) # plot showing that 6 is best, but 7 is also very good here

###  We have found our optimal knn model for the dataset, but it doesnt mean it is the best possible model. 
### We have also not considered potential feature and target engineering options

# 3 Feature and Target engineering ----















