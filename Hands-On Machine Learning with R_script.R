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






