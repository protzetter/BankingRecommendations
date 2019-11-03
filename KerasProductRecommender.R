# This is the R code for simple product recommendation 
# experiment with Keras
# Patrick Rotzetter, protzetter@buewin.ch, October 2019

# load required libraries
library(keras)
library(dplyr)


#Read training and test files

trainData<-read.csv("productrecommendations.csv",header=TRUE,sep = ",", na.strings = "#DIV/0!")


# split training and test data

set.seed(101) # Set Seed so that same sample can be reproduced in future also

# Now Selecting 80% of data as sample from total 'n' rows of the data  
sample <- sample.int(n = nrow(trainData), size = floor(.8*nrow(trainData)), replace = F)
train <- trainData[sample, ]
test  <- trainData[-sample, ]
yTrain<-train$Product
yTest<-test$Product

#remove unwanted features
train<-select(train,-Age.Group)
test<-select(test,-Age.Group)
train<-select(train,-Product)
test<-select(test,-Product)

# one hot encode target values
yTrainInt<-as.integer(yTrain)
yTestInt<-as.integer(yTest)

yTraincat<-to_categorical(yTrainInt)
yTestCat<-to_categorical(yTestInt)

#convert input to matrix
trainMatrix<-as.matrix(as.data.frame(lapply(train, as.numeric)))
testMatrix<-as.matrix(as.data.frame(lapply(test, as.numeric)))

# Initialize a sequential model
model <- keras_model_sequential() 

# Add layers to the model
model %>% 
  layer_dense(units = 8, activation = 'relu', input_shape = c(10)) %>% 
  layer_dense(units = 15, activation = 'softmax')

# Compile the model
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

# Fit the model 
model %>% fit(
  trainMatrix, 
  yTraincat, 
  epochs = 200, 
  batch_size = 5, 
  validation_split = 0.2
)


# test the model
# Predict the classes for the test data
classes <- model %>% predict_classes(testMatrix, batch_size = 128)
proba<-model %>% predict_proba(testMatrix, batch_size = 128)
# Confusion matrix
table(as.array(k_argmax(yTestCat)), classes)

# explain model with Lime
library(lime)


model_type.keras.engine.sequential.Sequential <- function(x, ...) {
  "classification"}


predict_model.keras.engine.sequential.Sequential <- function(x, newdata, type, ...) {
  pred <- predict_proba(object = x, x = as.matrix(newdata))
  return(data.frame(pred))
}


# Test our predict_model() function
predict_model(x = model, newdata = testMatrix, type = 'raw') %>%
  tibble::as_tibble()



# Run lime() on training set
explainer <- lime::lime(
  x              = as.data.frame(trainMatrix), 
  model          = model, 
  bin_continuous = FALSE)

explanation <- lime::explain(
  as.data.frame(testMatrix[1:5,]), 
  #test,
  explainer    = explainer, 
  n_labels     = 1, 
  n_features   = 5,
  kernel_width = 0.5)


plot_features(explanation) +
  labs(title = "LIME Feature Importance Visualization",
       subtitle = "Hold Out (Test) Set, First 10 Cases Shown")



