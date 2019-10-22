# This is the R code for simple product recommendation 
# experiment with Naive Bayes
# Patrick Rotzetter, protzetter@buewin.ch, October 2019

# load required libraries
library(dplyr)
library(e1071)
library(C50)
library(rpart)
library(rpart.plot)

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

train<-select(train,-Age.Group)
test<-select(test,-Age.Group)

# Naive Bayes classifier

classifier<- naiveBayes(Product ~ ., data=train,laplace=0)
print(classifier)
predTrain<-predict(classifier, train, type='class')
predTest<-predict(classifier, test, type='class')
testTable=table(test$Product, predTest)
sum(diag(testTable))/sum(testTable)
trainTable=table(train$Product, predTrain)
sum(diag(trainTable))/sum(trainTable)
#print(trainTable)
#print(testTable)

# Decision tree

train<-select(train,-Product)
test<-select(test,-Product)
model<-C5.0(train,yTrain, trials = 1)
plotpredTrain<-predict(model,train)
trainTable=table(yTrain, predTrain)
sum(diag(trainTable))/sum(trainTable)
predTest<-predict(model,test)
testTable=table(yTest, predTest)
sum(diag(testTable))/sum(testTable)

