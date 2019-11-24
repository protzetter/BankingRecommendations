# This is the R code for simple product recommendation 
# experiment with Naive Bayes
# Patrick Rotzetter, protzetter@buewin.ch, October 2019

# load required libraries
library(dplyr)
library(e1071)
library(C50)
library(partykit)
library(stats)
library(factoextra)


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
myTree <- C50:::as.party.C5.0(model)
plot(myTree[2])
plot(myTree[33])
modelRules<-C5.0(train,yTrain, trials = 1, rules=TRUE)
plotpredTrain<-predict(modelRules,train)
trainTable=table(yTrain, predTrain)
sum(diag(trainTable))/sum(trainTable)


# experiment k-means
fullTrainMatrix<-as.matrix(as.data.frame(lapply(train, as.numeric)))
fullTestMatrix<-as.matrix(as.data.frame(lapply(test, as.numeric)))

# Initialize total within sum of squares error: wss
wss <- 0

# For 1 to 15 cluster centers
for (i in 1:15) {
  km.out <- kmeans(fullTrainMatrix, centers = i, nstart=20)
  # Save total within sum of squares to wss variable
  wss[i] <- km.out$tot.withinss
}

# Plot total within sum of squares vs. number of clusters
plot(1:15, wss, type = "b", 
     xlab = "Number of Clusters", 
     ylab = "Within groups sum of squares")


# Set seed
set.seed(1)
par(mfrow = c(2, 3))
for(i in 1:6) {
  # Run kmeans() on x with three clusters and one start
  km.out <- kmeans(fullTrainMatrix, centers=3, nstart=1)
  
  # Plot clusters
  plot(fullTrainMatrix, col = km.out$cluster, 
       main = km.out$tot.withinss, 
       xlab = "", ylab = "")
}
cluster<-kmeans(fullTrainMatrix, centers=3, iter.max = 20)
print(cluster)
fviz_cluster(cluster, data = fullTrainMatrix,
             #palette = c("#00AFBB","#2E9FDF", "#E7B800", "#FC4E07"),
             ggtheme = theme_minimal(),
             main = "Partitioning Clustering Plot"
)

#random forest
require('randomForest')
modelRF <- randomForest(Product ~ ., data=train, ntree=1)
plot(modelRF$err.rate[, 1], type = "l", xlab = "nombre d'arbres", ylab = "erreur OOB")

