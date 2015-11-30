#install.packages(partykit)
#install.packages(caret)
#install.packages(kknn)
#install.packages(e1071)
#install.packages(FNN)
#install.packages(RWeka)
library(partykit)
library(caret)
library(kknn)
library(e1071)
library(FNN)
library(RWeka)

set.seed(1911)
#IRIS Training and testing set
trainIndex <- createDataPartition(iris$Species, p=0.8, list = FALSE, times = 1)
head(trainIndex)
irisTrain <- iris[trainIndex,]
irisTest <- iris[-trainIndex,]

#Life expectancy training and testing sets
LifeExpectancy <- life_expectancy_csv
trainIndexLife <- createDataPartition(LifeExpectancy$Continent, p = 0.8,list = FALSE, times = 1)
LifeExpectancyTrain <- LifeExpectancy[trainIndexLife,]
LifeExpectancyTest <- LifeExpectancy[- trainIndexLife,]

#KNN-IRIS
cl<-factor(c(rep("setosa",nrow(irisTrain[irisTrain[,5]=="setosa",])), rep("versicolor", nrow(irisTrain[irisTrain[,5]=="versicolor",])), rep("virginica",nrow(irisTrain[irisTrain[,5]=="setosa",]))))
knnModel <- knn(irisTrain[,1:4], irisTest[,1:4],cl,k=3,prob=FALSE)
knnIrisCM <- confusionMatrix(knnModel, irisTest$Species)                 #Knn Confusion Matrix
knnIrisCM
knnIrisSummary <- summary(knnModel)
knnIrisSummary
plot(knnModel)

#KNN-LifeExpectancy
c1 = sum(LifeExpectancyTrain$Continent=="Africa")
c2 = sum(LifeExpectancyTrain$Continent=="Europe")
c3 = sum(LifeExpectancyTrain$Continent=="Asia")
c4 = sum(LifeExpectancyTrain$Continent=="North America")
c5 = sum(LifeExpectancyTrain$Continent=="South America")
cl_LE <- factor(c(rep("Africa",c1), rep("Europe", c2), rep("Asia", c3),rep("North America", c4), rep("South America", c5)))
KNNModel_LE <- knn(LifeExpectancyTrain[,c(3,5,7)], LifeExpectancyTest[,c(3,5,7)], cl_LE, k = 30, prob=FALSE) #KNN Model
KNNModel_LE
knnCM_LE <- confusionMatrix(KNNModel_LE, testset_L$Continent)
knnCM_LE
knnSummary_LE <- summary(KNNModel_LE)
knnSummary_LE
plot(KNNModel_LE)
#Naive Bayes - IRIS
head(iris)
names(iris)
x = iris[,-5]
y = iris$Species
model = train(x,y,'nb',trControl=trainControl(method='cv',number=10))
bayesPredict_iris <- predict(model$finalModel,x)
table(predict(model$finalModel,x)$class,y)
naive_iris <- NaiveBayes(iris$Species ~ ., data = iris)
naive_confusionMatrix_iris <- confusionMatrix(bayesPredict_iris, irisTest[,5])
bayes_summary_iris <- summary(bayesPredict_iris)
bayes_summary_iris
naive_confusionMatrix_iris
plot(naive_iris)

#Naive Bayes - Life Expectancy
head(LifeExpectancy)
names(LifeExpectancy)
x = LifeExpectancy[,-8]
y = LifeExpectancy$Continent
model = train(x,y,'nb',trControl=trainControl(method='cv',number=10))
bayesPredict_LE <- predict(model$finalModel,x)
table(predict(model$finalModel,x)$class,y)
naive_LE <- NaiveBayes(LifeExpectancy$Continent ~ ., data = LifeExpectancy)
naive_confusionMatrix_LE <- confusionMatrix(bayesPredict_LE, LifeExpectancyTest[,8])
bayes_summary_LE <- summary(bayesPredict_LE)
bayes_summary_LE
naive_confusionMatrix_LE
plot(naive_LE)


#C4.5 - Iris
library(RWeka)
data(iris)
TrainData <- irisTrain[,1:4]
TrainClasses <- irisTrain[,5]
JRipFit <- train(TrainData, TrainClasses,method = "JRip")
C4.5Model <- J48(Species~., irisTrain)
plot(C4.5Model)
C4.5predict <- predict(C4.5Model, irisTest)
con <- confusionMatrix(C4.5predict, irisTest[,5])
con
summary(C4.5Model)

#C4.5 - Life Expectancy
library(RWeka)
data(LifeExpectancy)
TrainData_LE <- LifeExpectancyTrain[,1:7]
TrainClasses_LE <- LifeExpectancyTrain[,8]
JRipFit <- train(TrainData_LE, TrainClasses_LE,method = "JRip")
C4.5Model_L <- J48(Continent~., trainset_L)
plot(C4.5Model_L)
C4.5predict_L <- predict(C4.5Model_L, LifeExpectancyTest)
con_L <- confusionMatrix(C4.5predict_L, LifeExpectancyTest[,8])
con_L
summary(C4.5Model_L)

#RIPPER - Iris
RipperModel_Iris <- JRip(Species ~., data=irisTrain)
RipperModel_Iris
RipperModelSummary_Iris <- summary(RipperModel_Iris)
RipperModelSummary_Iris
Ripperpredict_Iris <- predict(RipperModel_Iris, irisTest[,1:4])
RipperConfusionMatrix_Iris <- confusionMatrix(Ripperpredict_Iris, irisTest[,5])
RipperConfusionMatrix_Iris
RipperPredictSummary_Iris <- summary(Ripperpredict_Iris)
plot(Ripperpredict_Iris)
rpartTree <- rpart::rpart(RipperModel_Iris, data=irisTrain, method="anova")
rpartTree
plot(rpartTree)

#RIPPER- LifeExpectancy
RipperModel_LE <- JRip(Continent ~., data=LifeExpectancyTrain)
RipperModel_LE
RipperModelSummary_LE <- summary(RipperModel_LE)
RipperModelSummary_LE
Ripperpredict_LE <- predict(RipperModel_LE, LifeExpectancyTest[,1:7])
RipperConfusionMatrix_LE <- confusionMatrix(Ripperpredict_LE, LifeExpectancyTest[,5])
RipperConfusionMatrix_LE
RipperPredictSummary_LE <- summary(Ripperpredict_LE)
plot(Ripperpredict_LE)
rpartTree <- rpart::rpart(RipperModel_LE, data=LifeExpectancyTrain, method="anova")
rpartTree
plot(rpartTree)

#Oblique - Iris
oModel <- oblique.tree(Species~., data = irisTrain, split.impurity = "gini", oblique.splits = "only")
plot(oModel);text(oModel)
summary(oModel)
oPredict <- predict(oModel, irisTest, type = "class")
oConfusion <- confusionMatrix(oPredict, irisTest[,5])
oSummary <- summary(oPredict)

#Oblique - Life Expectancy
oModel_L <- oblique.tree(Continent~., data = LifeExpectancyTrain[,c(3,5,7,8)], split.impurity = "gini", oblique.splits = "only")
plot(oModel_L)
summary(oModel_L)
oPredict_L <- predict(oModel_L, LifeExpectancyTest[,c(3,5,7,8)], type = "class")
oConfusion_L <- confusionMatrix(oPredict_L, LifeExpectancyTest[,8])
oSummary_L <- summary(oPredict_L)
oConfusion
oSummary
oConfusion_L
oSummary_L
plot(oPredict);title(main="Oblique predict plot")
plot(oPredict_L);title(main="Oblique Life Expectancypredict plot")


