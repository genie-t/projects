set.seed(1101)
setwd("C:/Users/genie/OneDrive/R (DSA1101)/Data_export")

diab <- read.csv("diabetes_5050.csv")
head(diab)
attach(diab)

Diabetes_binary <- as.factor(Diabetes_binary)
HighBP <- as.factor(HighBP)
HighChol <- as.factor(CholCheck)
Smoker <- as.factor(Stroke)
HeartDiseaseorAttack <- as.factor(HeartDiseaseorAttack)
PhysActivity <- as.factor(PhysActivity)
Fruits <- as.factor(Fruits)
Veggies <- as.factor(Veggies)
HvyAlcoholConsump <- as.factor(HvyAlcoholConsump)
AnyHealthcare <- as.factor(AnyHealthcare)
NoDocbcCost<- as.factor(NoDocbcCost)


# to get statistics of Diabetes binary response variable
summary(Diabetes_binary)
table(Diabetes_binary)

# summary statistics for all variables
summary(diab)

# correlation between response and input variables
M1 <- glm(Diabetes_binary~.,data = diab, family =  binomial(link = "logit"))
summary(M1)

#drop insignificant variables
diab = diab[, -c(6,9,10,11,13,14)]

# train and test data
n = dim(diab)[1] # sample size = 70692
test = sample(1:n, 14138) #14138 is about 20% of sample size

train.Y = Diabetes_binary[-test]
test.Y = Diabetes_binary[test]

testDT.X = diab[test,][,-1]
testDT.Y = diab[test,][,1]

# decision tree classifier
library("rpart")
library("rpart.plot")
fit <- rpart(Diabetes_binary~.,method = "class",
data = diab[-test,],
control = rpart.control(minsplit = 5000), #10% of train size
parms = list(split = 'information'))

rpart.plot(fit, type = 4, extra =2, clip.right.labs = FALSE, 
main = "Decision Tree")

# goodness of fit of decision tree

predDT <- predict(fit, newdata = testDT.X, type="class")
confusion.matrixDT = table(predDT, testDT.Y)

accDT = sum(diag(confusion.matrixDT))/sum(confusion.matrixDT)

accDT #accuracy is 0.7254209

library(ROCR)
predDT <- predict(fit, testDT.X, type="class")
predDT <- as.numeric(paste(predDT))
predictionDT <- prediction(predDT, testDT.Y)
rocDT <- performance(predictionDT, measure = "tpr", x.measure = "fpr")
plot(rocDT, col = "red", main = "ROC plot for various classifiers" )

aucDT = performance(predictionDT, measure = "auc")
aucDT@y.values[[1]] # 0.7258866

# calculating by probabilities instead of class
predDT <- predict(fit, testDT.X, type="prob")
scoreDT <- predDT[,2]
predictionDT <- prediction(scoreDT, testDT.Y)
rocDT <- performance(predictionDT, measure = "tpr", x.measure = "fpr")
plot(rocDT, col = "blue", add =TRUE) 

aucDT = performance(predictionDT, measure = "auc")
aucDT@y.values[[1]] # 0.7668546, higher than using class


# naive bayes classifier

library(e1071)
NB <- naiveBayes(Diabetes_binary~., diab[-test,])

# goodness of fit of Naive Bayes
predNB <- predict(NB, newdata = testDT.X, type='raw')
confusion.matrixNB = table(predNB, testDT.Y)

accNB = sum(diag(confusion.matrixNB))/sum(confusion.matrixNB)

accNB #accuracy is 0.7218843, lower than Decision Tree

predNB <- predict(NB, testDT.X, type='raw')
score <- predNB[,2]
predictionNB <- prediction(score,testDT.Y)
rocNB <- performance(predictionNB, measure = "tpr", x.measure = "fpr")
plot(rocNB, add=TRUE)


aucNB = performance(predictionNB, measure = "auc")
aucNB@y.values[[1]] # 0.7862417

# logistic regression classifier
stand = diab
stand[,c(5,9,10,11,14,15,16)] = scale(stand[,c(5,9,10,11,14,15,16)])


LR <- glm(Diabetes_binary~., data = stand[-test,], family = binomial(link="logit"))
summary(LR)

#goodness of fit of Logistic regression
testLR.X = stand[test,][,-1] 
testLR.Y = stand[test,][,1]

predLR <- predict(LR, newdata = testLR.X, type="response")
predLR<- ifelse(predLR >0.5, 1, 0)
confusion.matrixLR = table(predLR, testLR.Y)

accLR = sum(diag(confusion.matrixLR))/sum(confusion.matrixLR)

accLR # accuracy is 0.7435281, higher than the others


predLR <- predict(LR, testLR.X, type = "response")
predictionLR <- prediction(predLR, testLR.Y)
rocLR <- performance(predictionLR, measure = "tpr", x.measure="fpr")
plot(rocLR, col = "green", add=TRUE)

legend("bottomright", c("Decision Tree (Class)", "Decision Tree (Prob)", 
"Naive Bayes", "Logistic Regression"), col=c("red","blue","black","green"),
 lty=1)

aucLR <- performance(predictionLR, measure = "auc")
aucLR@y.values[[1]] # 0.8219271

# using N-folds cross validation for logistic regression
num_folds <- 5
folds_j <- sample(rep(1:num_folds, length.out = n))
table(folds_j)
acc <- numeric(num_folds)
X = stand[,-1]
Y = stand[,1]
for (j in 1:num_folds){
test_j <- which(folds_j == j)
LR <- glm(Diabetes_binary~., data = stand[-test_j,], family = binomial(link="logit"))
pred<- predict(LR, newdata = X[test_j,] , type = "response")
pred<- ifelse(pred >0.5, 1, 0)

acc[j] = mean(Y[test_j] == pred)}
mean(acc) # 0.748246


