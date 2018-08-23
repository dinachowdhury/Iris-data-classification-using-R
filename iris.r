#1.Loading the dataset.
#2.Summarizing the dataset.
#3.Visualizing the dataset.
#4.Evaluating some algorithms.
#5.Making some predictions.

install.packages("caret")
install.packages("lattice")
install.packages("ggplot2")
library(ggplot2)
library(lattice)
library(caret)


getwd()
setwd("C:\\AFROSE\\IRIS\\Data\\")

#1. Load Data

data_url<-"https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# for file download destination file should be a name not a directory

data_folder<-file.path("C:\\AFROSE\\IRIS\\Data\\iris.csv")
download.file(data_url,data_folder)
main_data<-read.csv("C:\\AFROSE\\IRIS\\Data\\iris.csv", sep=",", header= F,
                    col.names = c("Sepal length","Sepal width","Petal length","Petal width","Class"))


# Create training dataset which will carry 80% of data
train_data <- createDataPartition(main_data$Class,p=0.8,list = F)
test_data <- main_data[-train_data,]
main_data <-main_data[train_data,]

# 2.Summarize dataset
dim(main_data)
sapply(main_data,class)
head(main_data)
levels(main_data$Class)

class_percentage <-prop.table(table(main_data$Class))*100
cbind(freq=table(main_data$Class), percentage=class_percentage)

summary(main_data)

# 3. Dataset Visualizatioh
# Univariate plots to better understand each attribute.

x<-main_data[,1:4] #split dataset input & output
y<-main_data[,5]

# boxplot for each attribute on one image
par(mfrow=c(1,4))
for(i in 1:4) {
  boxplot(x[,i], main=names(main_data)[i])
}
#Barplot 
plot(y)

# Multivariate plots to better understand the relationships between attributes.
# Scatterplot
install.packages("ellipse")
library(ellipse)
featurePlot(x=x, y=y, plot="ellipse")
featurePlot(x=x, y=y, plot="box")
# density plots for each attribute by class value
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)

# 4.Evaluate Some Algorithms
# Run algorithms using 10-fold cross validation
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

#Linear Discriminant Analysis (LDA),Classification and Regression Trees (CART).,k-Nearest Neighbors (kNN).,Support Vector Machines (SVM) with a linear kernel.,Random Forest (RF)
#Linear algorithm LDA
install.packages("e1071")
library(e1071)

set.seed(10)
fit.lda<- train(Class~.,data=main_data,method ="lda", metric=metric,trControl=control)
set.seed(10)
fit.cart<- train(Class~.,data=main_data,method ="rpart", metric=metric,trControl=control)
set.seed(10)
fit.knn<- train(Class~.,data=main_data,method ="knn", metric=metric,trControl=control)
set.seed(10)
fit.svm<- train(Class~.,data=main_data,method ="svmRadial", metric=metric,trControl=control)
set.seed(10)
fit.rf<- train(Class~.,data=main_data,method ="rf", metric=metric,trControl=control)



# summarize accuracy of models
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)

#Compare accuracy model
dotplot(results)

#Summarize best modek
print(fit.lda)

#prediction on train data
predictions<-predict(fit.lda,test_data)
confusionMatrix(predictions,test_data$Class)
