rm(list=ls())
library("shape",lib.loc="")# pass path to the package
library("methods",lib.loc="")# pass path to the package
library("foreach",lib.loc="")# pass path to the package
library("Matrix",lib.loc="")# pass path to the package
library("glmnet",lib.loc="")# pass path to the package
library("sparsenet",lib.loc="")# pass path to the package
library("AUC",lib.loc="")# pass path to the package
library("pROC",lib.loc="")# pass path to the package
library("Metrics",lib.loc="")# pass path to the package

normalize = function(x){
# we use z-score normalization to standardize
y = (x-mean(x))/sd(x)
 return(y)
}

data <- read.csv("",header=TRUE,sep=",")# pass path to the package

data[which(data[,ncol(data)]==0),ncol(data)] <- -1# here we change label 0 to -1
x <- as.matrix(data[,1:ncol(data)-1])#here feature matrix is extracted
centered_x <- matrix(0,nrow=nrow(x),ncol=ncol(x))
# here each column of the feature matrix is standardized
for(i in 1:ncol(x)){
centered_x[,i] <- normalize(x[,i])
}

y <- as.vector(data[,ncol(data)])#here class lable is extracted
centered_y <- as.matrix(scale(y,center=TRUE))# here class is centered

n <- nrow(data)
folds <- split(sample(1:n),rep(1:5,length=n))#here we generate 5 fold train and test datasets
auc_all <- numeric()# here we create empty vector to store AUC for all the folds

# in the below loop we iterate through folds and fit the model
for(i in 1:5){
fit <- sparsenet(x[-folds[[i]],],y[-folds[[i]]])# here we fit the model on train data
gamma <- as.vector(fit$gamma)# here we extract the gamma sequence used in the model
lambda <- as.vector(fit$lambda)#here we extract lambda sequence used in the model
yhat <- predict(fit,x[folds[[i]],],type="response")# here we make prediction on the test data

mse_optimum <- numeric()# here we store the indices of gamma and lambda combination with minimum MSE
# here we iterate through all the combinations of gamma and lambda to find the minimum MSE
for(j in 1:length(gamma)){
yhat_gamma <- yhat[[j]]
mse_all <- numeric()
for(k in 1:length(lambda)){
mse_all <- c(mse_all,mse(y[folds[[i]]],yhat_gamma[,k]))
}
if(j==1){
mse_min <- min(mse_all)
mse_min_indx <- which(mse_all==min(mse_all))
mse_optimum <- c(mse_optimum,c(j,mse_min_indx))
}
if(mse_min < min(mse_all)){
mse_min <- min(mse_all)
mse_min_indx <- which(mse_all==min(mse_all))
mse_optimum <- mse_optimum[-c(1:length(mse_optimum))]
mse_optimum <- c(mse_optimum,c(j,mse_min_indx))
}
}

fold_auc <- roc(y[folds[[i]]],as.vector(sign(yhat[[mse_optimum[1]]][,mse_optimum[2]])))# here we calculate the AUC for the prediction using the minimum MSE combination 
auc_all <- c(auc_all,fold_auc$auc)
}

sd(auc_all)# here we calculate the standard deviation of AUC for all folds
mean(auc_all)# here we calculate the mean AUC for all folds
