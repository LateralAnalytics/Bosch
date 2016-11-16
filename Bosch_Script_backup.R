# Author: Haozhen Wu
# Date: 8/28/2016
# https://www.kaggle.com/c/bosch-production-line-performance/forums/t/23146/xgboost-simple-starter-auc-0-712

# remove files
rm(list=ls(all=TRUE))
gc()


# Load packages
package_list <- c("data.table","Matrix","xgboost","ggplot2","dplyr")
for(pkg in package_list){
  if (!require(pkg,character.only = T)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg,character.only = T)
  }
  library(pkg,character.only = T)
}

# Set working directory
setwd("C:/Users/John Doe/Google Drive/Kaggle/Bosch")


# Load Feature data
training_number <- 200000
label <- fread("Data/train_numeric.csv",select = c("Response"),nrow = training_number)
importance = fread("Data/imp_matrix_H2O_GBM_allFeatures.csv")

#----- numeric features
dt_train_numeric = fread("Data/train_numeric.csv",select = importance[,variable],nrow = training_number)
# assign 0 to numeric missing

# Change all NA values to 0
fix_missing_num <- function(x) {
  x[x==""|is.na(x)]= 0
  x=as.numeric(x)
  x
}
dt_train_numeric[] <- lapply(dt_train_numeric, fix_missing_num)


#----- categorical features
dt_train_categorical <- fread("Data/train_categorical.csv",
                              colClasses = "character",
                              select = importance[,variable],
                              nrow = training_number)


#replace empty value by "missing"
fix_missing_char <- function(x) {
  x[which(x==""|is.na(x))]= "missing"
  x=as.factor(x)
  x
}
dt_train_categorical[] <- lapply(dt_train_categorical, fix_missing_char)


#------ date features
dt_train_date <- fread("Data/train_date.csv",select = importance[,variable],nrow = training_number)
dt_train_date[] <- lapply(dt_train_date, fix_missing_num)



#----- combine and make xgb data
train_combine <- cbind(dt_train_numeric,dt_train_categorical,dt_train_date)
rm(dt_train_numeric,dt_train_categorical,dt_train_date)
gc()
train <- sparse.model.matrix(~.-1,data=train_combine)


----------------------------------------------------------------
  # https://datascience.stackexchange.com/questions/9364/hypertuning-xgboost-parameters/9368
  
  searchGridSubCol <- expand.grid(subsample = c(0.75, 1.5, 2.0), 
                                  colsample_bytree = c(0.1, 0.75, 1.5, 2.0),
                                  gamma = c(0.05, 0.25, 1.25),
                                  max.depth = c(1, 10, 15))



eval_mcc <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  DT <- data.table(y_true = labels, y_prob = preds, key="y_prob")
  
  nump <- as.numeric(sum(DT$y_true))
  numn <- as.numeric(length(DT$y_true)- nump)
  
  DT[, tn_v:= as.numeric(cumsum(y_true == 0))]
  DT[, fp_v:= numn - tn_v]
  DT[, fn_v:= as.numeric(cumsum(y_true == 1))]
  DT[, tp_v:= nump+numn-tn_v-fp_v-fn_v]
  DT[, mcc_v:= (tp_v * tn_v - fp_v * fn_v) / sqrt((tp_v + fp_v) * (tp_v + fn_v) * (tn_v + fp_v) * (tn_v + fn_v))]
  DT[, mcc_v:= ifelse(!is.finite(mcc_v), 0, mcc_v)]
  
  best_mcc <- max(DT[['mcc_v']])
  best_proba <- DT[['y_prob']][which.max(DT[['mcc_v']])]
  
  return(list(metric = "MCC", value = best_proba))
}


ntrees <- 50
gc()
#Build a xgb.DMatrix object
DMMatrixTrain <- xgb.DMatrix(data = train,label = label[,Response])
gc()

rmseErrorsHyperparameters <- apply(searchGridSubCol, 1, function(parameterList){
  
  #Extract Parameters to test
  currentSubsampleRate <- parameterList[["subsample"]]
  currentColsampleRate <- parameterList[["colsample_bytree"]]
  currentGamma <- parameterList[["gamma"]]
  currentMax.depth <- parameterList[["max.depth"]]
  
  
  xgboostModelCV <- xgb.cv(data =  DMMatrixTrain, nrounds = ntrees, nfold = 5, showsd = TRUE, 
                           verbose = TRUE, eval_metric = eval_mcc, maximize = TRUE, prediction = FALSE,
                           "objective" = "binary:logistic", "max.depth" = currentMax.depth, "eta" = 2/ntrees,                               
                           "subsample" = currentSubsampleRate, "colsample_bytree" = currentColsampleRate,
                           gamma = currentGamma, stratified = TRUE)
  
  gc()
  
  xvalidationScores <- as.data.frame(xgboostModelCV)
  
  return(c(currentSubsampleRate, currentColsampleRate, currentGamma, currentMax.depth, xvalidationScores))
  
})


### Unpack hyperparameter search data structure
num_rows <- length(searchGridSubCol[,1])*ntrees
plug_zeros <- data.frame(numeric(num_rows))
df_length <- length(searchGridSubCol)+4

hyperParameterResults <- plug_zeros
for (i in 1:(df_length)){
  hyperParameterResults <- cbind(hyperParameterResults,plug_zeros)
}



name_1 <- colnames(searchGridSubCol[1])
name_2 <- colnames(searchGridSubCol[2])
name_3 <- colnames(searchGridSubCol[3])
name_4 <- colnames(searchGridSubCol[4])
colnames(hyperParameterResults) <- c(name_1, name_2, name_3, name_4, "ntrees", 
                                     "train.metric.mean", "train.metric.std", "test.metric.mean", "test.metric.std")


k=0
for (i in 1:length(searchGridSubCol[,1])){
  for (j in 1:(ntrees)){
    paste(rmseErrorsHyperparameters[[1]][5][[1]], sep = " ")[j]
    k <- k+1
    hyperParameterResults[k,1] <- rmseErrorsHyperparameters[[i]][1]
    hyperParameterResults[k,2] <- rmseErrorsHyperparameters[[i]][2]
    hyperParameterResults[k,3] <- rmseErrorsHyperparameters[[i]][3]
    hyperParameterResults[k,4] <- rmseErrorsHyperparameters[[i]][4]
    hyperParameterResults[k,5] <- j
    hyperParameterResults[k,6] <- paste(rmseErrorsHyperparameters[[i]][5][[1]], sep = " ")[j]
    hyperParameterResults[k,7] <- paste(rmseErrorsHyperparameters[[i]][6][[1]], sep = " ")[j]
    hyperParameterResults[k,8] <- paste(rmseErrorsHyperparameters[[i]][7][[1]], sep = " ")[j]
    hyperParameterResults[k,9] <- paste(rmseErrorsHyperparameters[[i]][8][[1]], sep = " ")[j]
    
  }
  
  
}

### Analyze results of hyperparameter search

result <- hyperParameterResults %>% group_by(subsample, colsample_bytree, gamma, max.depth) %>%
  filter(test.metric.mean == min(test.metric.mean)) %>%
  arrange(subsample, colsample_bytree, gamma, max.depth)


p <- ggplot(result, aes(subsample, test.metric.mean))
p <- p + geom_point(aes(colour = factor(colsample_bytree)))
p <- p + facet_grid(gamma ~ max.depth)
p


result_2 <- hyperParameterResults %>% 
  filter(max.depth == 15) %>%
  filter(gamma == 0.05) %>%
  filter(colsample_bytree == c(1.5)) %>%
  arrange(subsample, colsample_bytree, gamma, max.depth)

q <- ggplot(result_2, aes(ntrees, test.metric.mean))
q <- q + geom_point(aes(colour = factor(colsample_bytree)))
q <- q + facet_grid(subsample ~ colsample_bytree)
q


### Generate final model for prediction

ntrees = 100



xgboostModelCV_final <- xgb.train(data =  DMMatrixTrain, nrounds = ntrees, nfold = 5, showsd = TRUE, 
                                  verbose = TRUE, eval_metric = eval_mcc, maximize = TRUE, prediction = FALSE,
                                  "objective" = "binary:logistic", "max.depth" = 15, "eta" = 2/ntrees,                               
                                  "subsample" = 2.0, "colsample_bytree" = 1.5,
                                  gamma = 0.05, stratified = TRUE)


### Get MCC cutoff probability
pred_MCC <- as.data.frame(predict(xgboostModelCV_final,DMMatrixTrain))
colnames(pred_MCC) <- c("y_prob")

best_proba <- eval_mcc_last(pred_MCC,DMMatrixTrain)




----------------------------------------------------------------
  
  
  
  
  
  #--------------------------make prediction
  
  #----- numeric features 
  dt_test_numeric <- fread("Data/test_numeric.csv",select = importance[,variable])

dt_test_numeric[] <- lapply(dt_test_numeric, fix_missing_num)

#----- categorical features
dt_test_categorical <- fread("Data/test_categorical.csv",select = importance[,variable])
#replace empty value by "missing"
dt_test_categorical[] <- lapply(dt_test_categorical, fix_missing_char)



#------ date features
dt_test_date <- fread("Data/test_date.csv",select = importance[,variable])
dt_test_date[] <- lapply(dt_test_date, fix_missing_num)


#------ combine
test_combine <- cbind(dt_test_numeric,dt_test_categorical,dt_test_date)
rm(dt_test_numeric,dt_test_categorical,dt_test_date)
gc()
test <- sparse.model.matrix(~.-1,data=test_combine)
gc()



### Separate submission calculation into X-1 pieces to save RAM
x <- 10
num_row <- length(test[,1])
inc_low <- as.integer(seq(1, num_row, length.out = x))
inc_high <- inc_low - 1
inc_high[length(inc_high)] <- inc_high[length(inc_high)] + 1
inc_low <- inc_low[-length(inc_low)]
pred <- numeric() 
for(i in seq_along(inc_low)){
  
  pred_temp <- as.data.frame(predict(xgboostModelCV_final,test[inc_low[i]:inc_high[i+1],]))
  pred <- rbind(pred,pred_temp)
  
}
colnames(pred) <- c("pred")






Id <- fread("Data/test_numeric.csv",select = "Id")

result <- data.table(Id = Id[,Id],Response = pred)
colnames(result) <- c("Id","pred")
result <- result[order(pred,decreasing = F)]



result$Response[result$pred>=0.06462096]= 1
result$Response[result$pred<0.06462096]= 0






result$pred <- NULL
write.csv(result,paste0("./xgb_200k_Sample.csv"),row.names=F)







