# Author: Haozhen Wu
# Date: 8/28/2016
# https://www.kaggle.com/c/bosch-production-line-performance/forums/t/23146/xgboost-simple-starter-auc-0-712

# XGBosst simple starter
# use 300+ important features selected by H2O GBM
rm(list=ls(all=TRUE))
gc()

package_list <- c("data.table","Matrix","xgboost")
for(pkg in package_list){
  if (!require(pkg,character.only = T)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg,character.only = T)
  }
  library(pkg,character.only = T)
}
setwd("C:/Users/John Doe/Google Drive/Kaggle/Bosch")

training_number <- 200000
label <- fread("Data/train_numeric.csv",select = c("Response"),nrow = training_number)
importance = fread("Data/imp_matrix_H2O_GBM_allFeatures.csv")

#----- numeric features
dt_train_numeric = fread("Data/train_numeric.csv",select = importance[,variable],nrow = training_number)
# assign 0 to numeric missing

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
dtrain <- xgb.DMatrix(data = train,label = label[,Response])

set.seed(2016)
param <- list(objective = "binary:logistic"
              ,eta = 0.1
              ,max.depth = 5
              ,min_child_weight = 10
              ,max_delta_step = 5
              ,subsample = 0.7
              ,colsample_bytree = 0.7
              # ,scale_pos_weight = table(label)[1]/table(label)[2]
)
round <- 100
xgbtrain <- xgb.train(data = dtrain, params=param, nrounds = round)
# 3 fold cross vaildation achieved 0.712 based on this setting
rm(dtrain,train,train_combine)
gc()

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



### Separate submission calculatin into X-1 pieces to save RAM
x <- 10
num_row <- length(test[,1])
inc_low <- as.integer(seq(1, num_row, length.out = x))
inc_high <- inc_low - 1
inc_high[length(inc_high)] <- inc_high[length(inc_high)] + 1
inc_low <- inc_low[-length(inc_low)]
pred <- numeric() 
for(i in seq_along(inc_low)){
  
  pred_temp <- as.data.frame(predict(xgbtrain,test[inc_low[i]:inc_high[i+1],]))
  pred <- rbind(pred,pred_temp)

}
colnames(pred) <- c("pred")






Id <- fread("Data/test_numeric.csv",select = "Id")

result <- data.table(Id = Id[,Id],Response = pred)
colnames(result) <- c("Id","pred")

result <- result[order(pred,decreasing = T)]
# Naively choose top 5500 instance as label 1, you can change the threshold by yourself.
result[1:5500,Response:=1]
result[5501:nrow(result),Response:=0]
result$pred <- NULL
write.csv(result,paste0("./xgb_200k_Sample.csv"),row.names=F)
