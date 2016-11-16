### https://www.kaggle.com/c/bosch-production-line-performance/forums/t/23146/xgboost-simple-starter-auc-0-712
library(h2o) 
library(data.table) 
h2o.init(ip='localhost',port=54321,nthreads=-1,max_mem_size='8g')

setwd("C:/Users/John Doe/Google Drive/Kaggle/Bosch")

#load to H2O
train_numeric <- h2o.importFile(path = "Data/train_numeric.csv",destination_frame = "train_numeric.hex") 
train_categorical <- h2o.importFile(path = "Data/train_categorical.csv",destination_frame = "train_categorical.hex") 
train_date <- h2o.importFile(path = "Data/train_date.csv",destination_frame = "train_date.hex")

train_numeric[,"Response"] <- as.factor(train_numeric[,"Response"]) 
train <- h2o.cbind(train_numeric,train_categorical,train_date) 
train_numeric_name <- colnames(train_numeric)[-c(1,dim(train_numeric)[2])] 
train_categorical_name <- colnames(train_categorical)[-1] 
train_date_name <- colnames(train_date)[-1] 
train_name <- c(train_numeric_name,train_categorical_name,train_date_name)

#Train GBM model
gbm1 <- h2o.gbm(train_name,"Response",train, 
                ignore_const_cols = TRUE, 
                ntrees = 100, 
                max_depth = 5, 
                min_rows = 20, 
                learn_rate = 0.1, 
                sample_rate = 0.8, 
                col_sample_rate = 0.8, 
                balance_classes = FALSE, 
                seed = 2016, 
                nfolds = 3, 
                fold_assignment = "Stratified", 
                keep_cross_validation_predictions = TRUE, 
                keep_cross_validation_fold_assignment = TRUE, 
                score_tree_interval = 20, 
                stopping_rounds = 10, 
                stopping_metric = "AUC", 
                stopping_tolerance = 0.01 
)

importance <- h2o.varimp(gbm1) 
importance <- as.data.table(importance) 
importance <- importance[scaled_importance>0]