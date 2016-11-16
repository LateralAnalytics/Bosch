### https://www.kaggle.com/cartographic/bosch-production-line-performance/bish-bash-xgboost/discussion
### Just get started and modify from here
library(data.table)
library(Matrix)
library(caret)
library(xgboost)

### Set working directory
setwd("C:/Users/John Doe/Google Drive/Kaggle/Bosch")

### Clean up environment
rm(list=ls(all=TRUE))
gc()

dt <- fread("Data/train_numeric.csv",
            drop = "Id",
            nrows = 200000,
            showProgress = F)

Y  <- dt$Response
dt[ , Response := NULL]

### Once the numeric data has been read, all the values are offset by +2 to allow the nas to be set to zero.

for(col in names(dt)) set(dt, j = col, value = dt[[col]] + 2)
for(col in names(dt)) set(dt, which(is.na(dt[[col]])), col, 0)

X <- Matrix(as.matrix(dt), sparse = T)
rm(dt)


### How do I make this stratified?
folds <- createFolds(as.factor(Y), k = 6)
valid <- folds$Fold1
model <- c(1:length(Y))[-valid]

param <- list(objective = "binary:logistic", 
              eval_metric = "auc",
              eta = 0.01,
              base_score = 0.005,
              col_sample = 0.5) 

dmodel <- xgb.DMatrix(X[model,], label = Y[model])
dvalid <- xgb.DMatrix(X[valid,], label = Y[valid])

m1 <- xgb.train(data = dmodel, param, nrounds = 20,
                watchlist = list(mod = dmodel, val = dvalid))

imp <- xgb.importance(model = m1, feature_names = colnames(X))

cols <- imp$Feature
length(cols)

head(cols, 10)

rm(list = setdiff(ls(), "cols"))

### Filter to the good stuff
dt <- fread("Data/train_numeric.csv",
            select = c(cols, "Response"),
            showProgress = F)

Y  <- dt$Response
dt[ , Response := NULL]

for(col in names(dt)) set(dt, j = col, value = dt[[col]] + 2)
for(col in names(dt)) set(dt, which(is.na(dt[[col]])), col, 0)

X <- Matrix(as.matrix(dt), sparse = T)
rm(dt)



###Bathe in xgboost

set.seed(7579)
folds <- createFolds(as.factor(Y), k = 6)
valid <- folds$Fold3
model <- c(1:length(Y))[-valid]

param <- list(objective = "binary:logistic",
              eval_metric = "auc",
              eta = 0.01,
              max_depth = 7,
              colsample_bytree = 0.5,
              base_score = 0.005) 

dmodel <- xgb.DMatrix(X[model,], label = Y[model])
dvalid <- xgb.DMatrix(X[valid,], label = Y[valid])

m1 <- xgb.train(data = dmodel, param, nrounds = 20,
                watchlist = list(mod = dmodel, val = dvalid))

pred <- predict(m1, dvalid)

summary(pred)

imp <- xgb.importance(model = m1, feature_names = colnames(X))

head(imp, 30)

### Gauge Matthews Coefficient

mc <- function(actual, predicted) {
  
  tp <- as.numeric(sum(actual == 1 & predicted == 1))
  tn <- as.numeric(sum(actual == 0 & predicted == 0))
  fp <- as.numeric(sum(actual == 0 & predicted == 1))
  fn <- as.numeric(sum(actual == 1 & predicted == 0))
  
  numer <- (tp * tn) - (fp * fn)
  denom <- ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ^ 0.5
  
  numer / denom
}

matt <- data.table(thresh = seq(0.990, 0.999, by = 0.001))

matt$scores <- sapply(matt$thresh, FUN =
                        function(x) mc(Y[valid], (pred > quantile(pred, x)) * 1))

print(matt)

best <- matt$thresh[which(matt$scores == max(matt$scores))]

### Test cycle

dt  <- fread("Data/test_numeric.csv",
             select = c("Id", cols),
             showProgress = T)

Id  <- dt$Id
dt[ , Id := NULL]

for(col in names(dt)) set(dt, j = col, value = dt[[col]] + 2)
for(col in names(dt)) set(dt, which(is.na(dt[[col]])), col, 0)

X <- Matrix(as.matrix(dt), sparse = T)
rm(dt)

### Submission emission

dtest <- xgb.DMatrix(X)
pred  <- predict(m1, dtest)

summary(pred)

sub   <- data.table(Id = Id,
                    Response = (pred > quantile(pred, best)) * 1)

write.csv(sub, "sub.csv", row.names = F)

gc()




