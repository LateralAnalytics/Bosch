
  
  
  eval_mcc <- function(preds, dtrain) {
    labels <- getinfo(dtrain, "label")
    
    DT <- data.table(y_true = labels, y_prob = preds, key="y_prob")
    #print(colnames(DT))
    #print(head(DT))
    
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
  
  
  
  
  eval_mcc_last <- function(DT) {

    colnames(DT) <- c("y_true","y_prob")

    
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
  
  
  
  




