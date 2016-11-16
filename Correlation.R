
#################################################################################################
### This is a way to remove features which are perfectly or highly correllated with eachother ###
### This may be necessary if data files are too big to proces such as thos found at           ###
### https://www.kaggle.com/c/bosch-production-line-performance                                ###
#################################################################################################


#################################################################################################
### To Do: Turn this into a function                                                          ###
#################################################################################################



# remove files
rm(list=ls(all=TRUE))
gc()


#imports for data wrangling
library(data.table)
library(dplyr)
library(caret)
library(ggplot2)
library(polycor)

#get the data - nrows set to 10000 to keep runtime manageable.
#one expansion option would be to select a time frame to visualized
dtNum <- fread("../Data/train_numeric.csv",nrows = 10000)
dtDate <- fread("../Data/train_date.csv", nrows = 10000)

# Change all NA values to 0
fix_missing_num <- function(x) {
  x[x==""|is.na(x)]= 0
  x=as.numeric(x)
  x
}
dtNum[] <- lapply(dtNum, fix_missing_num)
dtNum$Id <- NULL

#select only the upper triangle to avoid duplicates
dtNumCorr <- cor(dtNum,dtNum)
dtNumCorr[lower.tri(dtNumCorr)] <- NA
dtNumCorr <- as.data.frame(dtNumCorr)


#Melt the data and remove NA's
row <- row.names(dtNumCorr)
dtNumCorr$row <- row
dtNumCorrMelt <- melt(dtNumCorr, id.vas = row, na.rm=TRUE)

#Remove self correlations
dtNumCorrMelt %>%
  filter((row != variable)) %>%
  arrange(-value, row, variable)-> dtNumCorrMelt

#Select the threshold corellation value to focus on
dtNumCorrMelt %>%
  filter(value > .95) -> dtNumCorrMeltOne

dtNumCorrMeltOne$row <-as.factor(dtNumCorrMeltOne$row)
dtNumCorrMeltOne$variable <-as.factor(dtNumCorrMeltOne$variable)

dtNumCorrMeltOne %>%
  arrange(row, variable) -> dtNumCorrMeltOne

dtNumCorrMeltOne[,1] <- as.character(dtNumCorrMeltOne[,1])
dtNumCorrMeltOne[,2] <- as.character(dtNumCorrMeltOne[,2])

#Assign group numbers
dtNumCorrMeltOne$group <- 0
i_len <- length(dtNumCorrMeltOne[,1])
j_len <- length(dtNumCorrMeltOne[,1])
for (i in 1:i_len){
  if (dtNumCorrMeltOne[i,4]==0) {
    dtNumCorrMeltOne[i,4]=i
    }
    for (j in 1:j_len){
      if (j>i) {
        a <- as.matrix(dtNumCorrMeltOne[dtNumCorrMeltOne$group==i,1:2 ])
        b <- c(dtNumCorrMeltOne[j,1],dtNumCorrMeltOne[j,2])
        if (length(intersect(a,b))>0) {  
            
          if (dtNumCorrMeltOne[j,4]==0) {
            dtNumCorrMeltOne[j,4] <- i
          }
        }
      }
    }
}

#Put the column names together in one column
temp_1 <- dtNumCorrMeltOne[,c(1,4)]
colnames(temp_1)<- c("field","group")
temp_2 <- dtNumCorrMeltOne[,c(2,4)]
colnames(temp_2)<- c("field","group")
dtNumCorrMeltOneLong <- rbind(temp_1,temp_2)

#Sort and remove duplicates
dtNumCorrMeltOneLong %>%
  arrange(group, field) -> dtNumCorrMeltOneLong

dtNumCorrMeltOneLong <- unique(dtNumCorrMeltOneLong)

#Find the members of a group (column names) other than the first one.
dtNumCorrMeltOneLongMin <- group_by(dtNumCorrMeltOneLong, group)
dtNumCorrMeltOneLongMin <- filter(dtNumCorrMeltOneLongMin, row_number(group) == 1)

#Below are the column names which should not be evaluated
remove <- setdiff(dtNumCorrMeltOneLong,dtNumCorrMeltOneLongMin)
