# Please note: Saving and Removing objects is optional (appears as a non-executable comment with the sign "#" before the code)
# throughout the code below. If you would like to save/remove certain objects, you should first remove the comment sign "#" 
# at the beginning of the respective lines of code that save/remove  these objects below.

####################
# Loading Libraries
####################

# Loading required libraries for the project.

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(rpart)
library(randomForest)
library(knitr)



################################
# Setting the working directory
################################


# I created two sub-directories of the working directory - "rdas" and "data". 
# The zip file was saved into the "data" sub-directory. All other files created were saved into the "rdas" 
# sub-directory. This is optional. 

# You need to make sure though that you are in the directory where you place the zip file from the url below 
# in order for the code to work.

getwd()

setwd("./data")

# getwd()



################
# Download file.
################


# The dataset can be downloaded as a zip file to your computer from:

# https://github.com/EW-10/capstone-cyo-work/blob/master/data/census.zip.

# Make sure you are in the directory where the zip file is located.

# getwd()

# list.files() # Confirm that the zip file is located in your current directory.

# file <- "census.zip"

# adult_csv <- unzip(file, "adult.csv")

# adult <- read_csv(adult_csv)

# Alternatively, the file can be downloaded directly from the url below and processed, without the need to save the file 
# on your computer yourself.

url <- "https://github.com/EW-10/capstone-cyo-work/raw/master/data/census.zip"

dl <- tempfile()

download.file(url, dl)

adult <- read_csv(unzip(dl, "adult.csv"))

rm(dl)

##################
# Observe dataset.
##################

# head(adult)

# str(adult)

# dim(adult)

sum(is.na(adult))

# There are no NAs in the "adult" dataset.

sum(adult == "?")

# Total missing values ("?") in the "adult" object: 4262. At this stage I leave the missing values in the dataset and treat them
# as a separate category for the categorical variables where they appear.

sum(adult == "0")

# There are 60891 values equal to "0" in the dataset. I will now observe which variables contain zeros.

# dim(adult)

zeros <- sapply(adult[,1:15], function(x){
  sum(x == 0)
})

zeros

# The zeros are located only in the columns of "capital.gain" and "capital.loss". Large parts of the population do not
# have capital.gain OR capital.loss. I will show below how many observations have capital.gain or capital.loss that are equal to 
# zero.

table(adult$capital.gain == 0)

table(adult$capital.loss == 0)

# Most of the observations in the dataset have capital.gain or capital.loss that is equal to zero.

sum(adult$income == "?")

# There are no labels (income column) that are unknown ("?").

missing_values <- sapply(adult[,1:15], function(x){
  sum(x == "?")
})

missing_values

# The missing values ("?") are in the columns "workclass" (1836 missing values), "occupation" (1843 missing values), 
# and native.country (583 missing values). At this stage, I chose not to remove observations that contained missing values
# In the analysis below, a missing value (which was represented in the dataset by a "?") was considered
# a category (level) on its own for the categorical variables - "workclass", "occupation", and "native.country". All missing values
# were included in the analysis. Both the train and the test sets below included missing values ("?").

miss_values_workclass_occupation <- with(adult, sum(workclass == "?" & occupation == "?"))

miss_values_workclass_occupation

# All of the missing values in the "workclass" column were also missing for the same observations in the "occupation" column. 
# The column "occupation" had additional 7 missing values that were not missing from the "workclass" column.

adult$workclass[with(adult, workclass != "?" & occupation == "?")]

# All of the 7 observations in which the column "occupation" had a missing value ("?"), but the column "workclass" did not
# have a missing value, had "Never-worked" at the "workclass" column.

sum(adult$workclass == "Never-worked")

# All of the observations that had "Never-worked" at the "workclass" column had a missing value ("?") at the "occupation" column. 
# There were only 7 observations that had "Never-worked" at the "workclass" column. I elected to leave the missing values in at
# this stage. Future work could possibly remove observations with missing values from the dataset before partitioning it to train
# and test sets. I elected to treat the missing values in the columns "workclass", "occupation" and "native.country" as a category
# by itself for these three categorical variables. My main reasons were that missing values were present only at these three columns, and as I observed,
# and that some of the missing values in the "occupation" column had "Never-worked" in the "workclass" column - and were therefore
# meaningful. I did not consider it as a problem having one of the categories of the "native.country" variable as "?" - 
# which means unknown native country. Thus, I left all missing values ("?") in the dataset for my analysis. Future work could
# remove observations with missing values and run the analysis without them, specifically looking to see if this step improves
# model performance (the accuracy as measured on the test set).


#################
# Data wrangling
#################

s <- sapply(adult[,1:15], class)

s

census <- adult %>% mutate_at(which(s == "character"), as.factor)

levels(census$income)

n_distinct(census$income)

# The only two values in the labels column (income) are "<=50k" and ">50k".

# A variable "b_income" is created that contains the "income" variable as a binary variable, with "0" for "<=50k" and "1" for
# ">50k".

census <- census %>% mutate(b_income = as.numeric(income) -1)

# head(census$b_income)

census %>% select(income, b_income) %>% head(10)

# Building a matrix "x" with all the predictors and a vector "y" that contains the binary outcome. The binary column b_income
# has "0" for "income <= 50K", and "1" for "income > 50K".

x <- census[,1:14]

y <- census[16]

# Building a dataframe "df" containing the matrix of predictors "x", and the vector of outcomes "y".

df <- data.frame(c(x,y))

# I will now remove the column "fnlwgt" and run the analysis without it. "fnlwgt" was removed from reasons explained elsewhere.

df <- df[,-3]



#################################
# Creation of train and test set. 
#################################

# 80% of the original dataset will be included in the train set. 20% of the original dataset will be included in the test set.

# head(y, 10)

set.seed(1, sample.kind = "Rounding")

test_index <- createDataPartition(df$b_income, times = 1, p = 0.2, list = FALSE)

train_set <- slice(df, -test_index)

test_set <- slice(df, test_index)

# I will now make sure that there are no levels in any factor at the test_set that are not included in the 
# train_set.

temp <- test_set

test_set <- test_set %>%
  semi_join(train_set, by = "workclass") %>%
  semi_join(train_set, by = "education") %>%
  semi_join(train_set, by = "marital.status") %>%
  semi_join(train_set, by = "occupation") %>%
  semi_join(train_set, by = "relationship") %>%
  semi_join(train_set, by = "race") %>%
  semi_join(train_set, by = "sex") %>%
  semi_join(train_set, by = "native.country")
  
Removed <- temp %>%
  anti_join(test_set)

# nrow(Removed)

# I will add the line removed from the test_set back into the train_set.

train_set <- train_set %>%
  rbind(Removed)

rm(temp, Removed)

# nrow(train_set)

# nrow(test_set)

# s1 <- sum(train_set == "?")

# s2 <- sum(test_set == "?")

# tibble("Missing values train set" = s1, "Missing values test set" = s2)

# As stated above, I have left the missing values in the dataset for my analysis and have not removed them.



################################
# Model (1) - Linear Regression
################################

# The first model is linear regression.

fit_lm <- lm(b_income ~ ., data = train_set)

p_lm <- predict(fit_lm, type = "response", newdata = test_set)

y_hat_lm <- ifelse(p_lm > 0.5, 1, 0)

# Confusion matrix

cm_lm <- confusionMatrix(as.factor(y_hat_lm), as.factor(test_set$b_income))

# cm_lm

# The table "table_acc" will store the accuracy results for all of the models used.

table_acc <- tibble(Method = "Linear Regression", Accuracy = cm_lm$overall["Accuracy"])



##################
# Model (2) - LDA
##################

fit_lda <- train(as.factor(b_income) ~ ., data = train_set, method = "lda")

p_lda <- predict(fit_lda, newdata = test_set)

cm_lda <- confusionMatrix(p_lda, as.factor(test_set$b_income))

# cm_lda

# The accuracy of LDA is higher than the one obtained with linear regression.

table_acc <- table_acc %>%
  rbind(tibble(Method = "LDA", Accuracy = cm_lda$overall["Accuracy"]))



##################
# Model (3) - KNN
##################

fit_knn <- knn3(as.factor(train_set$b_income) ~ ., data = train_set, k = 5)

p_knn <- predict(fit_knn, type = "class", newdata = test_set)

cm_knn <- confusionMatrix(p_knn, as.factor(test_set$b_income))

# cm_knn

table_acc <- table_acc %>%
  rbind(tibble(Method = "KNN", Accuracy = cm_knn$overall["Accuracy"]))



#########################################
# Model (4) - RPART (Classification Tree)
#########################################

# Cross-validation to select the best parameter "cp" for the RPART model.

cp.tuning = data.frame(cp = seq(0, 0.1, len = 25))

set.seed(1, sample.kind = "Rounding")

train_rpart <- train(method = "rpart", as.factor(b_income) ~ ., data = train_set, tuneGrid = cp.tuning)

cp <- train_rpart$bestTune %>% pull(cp)

cp

# Plotting the effect of the parameter "cp" on the accuracy of the resulting tree and saving it as "plot_cp.png". Please note:
# This plot is used by the Rmd file. If you would like to knit the Rmd file yourself, you should execute the code below and 
# save this plot to the directory where the Rmd file is located.

# png("plot_cp.png")

# plot(train_rpart)

# dev.off()

# Running the RPART model with the best "cp" parameter selected above.

set.seed(1, sample.kind = "Rounding")

fit_rpart <- rpart(as.factor(b_income) ~ ., data = train_set, cp = cp)

# Plotting and saving the resulting tree. This plot is used by the Rmd file. If you want to knit it yourself, you would need to 
# remove the "#" sign before the code below and execute this code to create and save the plot.

# png("rpart_tree1.png")

plot(fit_rpart, margin = 0.0001)

text(fit_rpart, cex = 0.75)

# dev.off()

# The tree contains the following conditions: Relationship - if unmarried then the income depends on capital.gain. If married, 
# the income depends on education, then on capital.gain, then on occupation. If the education is below a certain level, then
# in addition to capital.gain and occupation, the income depends on age > 32.5, capital.loss > 1846, on hours.per.week > 34.5, 
# again on the level of education, on workclass, then on occupation again, and then again on workclass. Different categories of
# these variables are used at different levels of the tree.

p_rpart <- predict(fit_rpart, test_set, type = "class")

cm_rpart <- confusionMatrix(p_rpart, as.factor(test_set$b_income))

# cm_rpart

table_acc <- table_acc %>%
  rbind(tibble(Method = "RPART", Accuracy = cm_rpart$overall["Accuracy"]))



############################
# Model (5) - RANDOM FOREST
############################

set.seed(1, sample.kind = "Rounding")

fit_rf <- randomForest(as.factor(b_income) ~ ., data = train_set)

p_rf <- predict(fit_rf, newdata = test_set)

imp_n_f <- importance(fit_rf)

# view(imp_n_f)

cm_rf <- confusionMatrix(p_rf, as.factor(test_set$b_income))

# cm_rf

table_acc <- table_acc %>% 
  rbind(tibble(Method = "Random Forest", Accuracy = cm_rf$overall["Accuracy"]))

colnames(table_acc) <- c("Model without 'fnlwgt'", "Accuracy")


table_acc %>% knitr::kable()



###########
# Ensemble
###########

# The following code is used to build three ensembles: (1) With all 5 models; (2) With 4 models, not including LM, and (3) With 3
# models, including KNN, RPART and RANDOM FOREST (not including LM and LDA). The accuracy of the three ensembles built was less
# than the accuracy of RANDOM FOREST. The three ensembles I tried do not appear at the final accuracy results table.

# The first ensemble includes all 5 methods: LM, LDA, KNN, RPART, and RANDOM FOREST.

 table <- as.factor(y_hat_lm) %>%
  cbind(p_lda) %>%
  cbind(p_knn) %>%
  cbind(p_rpart) %>%
  cbind(p_rf)

pred <- sapply(1:nrow(table), function(i){
  ifelse(mean(table[i,] == 1) > 0.5, 0, 1)
})

 ensemble_5_acc <- mean(pred == test_set$b_income)


# The second ensemble includes all methods as above except linear regression.

 table1 <- p_lda %>%
  cbind(p_knn) %>%
  cbind(p_rpart) %>%
  cbind(p_rf)

 pred1 <- sapply(1:nrow(table1), function(i){
   ifelse(mean(table1[i,] == 1) > 0.5, 0, 1)
 })

 ensemble_4_acc <- mean(pred1 == test_set$b_income)


# The third ensemble includes 3 methods: KNN, RPART, and RANDOM FOREST.

 table2 <- p_knn %>%
  cbind(p_rpart) %>%
  cbind(p_rf)

 pred2 <- sapply(1:nrow(table2), function(i){
   ifelse(mean(table2[i,] == 1) > 0.5, 0, 1)
 })

 ensemble_3_acc <- mean(pred2 == test_set$b_income)

# The results of the three ensembles are presented below in the same table including the accuracy results for each of
# the 5 models.

 table_acc_ensemble <- table_acc %>%
  rbind(tibble("Model without 'fnlwgt'" = "Ensemble 5 Models", "Accuracy" = ensemble_5_acc)) %>%
  rbind(tibble("Model without 'fnlwgt'" = "Ensemble 4 Models", "Accuracy" = ensemble_4_acc)) %>%
  rbind(tibble("Model without 'fnlwgt'" = "Ensemble 3 Models", "Accuracy" = ensemble_3_acc))

 table_acc_ensemble %>% knitr::kable()

# The ensemble does not improve the accuracy, no matter what models were included in the ensemble. Still, the accuracy of the
# RANDOM FOREST model was the best, and better than each one of the three ensembles.

# Explanation may be that random forest has already averaged over a large number of random classification trees, and its
# result is already better than each one of the other four models tested. The ensembles tested did not acheive better accuracy
# result than the RANDOM FOREST model.

# save(table_acc_ensemble, file = "table_acc_ensemble.rda")



#################
# Saving objects.
#################

# getwd() # to make sure you are in the directory where you want the files to be saved to.

# setwd("./rdas") # the files will be saved to a sub-directory of the working directory called "rdas". This sub-directory 
# will need to be created in advance if you want to save the files there.

# getwd() # to make sure you are in the directory where you want the files to be saved to.

# The train and test sets will be saved under different names to signify that they do not include the "fnlwgt" column (n_f).

# train_set_n_f <- train_set

# test_set_n_f <- test_set

# save(test_set_n_f, file = "test_set_n_f.rda")

# save(train_set_n_f, file = "train_set_n_f.rda")

# Please note: The object names "train_set" and "test_set" are used below for running models with 'fnlwgt' included in the 
# analysis. If you would like to save both sets of train and test sets (with and without 'fnlwgt') you would need to give them
# different names, as above.
 

# The object below is required in order to knit the Rmd file. If you would like to knit the Rmd file, you will need to save the object below.
 
# save(imp_n_f, file = "imp_n_f.rda")




###################################################
# Results (of running the models without 'fnlwgt')
###################################################


table_acc %>% knitr::kable()

# save(table_acc, file = "table_acc.rda")





################################################################################################
# Running the same code as above, this time including the variable "fnlwgt" in the analysis.
################################################################################################


# Building a dataframe "df" containing the matrix of predictors "x", and the vector of outcomes "y".

df <- data.frame(c(x,y))


#################################
# Creation of train and test set. 
#################################


# 80% of the original dataset will be included in the train set. 20% of the original dataset will be included in the test set.

set.seed(1, sample.kind = "Rounding")

test_index <- createDataPartition(df$b_income, times = 1, p = 0.2, list = FALSE)

train_set <- slice(df, -test_index)

test_set <- slice(df, test_index)

# I will now make sure that there are no levels in any factor at the test_set that are not included in the 
# train_set.

temp <- test_set

test_set <- test_set %>%
  semi_join(train_set, by = "workclass") %>%
  semi_join(train_set, by = "education") %>%
  semi_join(train_set, by = "marital.status") %>%
  semi_join(train_set, by = "occupation") %>%
  semi_join(train_set, by = "relationship") %>%
  semi_join(train_set, by = "race") %>%
  semi_join(train_set, by = "sex") %>%
  semi_join(train_set, by = "native.country")

Removed <- temp %>%
  anti_join(test_set)

# nrow(Removed)

# I will add the line removed from the test_set back into the train_set.

train_set <- train_set %>%
  rbind(Removed)

rm(temp, Removed)

# nrow(train_set)

# nrow(test_set)



################################
# Model (1) - Linear Regression
################################

# The first model is linear regression.

fit_lm <- lm(b_income ~ ., data = train_set)

p_lm <- predict(fit_lm, type = "response", newdata = test_set)

y_hat_lm <- ifelse(p_lm > 0.5, 1, 0)


# Confusion matrix

cm_lm <- confusionMatrix(as.factor(y_hat_lm), as.factor(test_set$b_income))

# cm_lm

# The table "table_acc1" will store the accuracy results for all of the models used (with "fnlwgt" included in the analysis).

table_acc1 <- tibble(Method = "Linear Regression", Accuracy = cm_lm$overall["Accuracy"])



##################
# Model (2) - LDA
##################

fit_lda <- train(as.factor(b_income) ~ ., data = train_set, method = "lda")

p_lda <- predict(fit_lda, newdata = test_set)


cm_lda <- confusionMatrix(p_lda, as.factor(test_set$b_income))

# cm_lda

# The accuracy of LDA is higher than the one obtained with linear regression.


table_acc1 <- table_acc1 %>%
  rbind(tibble(Method = "LDA", Accuracy = cm_lda$overall["Accuracy"]))




##################
# Model (3) - KNN
##################

fit_knn <- knn3(as.factor(train_set$b_income) ~ ., data = train_set, k = 5)

p_knn <- predict(fit_knn, type = "class", newdata = test_set)

cm_knn <- confusionMatrix(p_knn, as.factor(test_set$b_income))

# cm_knn

table_acc1 <- table_acc1 %>%
  rbind(tibble(Method = "KNN", Accuracy = cm_knn$overall["Accuracy"]))



#########################################
# Model (4) - RPART (Classification Tree)
#########################################

# Cross-validation to select the best parameter "cp" for the RPART model.

cp.tuning = data.frame(cp = seq(0, 0.1, len = 25))

set.seed(1, sample.kind = "Rounding")

train_rpart <- train(method = "rpart", as.factor(b_income) ~ ., data = train_set, tuneGrid = cp.tuning)

cp <- train_rpart$bestTune %>% pull(cp)

cp

# Running the RPART model with the best "cp" parameter selected above.

set.seed(1, sample.kind = "Rounding")

fit_rpart <- rpart(as.factor(b_income) ~ ., data = train_set, cp = cp)

# Plotting the resulting tree.

# png("rpart_tree.png")

# plot(fit_rpart, margin = 0.05)

# text(fit_rpart, cex = 0.75)

# dev.off()

# Note that this tree is identical to the one created for the model run without the variable "fnlwgt". This tree does use 
# the variable "fnlwgt" to make predictions, therefore the two trees are identical, with or without the variable "fnlwgt".

# The tree contains the following conditions: Relationship - if unmarried then the income depends on capital.gain. If married, 
# the income depends on education, then on capital.gain, then on occupation. If the education is below a certain level, then
# in addition to capital.gain and occupation, the income depends on age > 32.5, capital.loss > 1846, on hours.per.week > 34.5, 
# again on the level of education, on workclass, then on occupation again, and then again on workclass. Different categories of
# these variables are used at different levels of the tree.


p_rpart <- predict(fit_rpart, test_set, type = "class")

cm_rpart <- confusionMatrix(p_rpart, as.factor(test_set$b_income))

# cm_rpart

table_acc1 <- table_acc1 %>%
  rbind(tibble(Method = "RPART", Accuracy = cm_rpart$overall["Accuracy"]))



############################
# Model (5) - RANDOM FOREST
############################

set.seed(1, sample.kind = "Rounding")

fit_rf <- randomForest(as.factor(b_income) ~ ., data = train_set)

p_rf <- predict(fit_rf, newdata = test_set)

imp_w_f <- importance(fit_rf)

# view(imp_w_f)

cm_rf <- confusionMatrix(p_rf, as.factor(test_set$b_income))

# cm_rf

table_acc1 <- table_acc1 %>% 
  rbind(tibble(Method = "Random Forest", Accuracy = cm_rf$overall["Accuracy"]))

colnames(table_acc1) <- c("Model with 'fnlwgt'", "Accuracy")


table_acc1 %>% knitr::kable()




###########
# Ensemble
###########

# The following code is used to build three ensembles: (1) With all 5 models; (2) With 4 models, not including LM, and (3) With 3
# models, including KNN, RPART and RANDOM FOREST (not including LM and LDA). The accuracy of the three ensembles built was less
# than the accuracy of RANDOM FOREST. The three ensembles I tried do not appear at the final accuracy results table.

# The first ensemble includes all 5 methods: LM, LDA, KNN, RPART, and RANDOM FOREST.

 table <- as.factor(y_hat_lm) %>%
  cbind(p_lda) %>%
  cbind(p_knn) %>%
  cbind(p_rpart) %>%
  cbind(p_rf)

pred <- sapply(1:nrow(table), function(i){
  ifelse(mean(table[i,] == 1) > 0.5, 0, 1)
})

 ensemble_5_acc <- mean(pred == test_set$b_income)


# The second ensemble includes all methods mentioned above except linear regression.

 table1 <- p_lda %>%
  cbind(p_knn) %>%
  cbind(p_rpart) %>%
  cbind(p_rf)

 pred1 <- sapply(1:nrow(table1), function(i){
   ifelse(mean(table1[i,] == 1) > 0.5, 0, 1)
 })

 ensemble_4_acc <- mean(pred1 == test_set$b_income)


# The third ensemble includes 3 methods, including: KNN, RPART, and RANDOM FOREST.

 table2 <- p_knn %>%
  cbind(p_rpart) %>%
  cbind(p_rf)

 pred2 <- sapply(1:nrow(table2), function(i){
   ifelse(mean(table2[i,] == 1) > 0.5, 0, 1)
 })

 ensemble_3_acc <- mean(pred2 == test_set$b_income)

# The results of the three ensembles are presented below in the same table including the accuracy results for each of
# the 5 models.

 table_acc_ensemble1 <- table_acc1 %>%
  rbind(tibble("Model with 'fnlwgt'" = "Ensemble 5 Models", "Accuracy" = ensemble_5_acc)) %>%
  rbind(tibble("Model with 'fnlwgt'" = "Ensemble 4 Models", "Accuracy" = ensemble_4_acc)) %>%
  rbind(tibble("Model with 'fnlwgt'" = "Ensemble 3 Models", "Accuracy" = ensemble_3_acc))

 table_acc_ensemble1 %>% knitr::kable()

# The ensemble does not improve the accuracy, no matter what models were included in the ensemble. Still, the accuracy of the
# RANDOM FOREST model was the best, and better than each one of the three ensembles.

# Explanation may be that random forest has already averaged over a large number of random classification trees, and its
# result is already better than each one of the other four models tested. The ensembles tested did not acheive better accuracy
# result than the RANDOM FOREST model.

# Furthermore, the results of the ensemble are even worse for the ensembles including models that included 'fnlwgt' in their
# analysis in comparison with ensembles that did not include 'fnlwgt' in their analysis. This might result for the different 
# models using 'fnlwgt' making different mistakes, therefore when an ensemble is created, it performs worse then each one of
# the models separately.
 
 

#################################################################################################################
# Results - Showing Accuracy Results of 5 Models With and Without the Variable "fnlwgt" included in the analysis.
#################################################################################################################


table_acc_all <- table_acc %>%
  cbind(table_acc1)

table_acc_all %>% knitr::kable()

table_acc_all_ensemble <- table_acc_ensemble %>%
  cbind(table_acc_ensemble1)

table_acc_all_ensemble %>% knitr::kable()

##############################
# Saving and removing objects.
##############################

# getwd() # to make sure you are in the directory where you want the files to be saved to.

# setwd("./rdas") # the files will be saved to a sub-directory of the working directory called "rdas". This sub-directory 
# will need to be created in advance if you want to save the files there.

# getwd() # to make sure you are in the directory where you want the files to be saved to. If you want to knit the attached Rmd
# file yourself, you will need to save objects into the directory where the Rmd file is located.

# save(adult, file = "adult.rda")

# save(census, file = "census.rda")

# save(file, file = "file.rda")

# save(url, file = "url.rda")


# save(imp_w_f, file = "imp_w_f.rda")


# save(table_acc1, file = "table_acc1.rda")

# save(table_acc_ensemble1, file = "table_acc_ensemble.rda")


# save(table_acc_all, file = "table_acc_all.rda")

# save(table_acc_all_ensemble, file = "table_acc_all_ensemble.rda")


# Please note: the following train_set and test_set objects are the train and test sets that include the column "fnlwgt".

# save(train_set, file = "train_set.rda")

# save(test_set, file = "test_set.rda")


# I will give new names to the train and test sets that include 'fnlwgt' and then save them again.

# test_set_w_f <- test_set

# train_set_w_f <- train_set

# save(test_set_w_f, file = "test_set_w_f.rda")

# save(train_set_w_f, file = "train_set_w_f.rda")


# Please note: I have not given new names to the objects below. Therefore, The following objects represent the analysis conducted
# with 'fnlwgt'.

# save(df, file = "df.rda")

# save(cp, file = "cp.rda")

# Before removing all objects, I present the tables of results again.



################
# Final Results
################

table_acc_all %>% knitr::kable()

table_acc_all_ensemble %>% knitr::kable()

# rm(list = ls())

# All objects have now been removed.
