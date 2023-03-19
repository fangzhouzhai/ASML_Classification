banks <- readr::read_csv("https://www.louisaslett.com/Courses/MISCADA/bank_personal_loan.csv")
View(banks)

library("skimr")
skim(banks)

library("tidyverse")
library("ggplot2")

DataExplorer::plot_bar(banks, ncol = 3)
DataExplorer::plot_histogram(banks, ncol = 3)
DataExplorer::plot_boxplot(banks, by = "Personal.Loan", ncol = 3)

#install.packages("patchwork")
library(patchwork)

p1 <- ggplot(banks, aes(x = Personal.Loan, fill = factor(Securities.Account))) + geom_bar(position = "stack") +  scale_x_continuous(breaks=c(0,1))
p2 <- ggplot(banks, aes(x = Personal.Loan, fill = factor(CD.Account))) + geom_bar(position = "stack") +  scale_x_continuous(breaks=c(0,1))
p3 <- ggplot(banks, aes(x = Personal.Loan, fill = factor(Online))) + geom_bar(position = "stack") +  scale_x_continuous(breaks=c(0,1))
p4 <- ggplot(banks, aes(x = Personal.Loan, fill = factor(CreditCard))) + geom_bar(position = "stack") +  scale_x_continuous(breaks=c(0,1))
p5 <- ggplot(banks, aes(x = Personal.Loan, fill = factor(Education))) + geom_bar(position = "stack") +  scale_x_continuous(breaks=c(0,1))

p1+p2
p3+p4
p5

ggplot(data = banks) + geom_point(mapping = aes(x = Income, y = Personal.Loan)) + scale_y_continuous(breaks=c(0,1))
ggplot(data = banks) + geom_point(mapping = aes(x = Mortgage, y = Personal.Loan)) + scale_y_continuous(breaks=c(0,1))
ggplot(data = banks) + geom_point(mapping = aes(x = CCAvg, y = Personal.Loan)) + scale_y_continuous(breaks=c(0,1))
ggplot(data = banks) + geom_point(mapping = aes(x = Experience, y = Personal.Loan)) + scale_y_continuous(breaks=c(0,1))

p1 <- ggplot(banks, aes(Income)) + geom_histogram()
p2 <- ggplot(banks, aes(Age)) + geom_histogram()
p3 <- ggplot(banks, aes(Mortgage)) + geom_histogram()
p4 <- ggplot(banks, aes(CCAvg)) + geom_histogram()
p1+p2+p3+p4

library("GGally")

ggpairs(data = banks, 
        columns = c("Income","CCAvg","Education","Securities.Account"),
        aes(color = factor(Personal.Loan))) 

banks[, 'Personal.Loan'] <- as.factor(banks$Personal.Loan) 

set.seed(212)

loan_task <- TaskClassif$new(id = "BanksLoan",
                             backend = banks, 
                             target = 'Personal.Loan')
loan_task

mlr_learners

?mlr_learners_classif.log_reg

library("mlr3learners")
learner_logreg = lrn("classif.log_reg")

# logistic regression 

train_set = sample(loan_task$row_ids, 0.8 * loan_task$nrow)
test_set = setdiff(loan_task$row_ids, train_set)

learner_lr$train(loan_task, row_ids = train_set)

learner_lr$model

summary(learner_lr$model)

# random forest
learner_rf = lrn("classif.ranger", importance = "permutation")
learner_rf$train(loan_task, row_ids = train_set)

learner_rf
importance = as.data.table(learner_rf$importance(), keep.rownames = TRUE)
colnames(importance) = c("Feature", "Importance")

ggplot(data=importance,
       aes(x = reorder(Feature, Importance), y = Importance)) + 
  geom_col() + coord_flip() + xlab("")


# Decision Tree
learner_dt = lrn("classif.rpart")
learner_dt$train(loan_task, row_ids = train_set)

learner_dt$model

summary(learner_dt$model)

pred_lr <- learner_lr$predict(loan_task, row_ids = test_set)
pred_rf <- learner_rf$predict(loan_task, row_ids = test_set)
pred_dt <- learner_dt$predict(loan_task, row_ids = test_set)

pred_lr
pred_rf
pred_dt

acc_lr <- pred_lr$score(msr("classif.acc"))
acc_rf <- pred_rf$score(msr("classif.acc"))
acc_dt <- pred_dt$score(msr("classif.acc"))

pred_lr$confusion
pred_rf$confusion
pred_dt$confusion

accuracy <- data.frame(model_name = c('logistic regression','random forest','Decision Tree'), accuracy = c(acc_lr,acc_rf,acc_dt), check.names = FALSE)

accuracy

mlr_measures

mlr_resamplings
?mlr_resamplings_cv

# repeat subsampling
resampling = rsmp("subsampling", repeats=10)

lrresample = resample(loan_task, learner = learner_lr, resampling = resampling)

frresample = resample(loan_task, learner = learner_rf, resampling = resampling)

dtresample = resample(loan_task, learner = learner_dt, resampling = resampling)

lrresample$aggregate()
frresample$aggregate()
dtresample$aggregate()

cv10 <- rsmp("cv", folds = 10)
cv_lr <- resample(loan_task, learner_lr, cv10)
cv_rf <- resample(loan_task, learner_rf, cv10)
cv_dt <- resample(loan_task, learner_dt, cv10)

cv5_lr$aggregate()
cv5_rf$aggregate()
cv5_dt$aggregate()

lrn_rf <- lrn("classif.ranger", predict_type = "prob")

bm_design = benchmark_grid(
  tasks = loan_task,
  learners = lrn_rf,
  resamplings = rsmp("cv", folds = 10)
)

bmrf = benchmark(bm_design)

bmrf$aggregate(list(msr("classif.ce"),
                    msr("classif.acc"),
                    msr("classif.auc"),
                    msr("classif.fpr"),
                    msr("classif.fnr")))

lrn_rf$param_set

lowrf = lrn("classif.ranger", id = "low", predict_type = "prob", num.trees = 10, mtry = 5)

medrf = lrn("classif.ranger", id = "med", predict_type = "prob")

highrf = lrn("classif.ranger", id = "high", predict_type = "prob", num.trees = 1000, mtry = 11)

newlearners = list(lowrf, medrf, highrf)

newbm = benchmark_grid(
  tasks = loan_task,
  learners = newlearners,
  resamplings = rsmp("cv", folds = 10)
)

newbmr = benchmark(newbm)

newbmr$aggregate(list(msr("classif.ce"),
                      msr("classif.acc"),
                      msr("classif.auc")))

library(mlr3verse)
#install.packages("precrec")

highrf = as_learner(lrn("classif.ranger", id = "high", predict_type = "prob", num.trees = 1000, mtry = 11))

highrf <- resample(task = loan_task,
                   learner = highrf,
                   resampling = rsmp("cv",folds = 10),
                   store_models = T)

newcfmatrix <- highrf$prediction()$confusion
autoplot(highrf$prediction())
highrf$aggregate(msrs(c("classif.auc","classif.acc","classif.bbrier")))
autoplot(highrf,type = "roc")

tp <- newcfmatrix[2, 2]
tn <- newcfmatrix[1, 1]
fp <- newcfmatrix[2, 1]
fn <- newcfmatrix[1, 2]
print(sensitivity <- tp/(tp + fn))
print(specificity <- tn/(tn + fp))
