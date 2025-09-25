##############################################
#####     PREDICTING ALGAE BLOOMS        #####
#####     ERROR CHECKING EXERCISES      #####
##############################################

# It is always a good idea to "get to know your
# residuals". Although regression trees do not
# rely on the assumptions of ordinary least
# squares regression, it can still be useful
# to get to know what your error distribution
# "looks like".

# We saw some potential problems in our model with
# the assumptions of linear regression, repeated below:

# Load the DMwR package, the algae data,

library("DMwR")
data(algae)

#########################################
###    Multiple Linear Regression     ###
#########################################

# We get rid of the rows that have more
# than 20% missing (rows 62 and 199):
algae <- algae[-manyNAs(algae), ]

# We use knnImputation() to fill remaining
# missing values with 10 "most similar" records
clean.algae <- knnImputation(algae, k = 10)

# Here are the observed values of a1, the target variable
(target.value <- clean.algae$a1)

# Now there are no missing values so we fit
# a linear regression model for predicting
# one of the algae (algae a1):
lm.a1 <- lm(a1 ~ .,data=clean.algae[,1:12])

# Here are the predicted values of a1
?predict.lm
(predicted.value <- predict(lm.a1))

# Here is the difference
target.value - predicted.value
# Should be same as residuals, 
# and it appears to be
lm.a1$residuals

# check diagnostics with

## 4 plots on 1 page;
## allow room for printing model formula in outer margin:
par(mfrow = c(2, 2), oma = c(0, 0, 2, 0))
plot(lm.a1)
# lots of problems...heteroscedascity, non-normal
# distribution in QQ, looks quadratic, and
# observation 153 has too much leverage.

# reset graphic parameters
par(mfrow =c(1,1))

# To read what these plots mean:
?plot.lm

# plot a histogram of the residuals
hist(lm.a1$residuals)

# plot a smooth density plot of residuals
plot(density(lm.a1$residuals))

# residuals (differences target versus predicted)
# appear to be right-skewed.

# Intuitively, the skewness is a measure of symmetry. 
# As a rule, negative skewness indicates that the 
# mean of the data values is less than the median, 
# and the data distribution is left-skewed. Positive
# skewness would indicates that the mean of the data
# values is larger than the median, and the data 
# distribution is right-skewed. 

# Can measure skewness with moments package
install.packages("moments")
library("moments")
skewness(lm.a1$residuals) # right-skewed

# Can also measure kurtosis ("peakedness")
kurtosis(lm.a1$residuals)
help(kurtosis)

# It is also possible to compare the distributions
# of the error terms with either the predicted values
# or with the observed values of the target variable.

# We want to know if the errors likely come from the
# same distribution as the target value and/or
# the predicted value, and for that matter whether
# the target and predicted likely come from the
# same distribution.

# We use non-parametric Kolmogorov-Smirnov test
# which is more robust, especially with outliers.
# Can use to test null hypothesis that both come
# from same distribution. It calculates a statistic
# that measures the maximum difference between the
# two empirical cumulative distribution functions.
# Similar distributions will have small ECDF diffs.

help(ks.test)

# test null hypothesis that the observed (target)
# value of a1 and the predicted value of a1
# come from the same distribution
ks.test(target.value,predicted.value)

# It does not appear so. Is this surprising?

# How about the residuals and the target?
ks.test(lm.a1$residuals, target.value)
# absolutely not ! Remember, the residuals
# should more or less center around zero. The
# target variable values do not.

?scale
# To compare the distribution of residuals with
# either the target variable values or the predicted
# values of the target variable, we should 'mean-
# center' the latter two.

# 'mean-center' or 'zero-center' target values:
centered.target.value <- scale(target.value,
                               center=TRUE,
                               scale=FALSE)

# 'mean-center' or 'zero-center' predicted values:
centered.predicted.value <- scale(predicted.value,
                               center=TRUE,
                               scale=FALSE)

# or, for brevity:
ctv <- centered.target.value
cpv <- centered.predicted.value
# shortened name for residuals:
crv <- lm.a1$residuals

# Then compare the distributions
ks.test(crv, ctv)
# They are still likely from different distributions

# How about the residuals and the predicted?
ks.test(crv, cpv)
# We fail to reject the null hypothesis and
# we so note that they might be 'sampled'
# from the same distribution

# Let's plot all three together:
par(mfrow = c(3, 1))
hist(crv)
hist(cpv)
hist(ctv)
par(mfrow = c(1, 1))

# We can clearly see that the original target
# variable values are heavily "loaded" on low
# original values (remember, we centered this
# column of data so now have negative values).

# It "looks" like it might be an exponential
# distribution (but we cannot rely on "looks")

# step() performs the backward elimination
# process to our original model fit. 
final.lm <- step(lm.a1)

# We have the same evident problems with
# the assumptions of linear regression

par(mfrow = c(2, 2), oma = c(0, 0, 2, 0))
plot(final.lm)
# same problems...heteroscedascity, non-normal
# distribution in QQ, looks quadratic, and
# observation 153 has too much leverage.

# reset graphic parameters
par(mfrow =c(1,1))

# draw a histogram of final residuals:
hist(final.lm$residuals)

# Give residuals of final model a shorter name:
cfrv <- final.lm$residuals

# plot the 2 sets of residuals together
par(mfrow = c(2, 1))
hist(crv,xlim=c(-40,60))
hist(cfrv,xlim=c(-40,60))
par(mfrow = c(1, 1))

# The residual distributions "look" similar.

# Do they come from the same distribution?
ks.test(crv, cfrv)
# Yes, it looks like it

# Below you have the script for modeling the
# regression trees and linear model using NMSE
# as the error metric. Can you similarly plot
# and test those error distributions to see if
# they "look" similar and if they likely are
# 'sampled' from the same distributions?

# What do you discover?

# How about performing a similar set of "tests"
# with the output from (some variant) of the
# random forests?

# What do you discover?

#########################################
### Regression Trees as Model
#########################################

# We now use regression trees to predict
# value of frequencies of algae a1.

# Regression trees are tolerant of missing
# values but still need to remove obs 62 and
# 199

# We need library rpart (acronym for Recursive
# Partitioning And Regression Trees)
library(rpart)

# We reload data set:
data(algae)

# Trim away record 62 and 199 only:
algae <- algae[-manyNAs(algae), ]

# fit the tree, note same function form as lm()
# tree estimating is simulation, to get consistent
# results, need to set.seed
set.seed(1234)
rt.a1 <- rpart(a1 ~ .,data=algae[,1:12])

?rpart

# RT is hierarchy of logical tests on most
# relevant explanatory variables. But unless
# set seed, will get varying results

# We see root node has 198 samples, average
# value of a1 is 16.99 and deviance from this
# average is 90401.29

rt.a1

# Are two branches on each node.

# we examine content of object rt.a1 
# which is the regression tree. Each node is
# related to the outcome of a test on one of
# the predictor variables. At each node, shows
# average value for a1 and the variance.
# We continue testing until we reach a leaf
# node (marked with asterisk) and there we find
# predictions for the levels of a1


# Trees are usually obtained in two steps.
# 1) Grow a large, bushy tree and then
# 2) Prune this tree by deleting bottom nodes
# through a process of statistical elimination.

# Can do both in one step with rpartXse() which
# takes se as argument (default is se=1.0)

set.seed(1234) # Just to ensure same results
(rt.a1 <- rpartXse(a1 ~ .,data=algae[,1:12]))

###################################################
### Model Evaluation and Selection
###################################################

# So which one should we use? Multiple regression
# model or regression trees? Need to specify
# preference criteria over the space of models.

# Are many criteria for evaluating models. Some
# of the most popular are criteria that calculate
# the predictive performance of the models.

# Assess predictive performance of regression
# models by comparing model predictions with
# real values of target variables and then
# calculating some average error measure.

# We will use Mean Absolute Error (MAE) for both
# multiple regression and regression tree models.

# First step is to obtain model predictions for 
# set of cases over which we want to evaluate.

# Generic R function for this is predict(), takes
# a model and test data set as arguments and
# returns correspondent model predictions.

# get predictions from final regression model, we
# use clean.algae because of missing values:
lm.predictions.a1 <- predict(final.lm,clean.algae)

# get predictions for final regression tree model:
rt.predictions.a1 <- predict(rt.a1,algae)

# We calculate Mean Absolute Errors:
(mae.a1.lm <- mean(abs(lm.predictions.a1-algae[,'a1'])))
(mae.a1.rt <- mean(abs(rt.predictions.a1-algae[,'a1'])))

# Another popular error is Mean Squared Error:
(mse.a1.lm <- mean((lm.predictions.a1-algae[,'a1'])^2))
(mse.a1.rt <- mean((rt.predictions.a1-algae[,'a1'])^2))

# MSE is not measured in same units as target
# variable so interpretation is more difficult.

# So we use normalized mean squared error which
# calculates a ratio between the performance of
# the models and that of a baseline predictor,
# usually the mean value of the target variable:

# Calculate NMSE for regression model:
(nmse.a1.lm <- mean((lm.predictions.a1-algae[,'a1'])^2)/
                mean((mean(algae[,'a1'])-algae[,'a1'])^2))

# Calculate NMSE for regression tree model:
(nmse.a1.rt <- mean((rt.predictions.a1-algae[,'a1'])^2)/
                mean((mean(algae[,'a1'])-algae[,'a1'])^2))

# NMSE is a unit-less error measure with values
# usually ranging from zero to one. If the model
# performs better than very simple baseline (mean),
# NMSE should be less than one, and the smaller,
# the better. Values greater than one means the
# model predicts worse than the simple (mean)
# baseline measure.

# regr.eval() function calculates the value of a 
# set of regression evaluation metrics:
regr.eval(algae[,'a1'],rt.predictions.a1,
          train.y=algae[,'a1'])

# We want to visually inspect model predictions.

# This is a common R theme and practice, to
# inspect data before and then performance
# evaluation measures after.

# We go with a scatterplot of the errors.
# Reset graphics parameters to draw two plots
# side-by-side in one frame:
old.par <- par(mfrow=c(1,2))
# Call the scatterplot (is a high-level function)
plot(lm.predictions.a1,
     algae[,'a1'],main="Linear Model",
     xlab="Predictions",ylab="True Values")
# Draw line (low-level base graphics function)
abline(0,1,lty=2)
# Call second plot (high-level graphics function)
plot(rt.predictions.a1,algae[,'a1'],main="Regression Tree",
     xlab="Predictions",ylab="True Values")
# Draws another line on the currently active plot:
abline(0,1,lty=2)
# resets the graphics parameters back to default:
par(old.par)

# Looking at the resulting plots we observe that the
# models have poor performance in several cases.
# Ideally, all circles lie on the dashed line which
# cross the origin and represent where X = Y.

# We can determine the sample numbers for the bad
# predictions with identify():

plot(lm.predictions.a1,algae[,'a1'],
     main="Linear Model",
     xlab="Predictions",ylab="True Values")
abline(0,1,lty=2)
algae[identify(lm.predictions.a1,algae[,'a1']),]

# So you identify the clicked circles by their rows
# which are then visible as  they are in a vector
# returned by identify() to index the algae dataframe.

# We see that some of the linear model predictions
# are negative algae frequencies. This does not make
# sense so we use this 'domain knowledge' and the
# minimum possible value of 0 algae frequency to
# improve the linear model performance, replacing the
# negative predictions with 0 predictions.

# Create new model 'sensible....':
sensible.lm.predictions.a1 <- ifelse(lm.predictions.a1 < 0,0,lm.predictions.a1)

# The previous model gave us:
regr.eval(algae[,'a1'],
          lm.predictions.a1,
          stats=c('mae','mse'))

# The sensible model gives us an improvement:
regr.eval(algae[,'a1'],
          sensible.lm.predictions.a1,
          stats=c('mae','mse'))

### k-fold CV

# We want to choose the best model for obtaining the
# predictions on the 140 test samples. As we do not
# know the target variables values for those samples,
# we have to estimate which of our models will perform
# better on these test samples.

# Our key issue here is to obtain a reliable estimate
# of model performance on data for which we do not
# know the true target value. You can overfit on
# training data, and predict that data perfectly, but
# that model will not necessarily generalize to new
# data.

# How do you achieve a reliable estimate of a model's
# performance on unseen data?

# k-fold Cross Validation is often used to obtain
# reliable estimates using small datasets: Obtain
# k equally-sized and random subsets of the training
# data. For each of the k subsets, build a model
# using the remaining k-1 sets and evaluate this
# model on the kth set. Store the performance of the
# model and repeat this process for all remaining
# subsets.

# In the end, we have k performance measures, all
# obtained by testing a model on data not used in
# the construction of the model.

# The k-fold Cross Validation estimate is the average
# of these k measures. Often k is set as = 10. Further,
# this overall k-fold CV process can be repeated to
# improve the reliability of the performance estimates.

### experimentalComparison() function in DMwR package

# In general, when faced with a predictive task, we
# 1) Select alternative models to consider;
# 2) Select the evaluation metrics which will be
#    used to compare the models; and
# 3) Choose among available experimental methodologies
#    for obtaining reliable estimates of these metrics.

# DMwR package has experimentalComparison() function.
# Result of the function is an object of class compExp

?experimentalComparison

class?compExp

# experimentalComparison() has 3 parameters: (1) data
# set to use; (2) alternative models; and (3) choice
# of experimental methodology for obtaining reliable
# performance evaluation metrics estimates.

# experimentalComparison() is generic in that it can
# be used for any model(s) and any dataset(s). The
# user supplies a set of functions implementing the
# models to be compared. Each function should imple-
# ment a full train+test+evaluate cycle for the
# given training and test datasets. The functions are
# called from the experimental routines on each
# iteration of the estimation process. They should
# return a vector with the values of whatever
# evaluation metrics the user wants to estimate by
# cross-validation.

# Here we construct such functions for two target
# models:

# user-defined cross validation function
# for regression tree model. The arguments are
# the formula, the training data, and the test data.
cv.rpart <- function(form,train,test,...) {
  # train:
  m <- rpartXse(form,train,...)
  # test:
  p <- predict(m,test)
  # evaluate:
  mse <- mean((p-resp(form,test))^2)
  c(nmse=mse/mean((mean(resp(form,train))-resp(form,test))^2))
}

# user-defined cross validation function
# for linear regression model. The arguments are
# the formula, the training data, and the test data.
cv.lm <- function(form,train,test,...) {
  # train:
  m <- lm(form,train,...)
  # test:
  p <- predict(m,test)
  # add our rule:
  p <- ifelse(p < 0,0,p)
  # evaluate:
  mse <- mean((p-resp(form,test))^2)
  c(nmse=mse/mean((mean(resp(form,train))-resp(form,test))^2))
}

# Note we assume the use of NMSE as evaluation metric.
# for both regression trees and linear models.

# Both functions carry out a train+test+evaluate
# cycle akthough each uses a different learning algo-
# rithm. Both return a named vector with the score
# in terms of NMSE.

# The '...' (triple dot argument) allows additional
# (variable number) of arguments to be passed to the
# function after the first three which are specified
# by name. So one may pass extra learning parameters
# to the learning function (to rpartXse() in one case
# and to lm() in another). The resp() function from
# DMwR obtains target variable values of a data set
# given a formula:

?resp

# Having defined our functions that will carry out
# the learning and testing phase of our models, we
# then carry out the cross-validation comparison:

res <- experimentalComparison(
  # first argument is vector of data sets in form
  # dataset(<formula>,<data frame>,<label>)
            c(dataset(a1 ~ .,clean.algae[,1:12],'a1')),
  # second argument is vector of learning system variants
  # with name of functions to carry out learn+test+evaluate cycle
            c(variants('cv.lm'), 
              variants('cv.rpart',
  # we allow different values for se:
                       se=c(0,0.5,1))),
  # third argument specifies 3 reps of k-fold cv process,
  # that k=10, and 1234 is random seed:
            cvSettings(3,10,1234))

# variants() function generates a set of alternative
# models resulting from all possible combinations of
# the parameter values. We use cv.lm() only with its
# default parameters and for cv.rpart() we specify
# different alternative values for parameter se. So
# we get three variants of regression trees. See info
# about third argument above.

# Result is complex object with all information
# about the experimental comparison

summary(res)

# We note the plots below shows that one of the
# variants of the regression tree achieves the
# best results.

# To visualize:
plot(res)

# experimental Comparison() assigns a label to 
# each model variant. If you want to know specific
# parameter settings for any one label:

getVariant('cv.rpart.v1',res)

# We can execute a similar comparative experiment
# for all seven prediction tasks with sapply():

# starts by creating vector of datasets to use
# for seven prediction tasks
DSs <- sapply(names(clean.algae)[12:18],
         function(x,names.attrs) { 
           # create a separate formula for each
           f <- as.formula(paste(x,"~ ."))
           dataset(f,clean.algae[,c(names.attrs,x)],x) 
         },
         names(clean.algae)[1:11])

# Now we use vector of datasets created above
res.all <- experimentalComparison(
                  DSs,
                  c(variants('cv.lm'),
                    variants('cv.rpart',
                             se=c(0,0.5,1))
                   ),
        # carry out 5 reps of 10-fold CV
                  cvSettings(5,10,1234))

summary(res.all)

# Visualize results, we have some very bad results
# When NMSE > 1, that is baseline as competitive as
# predicting always the average target variable
# value for all test cases.
plot(res.all)

# To check which is the best model, we use the
# function bestScores()
?bestScores

bestScores(res.all)

# We see that except for algae 1, results are
# disappointing. There is so much variability (see
# previous plot) that a random forest ensemble
# approach might be a better candidate.

# Ensemble approaches generate a large set of
# alternative models and combine their predictions.

# We can generate a random forest of trees (too
# much to cover here but let's look at the results)

library(randomForest)
cv.rf <- function(form,train,test,...) {
  m <- randomForest(form,train,...)
  p <- predict(m,test)
  mse <- mean((p-resp(form,test))^2)
  c(nmse=mse/mean((mean(resp(form,train))-resp(form,test))^2))
}

res.all <- experimentalComparison(
                  DSs,
                  c(variants('cv.lm'),
                    variants('cv.rpart',
                             se=c(0,0.5,1)),
                    variants('cv.rf',
                             ntree=c(200,500,700))
                   ),
                  cvSettings(5,10,1234))

# We see the advantages on an ensemble approach

bestScores(res.all)

# In all cases except for algae 7, the best score
# is obtained by some variant of a random forest.
# But some results are still not very good.

# But how tell if the difference between the scores
# of the best models and the remaining alternatives
# is statistically significant?

?compAnalysis # gives us Wilcoxon tests (non-parametric)

compAnalysis(res.all,against='cv.rf.v3',
               datasets=c('a1','a2','a4','a6'))

############################################
### Predictions for the seven algae
############################################

# We need to obtain predictions for the seven algae
# on the 140 test samples. We will use the model 
# that our cross validation indicated as "best"
# in our call to the bestScores() function, either
# "cv.rf.v3", "cv.rf.v2", "cv.rf.v1" or "cv.rpart.v3".

# We begin by obtaining these models using all of
# the available training data so we can apply them 
# to the test set.

# Regression trees incorporate their own method for
# handling missing values. Random forests do not, so
# we use the clean.algae dataframe.

# Here we obtain all seven models:

# Obtain vector with names of winning variants:
bestModelsNames <- sapply(bestScores(res.all),
                          function(x) x['nmse',
                                        'system'])

learners <- c(rf='randomForest',rpart='rpartXse') 

# Obtain names of R functions that
# learn these variants using strsplit():
funcs <- learners[sapply(strsplit(bestModelsNames,'\\.'),
                        function(x) x[2])]
# store parameter setting for each of winning
# variants, getVariant() function gives the model
# corresponding to a variant name:
parSetts <- lapply(bestModelsNames,
                   # 'pars' is a slot, a list with
                   # parameters of the variant:
                   function(x) getVariant(x,res.all)@pars)

# obtain models and stores them in bestModels list:
bestModels <- list()
for(a in 1:7) {
  form <- as.formula(paste(names(clean.algae)[11+a],'~ .'))
  # do.call() allows us to call any function by
  # providing its name as a string on the first
  # argument, and then including the arguments of
  bestModels[[a]] <- do.call(funcs[a],
  # the call as a list in the second argument:
          c(list(form,clean.algae[,c(1:11,11+a)]),parSetts[[a]]))
}

# So now we have a list with seven models obtained
# for each algae and can use for making predictions
# for the test set.

# clean up missing values for random forests.
# df 'test.algae' contains 140 test samples:
clean.test.algae <- knnImputation(test.algae,k=10,distData=algae[,1:11])

# set up matrix to hold the predictions, all NAs:
preds <- matrix(ncol=7,nrow=140)
# obtain the predictions, store each one as one of
# the seven rows in the matrix 'preds'
for(i in 1:nrow(clean.test.algae)) 
  preds[i,] <- sapply(1:7,
                      function(x) 
                        predict(bestModels[[x]],clean.test.algae[i,])
                     )

# obtain average predictions 'on the margin' of
# columns
avg.preds <- apply(algae[,12:18],2,mean)
# We compare the predictions with the true values
# as QA on our approach to this prediction problem.
# 'True' values are in 'algae.sols' df. scale()
# function normalizes a dataset, it subtracts the
# second argument by the first and divides the
# result by the third argument, unless FALSE (as here).
# We subtract a vector, the average target value,
# from each line of the matrix
apply( ((algae.sols-preds)^2),2,mean) / apply( (scale(algae.sols,avg.preds,F)^2),2,mean)

# Average value of target variable is our prediction
# of the baseline model used to calculate the NMSE,
# which in our case consists of predicting the
# average value of the target variable.

# Then we calculate NMSEs for the seven models/algae.
# Our results are similar to the cv estimates
# previously obtained.

# Is difficult to obtain good scores for algae a7,
# much better with algae a1.

##################   FINI   ######################
