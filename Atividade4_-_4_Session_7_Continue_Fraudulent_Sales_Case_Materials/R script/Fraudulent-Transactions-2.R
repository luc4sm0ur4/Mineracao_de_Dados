##############################################
######  Defining the Data Mining Tasks  ######
##############################################

# We want to decide which transaction reports
# should be considered for inspection as result
# of strong suspicion of being fraudulent.

# We want the guidance to take the form of a
# ranking of fraud probability.

# 'Insp' column has info about previous inspections.

# Have small number that have been judged, are
# 'OK' or 'Fraud'

# Have another large number that have not been
# inspected, they are marked 'unkn'.

# These represent two different groups for our
# purposes; there are different modeling approaches
# that can be applied to either group.

##### Unsupervised Techniques

# For reports not inspected ('unkn'), the 'Insp'
# column is irrelevant; it carries no useful info.

# So these sales reports are only described by a
# set of independent, predictor variables. We are
# facing a "descriptive" data mining task, not
# a "predictive" task as with the supervised methods

# Clustering is a descriptive data mining technique.
# Clustering methods try to find "natural" groupings
# of observations by discovering clusters of cases
# that are 'similar' (or dissimilar) to each other.

# Need a definition of a metric for 'similarity'.
# Distance is often used. Clusters near each other
# are considered members of the same group.

# Outlier detection is also used for "descriptive"
# data mining tasks. they may be based on the
# assumption of a certain distribution of the
# data, outliers deviate from this expected distri-
# bution.

# Or may be based on notion of distance: observations
# 'too far away' from the others are considered
# 'outliers'. This approach has similarities
# with clustering except that the outliers are
# generally not in groups.

# We want an outlier ranking, so the unsupervised
# technique should use must be able to identify
# outliers and to then rank them.

###### Supervised Techniques

# We have a set of transactions labeled as normal
# or fraudulent. Supervised models can be seen as
# approximating an unknown function
# Y = f(X1, X2, ...,Xp) that describes the rela-
# tionship between the target variable Y and the
# predictors (the X's). We want to obtain model
# Parameters that optimize a certain criterion,
# such as minimizing prediction error.

# If Y is continuous, is multiple regression.

# If Y is nominal (groups), is classification.

# Our's is a classification task (ok and fraud).

# And we only have 15,732 of the 401,146 reports
# available for use. Of the 15,732 inspected
# reports, 14,462 are normal and only 1,270 are
# examples of fraud. So must select evaluation
# criteria that can correctly measure and model
# on the less frequent class (fraud), so the
# modeling technique must be able to cope with
# a dataset with a strong class imbalance.

# We must also use adaptations: we want a ranking
# of transactions according to their probability
# of being frauds. So given a test set of new
# reports, the model must be able to correctly
# classify as fraud AND establish a ranking among
# cases classified as fraud.

# So we need a probabilistic classification that
# both classifies and ranks with a probability.

###### Semi-Supervised Techniques

# Here the domain is to include a large proportion
# of unlabeled data AND a small amount of labeled
# data.

# Are usually two types of semi-supervised methods:
# (1) Semi-supervised classification methods that
# try to improve the performance of standard 
# supervised classification algorithms with the
# help of the extra information provided by the
# unlabeled cases; and (2) Semi-supervised
# clustering methods that attempt to bias the
# clustering process by incorporating some form
# of constraints based on the labeled data used
# to form the groups.

# One will bias clustering so that cases with
# the same label end up in the same group ('must-
# link' constraints) while cases with different
# labels end up in different groups ('cannot-link')
# constraints.

# The criterion used to form the clusters is used
# to bias the methods to find the appropriate groups
# of cases.....so notion of distance is "distorted"
# to reflect "must-link" and "cannot-link"
# constraints.

# A well-known method is 'self-training', and 
# iterative approach that starts by obtaining
# a classification model with the given labeled
# data, and then to use the model to classify the
# unlabeled data.

# The cases for which the model has higher confidence
# on the classification are added together with
# the predicted label to the initial training set,
# thus extending it. Using this new set, a new
# model is obtained and the overall process
# repeated until some convergence criterion is
# met.

###### Evaluation Criteria

# We want to evaluate the models based on how well
# each model produces a ranking of the transaction
# reports.

# We have both labeled and unlabeled data.

# This makes it more difficult to compare models
# because the supervised and unsupervised methods
# are usually evaluated differently. It is easier
# to evaluate the methods that classify the labeled
# data.

###### Precision and Recall

# From Wikipedia:
# In pattern recognition and information retrieval, 
# precision (also called positive predictive value) 
# is the fraction of retrieved instances that are
# relevant, while recall (aka sensitivity) is the 
# fraction of relevant instances that are retrieved. 

# Both precision and recall are therefore based on 
# an understanding and measure of relevance. 
# Suppose a program for recognizing dogs in scenes
# identifies 7 dogs in a scene containing 9 dogs 
# and some cats. If 4 of the identifications are 
# correct, but 3 are actually cats, the program's 
# precision is 4/7 while its recall is 4/9. 

# When a search engine returns 30 pages only 20 
# of which were relevant while failing to return 
# 40 additional relevant pages, its precision is
# 20/30 = 2/3 while its recall is 20/60 = 1/3.

# In this application, a successful model should
# obtain a ranking that includes all known frauds
# at the top positions of the ranking. For the k
# top rankings that our resources allow, we want
# only frauds or unknowns and we want to include
# all of the fraud cases, which are in a minority.

# When the aim is to predict a small set of rare
# events (in this case frauds), precision and
# recall are adequate evaluation metrics.

# Given the inspection effort limit k, we can
# calculate the precision and recall of the k-top
# most positions of the ranking. The k-limit
# determines which reports are to be inspected.

# From a supervised classification perspective,
# this is the same as considering the top k
# positions as predictions of the class fraud,
# while the remaining are normal reports.

# The value of precision tells us the proportion
# of these k top-most reports that are, in effect,
# labeled as fraud. The value of recall will
# measure the proportion of frauds in the test
# set that are included in these k top-most
# positions.

# Note the obtained values are pessimistic. If the
# k top-most positions include unlabeled reports,
# they will not enter the calculation of precision
# and recall. However, if they are inspected, we
# may find that they are, in effect, frauds and so
# the real values of precision and recall should
# be higher.

# Usually there is a tradeoff between precision and
# recall. For instance, it is quite easy to achieve
# 100% recall if all test cases are predicted as
# events. However, this strategy leads to low
# precision.

# In our case, what we really want is to maximize
# the use of resources invested in inspection acti-
# vities. Recall is really the objective...we want
# to be able to achieve 100% recall with the resources
# we have available.

##### Lift Charts and Precision/Recall Curves

# Precision/recall (PR) curves are visual represen-
# tations of the performance of a model in terms of
# the precision and recall statistics.

# The curves are obtained at different working
# points which can be a function of different cut-
# off limits on a ranking of the class of interest.

# In our case this would be the different effort
# limits applied to the outlier ranking provided
# by the models. We can iterate over different
# limits (that is, inspect more/less reports) and
# see different values of precision and recall.

# R package ROCR has functions useful for classifying
# binary classifiers. ROCR implements many evaluation
# metrics and include methods to obtain a wide
# range of curves.

# We begin by obtaining an object of class prediction
# using predictions of the model and the true values
# of the test set. The resulting object is passed as
# an argument to function performance() to obtain
# evaluation metrics. Then the result of performance()
# is called with plot() to render different perfor-
# mance curves.

# Here we illustrate using example data in ROCR
# package:

install.packages("ROCR")
library("ROCR")
# example data in the ROCR package:
data(ROCR.simple)
# get predictions from the example model
pred <- prediction( ROCR.simple$predictions, 
                    ROCR.simple$labels )
# pass predictions to performance()
perf <- performance(pred,'prec','rec')
# and then plot PR curve
plot(perf)

# PR curves produced by ROCR have a sawtooth
# shape...there are methods to overcome this.
# One is to calculate the interpolated precision
# Prec(int) for a certain recall level r as the
# highest precision value found for any recall
# level greater than or equal to r.

# We take a close look at the object returned
# by performance() we see a slot 'y.values' with
# the y-axis values, the precision values.

# We can obtain a PR curve without 'sawteeth'
# by substituting the values of the inter-
# polated precision. Here is a function for
# implementing this idea in the general case:

PRcurve <- function(preds,trues,...) {
  require(ROCR,quietly=T)
  pd <- prediction(preds,trues)
  pf <- performance(pd,'prec','rec')
  # uses lapply() against list of y.values
  # which might contain several iterations
  # of an experimental process (we use later).
  pf@y.values <- lapply(pf@y.values,
                        function(x) rev(cummax(rev(x))))
  plot(pf,...)
}

# In function above we calculate interpolated
# precision using cummax() and rev(). cummax()
# obtains cumulative maximum of a set of numbers.

# We apply PRcurve() to example data:

PRcurve(ROCR.simple$predictions, 
        ROCR.simple$labels)

# First curve was a non-smoothed PR curve.
# The above is a smoothed PR curve.

# How do we evaluate our outlier ranking models
# with these curves? We have a test set with a
# variable Insp with possible values of unkn,
# ok, and fraud. And we have a ranking of the
# observations in this set, produced by a model.

# Our models obtain an outlier score for each
# observation in the test set between 0 and 1.

# If we order the test set observations by 
# decreasing outlier score, we can calculate
# different values of precision and recall,
# depending on where we put our inspection
# effort limit. Setting this limit is the same
# as selecting a threshold on the outlier score
# above which we will consider the observations
# as fraudulent.

# Suppose we have seven test cases with values
# {ok,ok,fraud,unk,fraud,fraud,unk} in Insp.

# Imagine our model produces outlier scores
# for these observations which are
# {0.2,0.1,0.7,0.5,0.4,0.3,0.25}

# When we rank the observations we end up with
# {fraud,unk,fraud,fraud,unk,ok,ok}

# If our resources limit us to only inspecting the
# top two {0.7,0.5} and sets the threshold above
# which we determine them to be outliers (fraud),
# then this is equivalent to a model "predicting"
# {fraud,fraud,ok,ok,ok,ok,ok} with the true values
# {fraud,unk,fraud,fraud,unk,ok,ok}. This gives
# us the following confusion matrix:

# Confusion Matrix (unk are considered ok)
#                  Predictions
#                  ok   fraud
#   True      ok    3      1    4
#   Values  fraud   2      1    3
#                   5      2    7

# The calculated values of precision and recall
# are prec = 1/(1+1)=0.5; recall=1/(2+1)=0.3333;

#### Lift Charts

# Lift Charts provide a different perspective on
# model predictions, emphasizing recall, and thus
# are probably more useful for our purposes.

# x-axis is the rate of positive predictions (RPP),
# or the probability that the model predicts a
# positive class which in the confusion matrix is
# (1+1)/7. The y-axis of lift charts is the value
# of recall divided by the value of RPP.

# This lift chart corresponds to the previous
# non-smoothed precision/recall curve.

pred <- prediction( ROCR.simple$predictions, 
                    ROCR.simple$labels )
perf <- performance(pred,'lift','rpp')
plot(perf,main='Lift Chart')

# An even more interesting plot might be one that
# shows the recall values in terms of the inspec-
# tion effort that is captured by the RPP. We call
# this type of graph the 'cumulative recall chart'.

# We can implement with the help of (this function
# is in the DMwR package):
CRchart <- function(preds,trues,...) {
  require(ROCR,quietly=T)
  pd <- prediction(preds,trues)
  pf <- performance(pd,'rec','rpp')
  plot(pf,...)
}  

# We call the CRchart:
CRchart(ROCR.simple$predictions, 
        ROCR.simple$labels, 
        main='Cumulative Recall Chart')

# The nearer the curve of a model to the top-left
# corner of the graph, the better.

### Normalized Distance to Typical Price

# Measures so far have only evaluated the quality
# of the rankings in terms of the labeled reports.
# So they are supervised evaluation metrics.

# What about the unlabeled cases (which we have
# been assuming are not-fraud, or ok by default).

# We might compare their unit price with the typical
# price of the reports of the same product. We would
# expect a higher difference to be an indication
# that something is wrong.

# So we can use this difference, or 'distance', as
# a good indicator of the quality of the outlier
# ranking obtained by the model.

# To nullify price scaling effects on our measure
# of outlier ranking quality, we normalize the
# distance to the typical unit price using IQR:
#
# NDTPp(u) = |u-Up| / IQRp
#
# where:
# Up is the typical unit price of product p as
# measured by the median unit price of its trans.
#
# IQRp is the inter-quartile range of the unit
# prices of that product.

# In our experiments we will use the average
# value of NDTPp as one of the evaluation metrics
# characterizing the performance of our models.

# Here is a function the calculates the value
# of this statistic:

# toInsp is set of transactions to inspect. The
# function must also receive either the training
# set (to calculate median and IQR of each product)
# or a data structure with this information.
# This is not a DMwR function.
avgNDTP <- function(toInsp,train,stats) {
  if (missing(train) && missing(stats)) 
    stop('Provide either the training data or the product stats')
  # If training set is received it calculates
  # median and IQR values of non-fraud transactions
  # of each product in training set:
  if (missing(stats)) {
    # these are not fraudulent transactions:
    notF <- which(train$Insp != 'fraud')
    # calculates median and IQR:
    stats <- tapply(train$Uprice[notF],
                    list(Prod=train$Prod[notF]),
                    function(x) {
                      bp <- boxplot.stats(x)$stats
                      c(median=bp[3],
                        iqr=bp[4]-bp[2])
                    })
    stats <- matrix(unlist(stats),
                    length(stats),2,byrow=T,
                    dimnames=list(names(stats),
                                  c('median',
                                    'iqr')))
    # If IQR is zero, uses median to avoid
    # dividing by zero in calculating NDTPp
    stats[which(stats[,'iqr']==0),'iqr'] <- 
      stats[which(stats[,'iqr']==0),'median']
  }
  
  mdtp <- mean(abs(toInsp$Uprice-stats[toInsp$Prod,'median']) /
                 stats[toInsp$Prod,'iqr'])
  return(mdtp)
}

###################################################
### Experimental Methodology
###################################################

# We have a large dataset so can use Hold Out
# method for experimental comparisons. H.O. splits
# the dataset typically into 70% and 30% partitions
# one to train models, and other to test them.

# Imbalance of different labels, particularly ok
# and fraud is also an issue. The re-sampling may
# easily produce a test set with different distr
# of ok / fraud reports --> use a stratified
# sampling strategy --> random samples from bags
# of the observations of different classes in the
# correct proportions.

# holdOut() function in DMwR conducts hold out
# experiments but we need precision, recall and
# average NDTP which is calculated with a user
# defined function:

# Must provide test set, ranking proposed by the
# model for this set, threshold of inspection
# limit effort and stats (median and IQR) of
# products. We use this function soon.
evalOutlierRanking <- function(testSet,
                               rankOrder,
                               Threshold,
                               statsProds) {
  # ranks the test set
  ordTS <- testSet[rankOrder,]
  # counts rows in test set
  N <- nrow(testSet)
  nF <- if (Threshold < 1) as.integer(Threshold*N) else Threshold
  cm <- table(c(rep('fraud',nF),
                rep('ok',N-nF)),
              ordTS$Insp)
  # calculates precision, recall, avgNDTP:
  prec <- cm['fraud','fraud']/sum(cm['fraud',])
  rec <- cm['fraud','fraud']/sum(cm[,'fraud'])
  AVGndtp <- avgNDTP(ordTS[nF,],
                     stats=statsProds)
  return(c(Precision=prec,Recall=rec,
           avgNDTP=AVGndtp))
}

##### OBTAINING OUTLIER RANKINGS

# We use different models to obtain outlier rankings.

##### UNSUPERVISED APPROACHES

### The modified box plot rule

# Can detect outliers of a continuous distribution
# provided it follows a near-normal distribution,
# such as is true for unit price of product.

# Can use as baseline method to apply to our data.

# We use MBPR on transactions for each product in
# a given test set. But we must still decide how
# to move into an outlier ranking of all test sets.

BPrule <- function(train,test) {
  notF <- which(train$Insp != 'fraud')
  ms <- tapply(train$Uprice[notF],list(Prod=train$Prod[notF]),
               function(x) {
                 bp <- boxplot.stats(x)$stats
                 c(median=bp[3],iqr=bp[4]-bp[2])
               })
  ms <- matrix(unlist(ms),length(ms),2,byrow=T,
               dimnames=list(names(ms),c('median','iqr')))
  ms[which(ms[,'iqr']==0),'iqr'] <- ms[which(ms[,'iqr']==0),'median']
  ORscore <- abs(test$Uprice-ms[test$Prod,'median']) /
    ms[test$Prod,'iqr']
  return(list(rankOrder=order(ORscore,decreasing=T),
              rankScore=ORscore))
}


notF <- which(sales$Insp != 'fraud')
globalStats <- tapply(sales$Uprice[notF],
                      list(Prod=sales$Prod[notF]),
                      function(x) {
                        bp <- boxplot.stats(x)$stats
                        c(median=bp[3],iqr=bp[4]-bp[2])
                      })
globalStats <- matrix(unlist(globalStats),
                      length(globalStats),2,byrow=T,
                      dimnames=list(names(globalStats),c('median','iqr')))
globalStats[which(globalStats[,'iqr']==0),'iqr'] <- 
  globalStats[which(globalStats[,'iqr']==0),'median']


ho.BPrule <- function(form, train, test, ...) {
  res <- BPrule(train,test)
  structure(evalOutlierRanking(test,res$rankOrder,...),
            itInfo=list(preds=res$rankScore,
                        trues=ifelse(test$Insp=='fraud',1,0)
            )
  )
}


bp.res <- holdOut(learner('ho.BPrule',
                          pars=list(Threshold=0.1,
                                    statsProds=globalStats)),
                  dataset(Insp ~ .,sales),
                  hldSettings(3,0.3,1234,T),
                  itsInfo=TRUE
)


summary(bp.res)


par(mfrow=c(1,2))
info <- attr(bp.res,'itsInfo')
PTs.bp <- aperm(array(unlist(info),dim=c(length(info[[1]]),2,3)),
                c(1,3,2)
)
PRcurve(PTs.bp[,,1],PTs.bp[,,2],
        main='PR curve',avg='vertical')
CRchart(PTs.bp[,,1],PTs.bp[,,2],
        main='Cumulative Recall curve',avg='vertical')




###################################################
### Local outlier factors (LOF)
###################################################
ho.LOF <- function(form, train, test, k, ...) {
  require(dprep,quietly=T)
  ntr <- nrow(train)
  all <- rbind(train,test)
  N <- nrow(all)
  ups <- split(all$Uprice,all$Prod)
  r <- list(length=ups)
  for(u in seq(along=ups)) 
    r[[u]] <- if (NROW(ups[[u]]) > 3) 
      lofactor(ups[[u]],min(k,NROW(ups[[u]]) %/% 2)) 
  else if (NROW(ups[[u]])) rep(0,NROW(ups[[u]])) 
  else NULL
  all$lof <- vector(length=N)
  split(all$lof,all$Prod) <- r
  all$lof[which(!(is.infinite(all$lof) | is.nan(all$lof)))] <- 
    SoftMax(all$lof[which(!(is.infinite(all$lof) | is.nan(all$lof)))])
  structure(evalOutlierRanking(test,order(all[(ntr+1):N,'lof'],
                                          decreasing=T),...),
            itInfo=list(preds=all[(ntr+1):N,'lof'],
                        trues=ifelse(test$Insp=='fraud',1,0))
  )
}


lof.res <- holdOut(learner('ho.LOF',
                           pars=list(k=7,Threshold=0.1,
                                     statsProds=globalStats)),
                   dataset(Insp ~ .,sales),
                   hldSettings(3,0.3,1234,T),
                   itsInfo=TRUE
)


summary(lof.res)


par(mfrow=c(1,2))
info <- attr(lof.res,'itsInfo')
PTs.lof <- aperm(array(unlist(info),dim=c(length(info[[1]]),2,3)),
                 c(1,3,2)
)
PRcurve(PTs.bp[,,1],PTs.bp[,,2],
        main='PR curve',lty=1,xlim=c(0,1),ylim=c(0,1),
        avg='vertical')
PRcurve(PTs.lof[,,1],PTs.lof[,,2],
        add=T,lty=2,
        avg='vertical')
legend('topright',c('BPrule','LOF'),lty=c(1,2))
CRchart(PTs.bp[,,1],PTs.bp[,,2],
        main='Cumulative Recall curve',lty=1,xlim=c(0,1),ylim=c(0,1),
        avg='vertical')
CRchart(PTs.lof[,,1],PTs.lof[,,2],
        add=T,lty=2,
        avg='vertical')
legend('bottomright',c('BPrule','LOF'),lty=c(1,2))



###################################################
### Clustering-based outlier rankings (OR_h)
###################################################
ho.ORh <- function(form, train, test, ...) {
  require(dprep,quietly=T)
  ntr <- nrow(train)
  all <- rbind(train,test)
  N <- nrow(all)
  ups <- split(all$Uprice,all$Prod)
  r <- list(length=ups)
  for(u in seq(along=ups)) 
    r[[u]] <- if (NROW(ups[[u]]) > 3) 
      outliers.ranking(ups[[u]])$prob.outliers 
  else if (NROW(ups[[u]])) rep(0,NROW(ups[[u]])) 
  else NULL
  all$lof <- vector(length=N)
  split(all$lof,all$Prod) <- r
  all$lof[which(!(is.infinite(all$lof) | is.nan(all$lof)))] <- 
    SoftMax(all$lof[which(!(is.infinite(all$lof) | is.nan(all$lof)))])
  structure(evalOutlierRanking(test,order(all[(ntr+1):N,'lof'],
                                          decreasing=T),...),
            itInfo=list(preds=all[(ntr+1):N,'lof'],
                        trues=ifelse(test$Insp=='fraud',1,0))
  )
}


orh.res <- holdOut(learner('ho.ORh',
                           pars=list(Threshold=0.1,
                                     statsProds=globalStats)),
                   dataset(Insp ~ .,sales),
                   hldSettings(3,0.3,1234,T),
                   itsInfo=TRUE
)


summary(orh.res)


par(mfrow=c(1,2))
info <- attr(orh.res,'itsInfo')
PTs.orh <- aperm(array(unlist(info),dim=c(length(info[[1]]),2,3)),
                 c(1,3,2)
)
PRcurve(PTs.bp[,,1],PTs.bp[,,2],
        main='PR curve',lty=1,xlim=c(0,1),ylim=c(0,1),
        avg='vertical')
PRcurve(PTs.lof[,,1],PTs.lof[,,2],
        add=T,lty=2,
        avg='vertical')
PRcurve(PTs.orh[,,1],PTs.orh[,,2],
        add=T,lty=1,col='grey',
        avg='vertical')        
legend('topright',c('BPrule','LOF','ORh'),
       lty=c(1,2,1),col=c('black','black','grey'))
CRchart(PTs.bp[,,1],PTs.bp[,,2],
        main='Cumulative Recall curve',lty=1,xlim=c(0,1),ylim=c(0,1),
        avg='vertical')
CRchart(PTs.lof[,,1],PTs.lof[,,2],
        add=T,lty=2,
        avg='vertical')
CRchart(PTs.orh[,,1],PTs.orh[,,2],
        add=T,lty=1,col='grey',
        avg='vertical')
legend('bottomright',c('BPrule','LOF','ORh'),
       lty=c(1,2,1),col=c('black','black','grey'))


###################################################
### The class imbalance problem
###################################################
data(iris)
data <- iris[,c(1,2,5)]
data$Species <- factor(ifelse(data$Species == 'setosa','rare','common'))
newData <- SMOTE(Species ~ .,data,perc.over=600)
table(newData$Species)


par(mfrow=c(1,2))
plot(data[,1],data[,2],pch=19+as.integer(data[,3]),main='Original Data')
plot(newData[,1],newData[,2],pch=19+as.integer(newData[,3]),main="SMOTE'd Data")



###################################################
### Naive Bayes
###################################################
nb <- function(train,test) {
  require(e1071,quietly=T)
  sup <- which(train$Insp != 'unkn')
  data <- train[sup,c('ID','Prod','Uprice','Insp')]
  data$Insp <- factor(data$Insp,levels=c('ok','fraud'))
  model <- naiveBayes(Insp ~ .,data)
  preds <- predict(model,test[,c('ID','Prod','Uprice','Insp')],type='raw')
  return(list(rankOrder=order(preds[,'fraud'],decreasing=T),
              rankScore=preds[,'fraud'])
  )
}


ho.nb <- function(form, train, test, ...) {
  res <- nb(train,test)
  structure(evalOutlierRanking(test,res$rankOrder,...),
            itInfo=list(preds=res$rankScore,
                        trues=ifelse(test$Insp=='fraud',1,0)
            )
  )
}


nb.res <- holdOut(learner('ho.nb',
                          pars=list(Threshold=0.1,
                                    statsProds=globalStats)),
                  dataset(Insp ~ .,sales),
                  hldSettings(3,0.3,1234,T),
                  itsInfo=TRUE
)


summary(nb.res)


par(mfrow=c(1,2))
info <- attr(nb.res,'itsInfo')
PTs.nb <- aperm(array(unlist(info),dim=c(length(info[[1]]),2,3)),
                c(1,3,2)
)
PRcurve(PTs.nb[,,1],PTs.nb[,,2],
        main='PR curve',lty=1,xlim=c(0,1),ylim=c(0,1),
        avg='vertical')
PRcurve(PTs.orh[,,1],PTs.orh[,,2],
        add=T,lty=1,col='grey',
        avg='vertical')        
legend('topright',c('NaiveBayes','ORh'),
       lty=1,col=c('black','grey'))
CRchart(PTs.nb[,,1],PTs.nb[,,2],
        main='Cumulative Recall curve',lty=1,xlim=c(0,1),ylim=c(0,1),
        avg='vertical')
CRchart(PTs.orh[,,1],PTs.orh[,,2],
        add=T,lty=1,col='grey',
        avg='vertical')        
legend('bottomright',c('NaiveBayes','ORh'),
       lty=1,col=c('black','grey'))


nb.s <- function(train,test) {
  require(e1071,quietly=T)
  sup <- which(train$Insp != 'unkn')
  data <- train[sup,c('ID','Prod','Uprice','Insp')]
  data$Insp <- factor(data$Insp,levels=c('ok','fraud'))
  newData <- SMOTE(Insp ~ .,data,perc.over=700)
  model <- naiveBayes(Insp ~ .,newData)
  preds <- predict(model,test[,c('ID','Prod','Uprice','Insp')],type='raw')
  return(list(rankOrder=order(preds[,'fraud'],decreasing=T),
              rankScore=preds[,'fraud'])
  )
}


ho.nbs <- function(form, train, test, ...) {
  res <- nb.s(train,test)
  structure(evalOutlierRanking(test,res$rankOrder,...),
            itInfo=list(preds=res$rankScore,
                        trues=ifelse(test$Insp=='fraud',1,0)
            )
  )
}


nbs.res <- holdOut(learner('ho.nbs',
                           pars=list(Threshold=0.1,
                                     statsProds=globalStats)),
                   dataset(Insp ~ .,sales),
                   hldSettings(3,0.3,1234,T),
                   itsInfo=TRUE
)


summary(nbs.res)


par(mfrow=c(1,2))
info <- attr(nbs.res,'itsInfo')
PTs.nbs <- aperm(array(unlist(info),dim=c(length(info[[1]]),2,3)),
                 c(1,3,2)
)
PRcurve(PTs.nb[,,1],PTs.nb[,,2],
        main='PR curve',lty=1,xlim=c(0,1),ylim=c(0,1),
        avg='vertical')
PRcurve(PTs.nbs[,,1],PTs.nbs[,,2],
        add=T,lty=2,
        avg='vertical')
PRcurve(PTs.orh[,,1],PTs.orh[,,2],
        add=T,lty=1,col='grey',
        avg='vertical')        
legend('topright',c('NaiveBayes','smoteNaiveBayes','ORh'),
       lty=c(1,2,1),col=c('black','black','grey'))
CRchart(PTs.nb[,,1],PTs.nb[,,2],
        main='Cumulative Recall curve',lty=1,xlim=c(0,1),ylim=c(0,1),
        avg='vertical')
CRchart(PTs.nbs[,,1],PTs.nbs[,,2],
        add=T,lty=2,
        avg='vertical')
CRchart(PTs.orh[,,1],PTs.orh[,,2],
        add=T,lty=1,col='grey',
        avg='vertical')        
legend('bottomright',c('NaiveBayes','smoteNaiveBayes','ORh'),
       lty=c(1,2,1),col=c('black','black','grey'))



###################################################
### AdaBoost
###################################################
library(RWeka)
WOW(AdaBoostM1)


data(iris)
idx <- sample(150,100)
model <- AdaBoostM1(Species ~ .,iris[idx,],
                    control=Weka_control(I=100))
preds <- predict(model,iris[-idx,])
head(preds)
table(preds,iris[-idx,'Species'])
prob.preds <- predict(model,iris[-idx,],type='probability')
head(prob.preds)


ab <- function(train,test) {
  require(RWeka,quietly=T)
  sup <- which(train$Insp != 'unkn')
  data <- train[sup,c('ID','Prod','Uprice','Insp')]
  data$Insp <- factor(data$Insp,levels=c('ok','fraud'))
  model <- AdaBoostM1(Insp ~ .,data,
                      control=Weka_control(I=100))
  preds <- predict(model,test[,c('ID','Prod','Uprice','Insp')],
                   type='probability')
  return(list(rankOrder=order(preds[,'fraud'],decreasing=T),
              rankScore=preds[,'fraud'])
  )
}


ho.ab <- function(form, train, test, ...) {
  res <- ab(train,test)
  structure(evalOutlierRanking(test,res$rankOrder,...),
            itInfo=list(preds=res$rankScore,
                        trues=ifelse(test$Insp=='fraud',1,0)
            )
  )
}


ab.res <- holdOut(learner('ho.ab',
                          pars=list(Threshold=0.1,
                                    statsProds=globalStats)),
                  dataset(Insp ~ .,sales),
                  hldSettings(3,0.3,1234,T),
                  itsInfo=TRUE
)


summary(ab.res)


par(mfrow=c(1,2))
info <- attr(ab.res,'itsInfo')
PTs.ab <- aperm(array(unlist(info),dim=c(length(info[[1]]),2,3)),
                c(1,3,2)
)
PRcurve(PTs.nb[,,1],PTs.nb[,,2],
        main='PR curve',lty=1,xlim=c(0,1),ylim=c(0,1),
        avg='vertical')
PRcurve(PTs.orh[,,1],PTs.orh[,,2],
        add=T,lty=1,col='grey',
        avg='vertical')        
PRcurve(PTs.ab[,,1],PTs.ab[,,2],
        add=T,lty=2,
        avg='vertical')        
legend('topright',c('NaiveBayes','ORh','AdaBoostM1'),
       lty=c(1,1,2),col=c('black','grey','black'))
CRchart(PTs.nb[,,1],PTs.nb[,,2],
        main='PR curve',lty=1,xlim=c(0,1),ylim=c(0,1),
        avg='vertical')
CRchart(PTs.orh[,,1],PTs.orh[,,2],
        add=T,lty=1,col='grey',
        avg='vertical')        
CRchart(PTs.ab[,,1],PTs.ab[,,2],
        add=T,lty=2,
        avg='vertical')        
legend('bottomright',c('NaiveBayes','ORh','AdaBoostM1'),
       lty=c(1,1,2),col=c('black','grey','black'))



###################################################
### Semi-supervised approaches
###################################################
set.seed(1234) # Just to ensrure you get the same results as in the book
library(DMwR)
library(e1071)
data(iris)
idx <- sample(150,100)
tr <- iris[idx,]
ts <- iris[-idx,]
nb <- naiveBayes(Species ~ .,tr)
table(predict(nb,ts),ts$Species)
trST <- tr
nas <- sample(100,90)
trST[nas,'Species'] <- NA
func <- function(m,d) {
  p <- predict(m,d,type='raw')
  data.frame(cl=colnames(p)[apply(p,1,which.max)],p=apply(p,1,max))
}
nbSTbase <- naiveBayes(Species ~ .,trST[-nas,])
table(predict(nbSTbase,ts),ts$Species)
nbST <- SelfTrain(Species ~ .,trST,learner('naiveBayes',list()),'func')
table(predict(nbST,ts),ts$Species)


pred.nb <- function(m,d) {
  p <- predict(m,d,type='raw')
  data.frame(cl=colnames(p)[apply(p,1,which.max)],
             p=apply(p,1,max)
  )
}
nb.st <- function(train,test) {
  require(e1071,quietly=T)
  train <- train[,c('ID','Prod','Uprice','Insp')]
  train[which(train$Insp == 'unkn'),'Insp'] <- NA
  train$Insp <- factor(train$Insp,levels=c('ok','fraud'))
  model <- SelfTrain(Insp ~ .,train,
                     learner('naiveBayes',list()),'pred.nb')
  preds <- predict(model,test[,c('ID','Prod','Uprice','Insp')],
                   type='raw')
  return(list(rankOrder=order(preds[,'fraud'],decreasing=T),
              rankScore=preds[,'fraud'])
  )
}
ho.nb.st <- function(form, train, test, ...) {
  res <- nb.st(train,test)
  structure(evalOutlierRanking(test,res$rankOrder,...),
            itInfo=list(preds=res$rankScore,
                        trues=ifelse(test$Insp=='fraud',1,0)
            )
  )
}


nb.st.res <- holdOut(learner('ho.nb.st',
                             pars=list(Threshold=0.1,
                                       statsProds=globalStats)),
                     dataset(Insp ~ .,sales),
                     hldSettings(3,0.3,1234,T),
                     itsInfo=TRUE
)


summary(nb.st.res)


par(mfrow=c(1,2))
info <- attr(nb.st.res,'itsInfo')
PTs.nb.st <- aperm(array(unlist(info),dim=c(length(info[[1]]),2,3)),
                   c(1,3,2)
)
PRcurve(PTs.nb[,,1],PTs.nb[,,2],
        main='PR curve',lty=1,xlim=c(0,1),ylim=c(0,1),
        avg='vertical')
PRcurve(PTs.orh[,,1],PTs.orh[,,2],
        add=T,lty=1,col='grey',
        avg='vertical')        
PRcurve(PTs.nb.st[,,1],PTs.nb.st[,,2],
        add=T,lty=2,
        avg='vertical')        
legend('topright',c('NaiveBayes','ORh','NaiveBayes-ST'),
       lty=c(1,1,2),col=c('black','grey','black'))
CRchart(PTs.nb[,,1],PTs.nb[,,2],
        main='Cumulative Recall curve',lty=1,xlim=c(0,1),ylim=c(0,1),
        avg='vertical')
CRchart(PTs.orh[,,1],PTs.orh[,,2],
        add=T,lty=1,col='grey',
        avg='vertical')        
CRchart(PTs.nb.st[,,1],PTs.nb.st[,,2],
        add=T,lty=2,
        avg='vertical')        
legend('bottomright',c('NaiveBayes','ORh','NaiveBayes-ST'),
       lty=c(1,1,2),col=c('black','grey','black'))


pred.ada <- function(m,d) {
  p <- predict(m,d,type='probability')
  data.frame(cl=colnames(p)[apply(p,1,which.max)],
             p=apply(p,1,max)
  )
}
ab.st <- function(train,test) {
  require(RWeka,quietly=T)
  train <- train[,c('ID','Prod','Uprice','Insp')]
  train[which(train$Insp == 'unkn'),'Insp'] <- NA
  train$Insp <- factor(train$Insp,levels=c('ok','fraud'))
  model <- SelfTrain(Insp ~ .,train,
                     learner('AdaBoostM1',
                             list(control=Weka_control(I=100))),
                     'pred.ada')
  preds <- predict(model,test[,c('ID','Prod','Uprice','Insp')],
                   type='probability')
  return(list(rankOrder=order(preds[,'fraud'],decreasing=T),
              rankScore=preds[,'fraud'])
  )
}
ho.ab.st <- function(form, train, test, ...) {
  res <- ab.st(train,test)
  structure(evalOutlierRanking(test,res$rankOrder,...),
            itInfo=list(preds=res$rankScore,
                        trues=ifelse(test$Insp=='fraud',1,0)
            )
  )
}
ab.st.res <- holdOut(learner('ho.ab.st',
                             pars=list(Threshold=0.1,
                                       statsProds=globalStats)),
                     dataset(Insp ~ .,sales),
                     hldSettings(3,0.3,1234,T),
                     itsInfo=TRUE
)


summary(ab.st.res)


par(mfrow=c(1,2))
info <- attr(ab.st.res,'itsInfo')
PTs.ab.st <- aperm(array(unlist(info),dim=c(length(info[[1]]),2,3)),
                   c(1,3,2)
)
PRcurve(PTs.ab[,,1],PTs.ab[,,2],
        main='PR curve',lty=1,xlim=c(0,1),ylim=c(0,1),
        avg='vertical')
PRcurve(PTs.orh[,,1],PTs.orh[,,2],
        add=T,lty=1,col='grey',
        avg='vertical')        
PRcurve(PTs.ab.st[,,1],PTs.ab.st[,,2],
        add=T,lty=2,
        avg='vertical')        
legend('topright',c('AdaBoostM1','ORh','AdaBoostM1-ST'),
       lty=c(1,1,2),col=c('black','grey','black'))
CRchart(PTs.ab[,,1],PTs.ab[,,2],
        main='Cumulative Recall curve',lty=1,xlim=c(0,1),ylim=c(0,1),
        avg='vertical')
CRchart(PTs.orh[,,1],PTs.orh[,,2],
        add=T,lty=1,col='grey',
        avg='vertical')        
CRchart(PTs.ab.st[,,1],PTs.ab.st[,,2],
        add=T,lty=2,
        avg='vertical')        
legend('bottomright',c('AdaBoostM1','ORh','AdaBoostM1-ST'),
       lty=c(1,1,2),col=c('black','grey','black'))


