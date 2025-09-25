#####################################################
### Regression Trees and Random Forests Exercises ###
#####################################################

# There is more than one way to build decision trees
# (regression trees) using R. In this exercise, we
# will build decision trees with package party. 
# Then we will look at an example of training
# a random forest model using package 
# randomForest.

### Decision Trees with Package party

# We will build a conditional decision tree for
# the iris data using function ctree() in
# package party.

# Look at the data:
data(iris)

# We predict the species of iris flowers with
# four numeric predictor variables. The target
# variable, species, is a factor with three
# levels "setosa", "versicolor", and "virginica":
str(iris)

# Note we are using the term 'predict' but this
# is actually a classification task. The function
# ctree() builds a decision tree and predict()
# makes predictions for new data. This is different
# than modeling trees with a1 in the algae data
# which was a numeric variable.

# We start by splitting the iris data into two
# subsets: training (70%) and test (30%). We 
# use a random seed so results are reproducible.

set.seed(1234)

# obtain sets for training and for test:
set <- sample(2, nrow(iris), 
              # we sample with replacement:
              replace=TRUE, 
              # two sets with these 
              # approximate proportions:
              prob=c(0.7, 0.3))

# fixed training set:
trainData <- iris[set==1,]

# Note '==' is equality operator
# whereas '=' is assignment operator

# fixed test data:
testData <- iris[set==2,]

# We load package party, build a decision tree,
# and check the prediction result:
install.packages("party")

# Load the party package:
library("party") # note numerous dependent packages

?ctree
  
# build the tree using all predictors:
iris.ctree <- ctree(Species~., data=trainData)

# check the predictions, rows are predicted,
# columns are actual. Interpret the outputted
# table in plain English:
table(predict(iris.ctree), trainData$Species)

# We look at the tree by printing the rules
print(iris.ctree)

# We plot the tree
plot(iris.ctree)

# plot a 'simple' tree
plot(iris.ctree, type="simple")

# Interpret, that is, explain, what information
# each of these trees convey.

# Now the built tree needs to be tested using
# the test data

# predict on test data. Note we do not have to
# re-generate the tree. That tree is our predictive
# model, we simply apply it to a new data set:
testPred <- predict(iris.ctree, newdata=testData)

# tabulate and interpret the predictions
# versus the actual results
table(testPred, testData$Species)

##### Random Forest

# Now we use package randomForest to build a 
# predictive model for the iris dataset. 

# We split iris data into two subsets: training (70%)
# and test (30%). We do not set seed.

tree.set <- sample(2, nrow(iris), 
                   replace=TRUE,
                   prob=c(0.7,0.3))

trainData <- iris[tree.set==1,]

testData <- iris[tree.set==2,]

# Load package randomForest and train a random
# forest. We do not set seed.

# What do the arguments below accomplish
# for our random forest model?
?randomForest

# might need to install
# install.packages("randomForest")
library(randomForest)
rf <- randomForest(Species~., data=trainData,
                   ntree=100, proximity=TRUE)

# what class is rf?
class(rf)

# Look at the tabulated predictions versus
# actual and interpret
table(predict(rf), trainData$Species)

# 'print' the results, can you interpret?:
print(rf)

# Look at the 'make-up' of the fitted rf model:
attributes(rf)  # note it is a list w 2 components

# Plot the error rates using different numbers
# of trees. Interpret. How many trees do the
# model require to optimize (lowest) the errors?
# Note: each dotted line corresponds to one level
# of iris species. The solid black line is the
# average.
plot(rf)

?plot.randomForest

# Now we perform these same activities
# using the test data. Interpret the findings:
irisPred <- predict(rf, newdata=testData)
table(irisPred, testData$Species)
