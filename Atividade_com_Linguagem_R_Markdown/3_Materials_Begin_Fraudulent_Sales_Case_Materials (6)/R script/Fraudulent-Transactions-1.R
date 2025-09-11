################################################
#####   Detecting Faudulent Transactions   #####
################################################

# The domain is sales data. We want to find 
# "strange" transaction reports that may be
# fraudulent.

# We want to provide a "fraud probability ranking"
# for events that has already occurred.

# We will look at several new data mining tasks
# including: (1) outlier or anomaly detection;
# (2) clustering; and (3) semi-supervised predic-
# tion modeling.

# Data is transactions reported by salespeople.
# They sell products and report these sales with
# a regular periodicity.

# Will need to load DMwR package
library(DMwR)

# Then can load sales data:
data(sales)

# Take a look:
head(sales)

# How many rows?:
nrow(sales)

# How is data structured?:
str(sales)

# ID - a factor, ID of salesman

# Prod - a factor, ID of the sold product

# Quant - number of reported sold units of product

# Val - reported total monetary value of the sale

# Insp - Factor, 3 possible values: (1) ok, if 
# transaction inspected and determined to be
# valid; (2) fraud, is found to be fraudulent;
# and (3) unk if unknown (not inspected yet)

##########################################
######    Exploring the data set    ######
##########################################

# Overview of statistical properties of data:
summary(sales)

# We see that: 
# there are large numbers of unknown values in
# columns Quant and Val. This would be a problem
# if they are NA is same record as the transac-
# tion report would be missing crucial info
# about the sale

?nlevels # shows number of levels of a factor

# have bunches of salespeople
nlevels(sales$ID)

# and lots of unique products:
nlevels(sales$Prod)

# We check if Quant and Val are both missing
# together much (length returns number 888)
length(which(is.na(sales$Quant) & is.na(sales$Val)))

# With large data sets, is likely easier to
# simply perform "logical arithmetic" where
# TRUE=1 and FALSE=0:
sum(is.na(sales$Quant) & is.na(sales$Val))

# From summary() results, look at distribution
# of values in inspection column...proportion
# of frauds is relatively low, even if we only
# take into account the reports that were
# inspected (about 0.003166)
table(sales$Insp)/nrow(sales)*100

# We look at number of transactions per
# salesperson. There is much variability, also
# if we look at number of transactions per 
# product

# Set up table for counts of transactions
# per salesperson:
(totS <- table(sales$ID))

# Then table of counts of transactions
# per product:
(totP <- table(sales$Prod))

# We can see this variability in plots;
# Plot of transactions per salesperson:
barplot(totS,
        main='Transactions per salesperson',
        names.arg='',xlab='Salespeople',
        ylab='Amount')

# Plot of transactions per product:
barplot(totP,
        main='Transactions per product',
        names.arg='',xlab='Products',
        ylab='Amount')

# Variables Quant and Val show a lot of
# variability also, indicating differences
# in the products, thus, they might be
# better handled separately.

# If prices of the products are very different
# it may only be possible to identify abnor-
# mal transactions in the context of the same
# product.

# However, given the disparate quantity of
# products that are sold on each transaction,
# it might make more sense to carry out this
# analysis over the unit price instead.

# We add this derived unit price per transac-
# tion as a new column to the data frame:
sales$Uprice <- sales$Val/sales$Quant

# Unit price should be relatively constant
# over the transactions of the same product.

# When analyzing transactions over a short
# period of time, one does not expect strong
# variations of the unit price of the
# products.

# We check the distribution of the unit price:
summary(sales$Uprice)

# We again observed a marked variability.

# Given this observation, we should analyze
# the set of transactions on each product 
# individually, looking for suspicious tran-
# sactions on each of these sets.

# One problem is that some products have
# very few transactions. Of the 4,548 products,
# 982 have fewer than 20 transactions.

# Declaring a report as unusual based on a
# sample of fewer than 20 reports may be
# too risky.

# It may be interesting to check what the
# top most expensive and least expensive
# products are. We use median unit price
# to represent the typical price at which
# a product is sold:

# attaching makes variables directly
# accessible without use of 'sales$...'
attach(sales)

# We obtain median unit price of each product.
# aggregate() applies a function that produces
# some scalar value (median here) to subsets
# separated on some factor (ind. products)
upp <- aggregate(Uprice,list(Prod),median,na.rm=T)
# Let's take a look at upp
upp[1:10,]

# We generate five most (and least) expensive
# by varying the parameter 'decreasing' of the
# function order() using sapply().
topP <- sapply(c(T,F),function(o) 
               upp[order(upp[,2],
                         decreasing=o)[1:5],1])
colnames(topP) <- c('Expensive','Cheap')
topP

# We confirm the completely different price
# distribution of the top products using a box
# plot of their unit price.

# The %in% operator tests if a value
# belongs to a set.
tops <- sales[Prod %in% topP[1,],
              c('Prod','Uprice')]
tops$Prod <- factor(tops$Prod)
# The scales of the prices of the most expensive
# and least expensive products are rather different.
# So we use a log scale to keep the values of
# the cheap product from being indistinguishable.
# Y-axis is on loag scale.
boxplot(Uprice ~ Prod,data=tops,
        ylab='Uprice',log="y")

# We carry out a similar analysis to discover
# whcih salespeople are ones who bring more
# (less) money into the company.

vs <- aggregate(Val,list(ID),sum,na.rm=T)
scoresSs <- sapply(c(T,F),function(o) 
                   vs[order(vs$x,decreasing=o)[1:5],1])
colnames(scoresSs) <- c('Most','Least')
scoresSs

# The top 100 salespeople account for almost
# 40% of the company income, while the bottom
# 2,000 (of 6,016 salespeople) generate less than
# 2% of the income:

# Percent of company income top 100 salespeople:
sum(vs[order(vs$x,decreasing=T)[1:100],2])/sum(Val,na.rm=T)*100

# Percent of company income bottom 2,000:
sum(vs[order(vs$x,decreasing=F)[1:2000],2])/sum(Val,na.rm=T)*100

# If we carry out a similar analysis in terms of
# the quantity that is sold for each product, the
# results are even more unbalanced:
qs <- aggregate(Quant,list(Prod),sum,na.rm=T)
scoresPs <- sapply(c(T,F),function(o) 
                   qs[order(qs$x,decreasing=o)[1:5],1])
colnames(scoresPs) <- c('Most','Least')
scoresPs

# Top 100 products represent nearly 75% of sales volume:
sum(as.double(qs[order(qs$x,decreasing=T)[1:100],2]))/sum(as.double(Quant),na.rm=T)*100

# 4,000 of the 4,548 products account for less
# than 10% of the sales volume:
sum(as.double(qs[order(qs$x,decreasing=F)[1:4000],2]))/sum(as.double(Quant),na.rm=T)*100

# Salespeople can change the price of an item if
# they want, but we still assume that the unit price
# of any product should follow a near-normal distribution.

# We can conduct some basic tests to find deviations
# from this normality assumption:

# The Box-Plot Rule: Box-plots show outliers. The
# rule is that an observation should be tagged as
# an anomaly, a high (low) value if it is above
# (below) the high (low) whisker which is defined
# as Q3+(1.5 x IQR) for high and Q1-(1.5 x IQR)
# for the low values, where Q1 is the first quartile,
# Q3 is the third quartile, and IQR=(Q3-Q1) and
# is the inter-quartile range.

# This 'Box-Plot' Rule works well for normally-
# distributed variables, and is robust to the
# presence of a few outliers since it is based
# in robust statistics using quartiles.

# We determine the number of outliers (by above
# definition) of each product:
out <- tapply(Uprice,list(Prod=Prod),
              function(x) length(boxplot.stats(x)$out))

# boxplot.stats() function obtains statistics used
# in the construction of box plots. It returns a
# list and the 'out' component contains observations
# considered to be outliers.

# The products with more outliers are:
out[order(out,decreasing=T)[1:10]]

# We see that 29,446 transactions are outliers:
sum(out)

# which is approximately 7% of total transactions:
sum(out)/nrow(sales)*100

# We could have data problems, we already know that
# some of the transactions have been flagged as
# fraudulent.

###################################
######     Data problems     ######
###################################

# Data problems can be an obstacle to the techniques
# we will use later in this case.

### Unknown (Missing) Values

# 3 basic alternatives:
# 1) Remove them;
# 2) Fill them in with some strategy; or
# 3) Use tools that can handle them.

# We are most concerned with transactions that have
# both Quant and Val missing. Removing all such
# 888 cases could be a problem if it eliminates
# some product or salesperson, so we should
# check this.

# Total transactions per salesperson:
(totS <- table(ID))

# Total transactions per product:
(totP <- table(Prod))

# The salespersons and products involved in the
# problematic transactions with unknowns in both
# Val and Quant are (note use of which()):
nas <- sales[which(is.na(Quant) & is.na(Val)),
             c('ID','Prod')]
nas

# How many?
nrow(nas) # 888 of them

# Salespeople with larger proportions of transac-
# tions with unknowns in both Val and Quant:
propS <- 100*table(nas$ID)/totS
# We just go for top 10:
propS[order(propS,decreasing=T)[1:10]]

# Seems reasonable to delete these transactions
# of salespeople as they only represent a small
# proportion of their transactions. The alternative
# of filling in both columns reliably is more
# risky.

# Here they are with respect to the products:
propP <- 100*table(nas$Prod)/totP
propP[order(propP,decreasing=T)[1:10]]

# Are several products that would have more than
# 20% of their transactions removed, particularly
# product p2689 would have 40% removed.

# However, after examining our alternatives, it
# appears that the option of removing all transactions
# with unknown values on both quantity and value is
# the best option.

# Need to detach sales before we begin to delete since
# so we do not end up with inconsistent views of
# the data....we want to make our permanent changes
# to the sales dataframe, not to the attached (and
# copied) dataset.
detach(sales)

# This make the change permanent to the sames dataset
# that is now active in our workspace (but not to
# the one loaded again with data(sales) statement).

# Note we 'negative out' the records with both
# Val and Quant missing.
sales <- sales[-which(is.na(sales$Quant) & is.na(sales$Val)),]

# We begin to analyze remaining reports with either
# Quant or Val missing.

# Calculate proportion of transactions of each product
# that have quantity unknown:
nnasQp <- tapply(sales$Quant,list(sales$Prod),
                 function(x) sum(is.na(x)))
propNAsQp <- nnasQp/table(sales$Prod)
# Look at top ten as usual:
propNAsQp[order(propNAsQp,decreasing=T)[1:10]]

# Products p2442 and p2443 have all transactions with
# unknown values for quantity, making it impossible to
# calculate typical unit price.

# We delete them:
sales <- sales[!sales$Prod %in% c('p2442','p2443'),]

# Having deleted two products, we update levels of
# the column Prod:
nlevels(sales$Prod)
sales$Prod <- factor(sales$Prod)
nlevels(sales$Prod)

# Are there salespeople with all transactions 
# that have an unknown quantity?:
nnasQs <- tapply(sales$Quant,list(sales$ID),function(x) sum(is.na(x)))
propNAsQs <- nnasQs/table(sales$ID)
propNAsQs[order(propNAsQs,decreasing=T)[1:10]]

# Yep. Are 5 salespeople who have not filled in
# the information on the quantity in their reports.
# However, if we have the other transactions of the
# same products reported by other salespeople, we
# can try to use this information to fill in the
# unknowns on the assumption that the unit price is
# similar. So we do not delete them.

# We carry out a similar analysis for transactions with
# an unknown value in Val. First, proportion of
# transactions of each product with unknown value in
# this column:
nnasVp <- tapply(sales$Val,list(sales$Prod),
                 function(x) sum(is.na(x)))
propNAsVp <- nnasVp/table(sales$Prod)
propNAsVp[order(propNAsVp,decreasing=T)[1:10]]

# The numbers are reasonable so it does not make sense
# to delete these transactions as we may try to fill in
# these holes using the other transactions. With respect
# to salesperson, the numbers are as follows:
nnasVs <- tapply(sales$Val,list(sales$ID),function(x) sum(is.na(x)))
propNAsVs <- nnasVs/table(sales$ID)
propNAsVs[order(propNAsVs,decreasing=T)[1:10]]

# Again, these proportions are not too high.

# So now we have removed all reports with insufficient
# information to use a fill-in strategy. For remaining
# missing values, we use a method based on assumption
# that transactions of the same products should have a
# similar unit price.

# We begin by obtaining this typical unit price for each
# product. We skip prices of transactions already deemed
# to be fraudulent. For remaining (non-fraud) transactions
# we use median unit price of the transactions as the
# 'typical' unit price:
tPrice <- tapply(sales[sales$Insp != 'fraud','Uprice'],
                 list(sales[sales$Insp != 'fraud',
                            'Prod']),median,na.rm=T)

# Look at tPrice
tPrice

# Still have 4,546 of them:
nrow(tPrice)

# There is a wide spread:
range(tPrice)

# But most of prices < $200
table(tPrice)

# With a typical unit price for each product, we can use
# it to calculate any of the two missing values of Quant
# and/or Val because we have none where both are missing.

# We fill in remaining missing values. Note the code is
# rather compact in spite of filling in 12,900 missing
# quantity values and 294 total values (of transactions).
noQuant <- which(is.na(sales$Quant))
# ceiling() avoids non-integer values of Quant; it returns
# the smallest integer not less than the number used as
# the argument:
sales[noQuant,'Quant'] <- ceiling(sales[noQuant,'Val'] /
                                  tPrice[sales[noQuant,'Prod']])
noVal <- which(is.na(sales$Val))
sales[noVal,'Val'] <- sales[noVal,'Quant'] *
                      tPrice[sales[noVal,'Prod']]

# We have filled in Quant and Val values so now we
# recalculate the Uprice column to fill in the previously
# unknown unit prices:
sales$Uprice <- sales$Val/sales$Quant
getwd()
# So now we have a dataset with no missing (or unknown)
# values. We want to save this current state of the sales
# data frame so we can restart our analysis from this
# point from now on:
save(sales,file='salesClean.Rdata')

# We can load saved objects back to workspace with load().

#######################################################
######     Few Transactions of Some Products     ######
#######################################################

# There are products with very few transactions. This is
# problematic as we need to use the information on these
# transactions to decide if any are unusual. We run into
# problems determining the statistical significance of
# these with so few transactions.

# We can try to infer relationships between products,
# particularly the similarity between their distributions
# of unit price.

# If we can find products with similar prices, we can
# merge those with too few transactions to search for
# the unusual (fraudulent) values.

# We obtain measures of central tendency (median) and
# spread (IQR), even though we are assuming normality,
# since we will be looking for outliers.

# Obtain median and IQR for all transactions of each
# product:
attach(sales)
notF <- which(Insp != 'fraud')
ms <- tapply(Uprice[notF],list(Prod=Prod[notF]),
             function(x) {
    # boxplot.stats() obtains values of median,
    # 1st and 3rd quartiles calculated for all
    # sets of transactions of each product, eliminating
    # the fraudulent transactions
     bp <- boxplot.stats(x)$stats
     c(median=bp[3],iqr=bp[4]-bp[2])
   })
# Using this we obtain a matrix of the median and
# IQR for each product:
ms <- matrix(unlist(ms),
             length(ms),2,
             byrow=T,
             dimnames=list(names(ms),c('median','iqr')))
head(ms)

# Now we plot each product according to its respective
# median and IQR
par(mfrow=c(1,2))
# Too many overlap with regular scales
# Note p3689 top right
plot(ms[,1],ms[,2],xlab='Median',
     ylab='IQR',main='Some Properties of the ')
# So we use a log scale in second plot and
# we use '+' to indicate transactions with
# fewer than 20 transactions. log=xy sets
# log scales of both axes of second plot:
plot(ms[,1],ms[,2],xlab='Median',
     ylab='IQR',main='Distributions of Unit Prices',
     col='grey',log="xy")
smalls <- which(table(Prod) < 20)
points(log(ms[smalls,1]),log(ms[smalls,2]),pch='+')

# Can see are many products with approximately
# same median and IQR, suggesting similarities
# of their distributions of unit prices.

# Also, for those with < 20 transactions, many
# are very similar to other products. But there
# are also products with few transactions that
# have distinct unit price distributions. These
# will be more difficult for determining fraud.

# Visual tests are not adequate, we need to use
# formal tests for more precision to compare the
# distributions of two products.

# We use non-parametric Kolmogorov-Smirnov test
# which is more robust, especially with outliers.
# Can use to test null hypothesis that both come
# from same distribution. It calculates a statistic
# that measures the maximum difference between the
# two empirical cumulative distribution functions.
# Similar distributions will have small CDF diffs.

# For each product with fewer than 20 transactions, 
# we search for the product with the most similar
# unit price distribution and then use a K-S test
# to check if similarity is statistically significant.

# Cannot do it exhaustively, for all combinations
# of products, so we search for the product with
# the most similar median and IQR, then do K-S
# test for their respective unit price distributions.

# Normalize data to avoid negative scale effects
# when we compute distance:
dms <- scale(ms)
# Then gather up those transactions with < 20
smalls <- which(table(Prod) < 20)
# Get list of unit prices by product:
prods <- tapply(sales$Uprice,sales$Prod,list)
# Set up matrix for results of K-S test:
similar <- matrix(NA,length(smalls),
                  7,dimnames=list(names(smalls),
                                  c('Simil',
                                    'ks.stat',
                                    'ks.p','medP',
                                    'iqrP','medS',
                                    'iqrS')))
# Main loop goes over all products with few trans:
for(i in seq(along=smalls)) {
  # the next two statements calculate distance
  # between the distribution properties of the
  # current product (i)
  d <- scale(dms,dms[smalls[i],],FALSE)
  # and all other products; d is those distances;
  # the second smallest distance is the product
  # that is the most similar
  d <- sqrt(drop(d^2 %*% rep(1,ncol(d))))
  # Perform Kolmogorov-Smirnov test to compare
  # the two distributions of unit prices
  stat <- ks.test(prods[[smalls[i]]],prods[[order(d)[2]]])
  similar[i,] <- c(order(d)[2],stat$statistic,stat$p.value,ms[smalls[i],],ms[order(d)[2],])
}

# ks.stat is the maximum difference between the
# two CDFs. ks.p is confidence level, then the
# medians and IQRs of the product, and the most
# similar product, respectively.
head(similar)
# First column is product ID for which
# we are obtaining the most similar product

# Can obtain respective similar product ID with:
levels(Prod)[similar[1,1]]

# So now we can check how many products have a
# product whose unit price distribution is
# significantly similar with 90% confidence.
nrow(similar[similar[,'ks.p'] >= 0.9,])

# Or, more efficiently:
sum(similar[,'ks.p'] >= 0.9)

###########################################
# So, for the 985 products with fewer than
# 20 transactions, we have found similarly
# priced products for only 117 of them. But
# this is still useful information as we can
# now include these additional 117 product
# transactions into the decision process for
# finding fraudulent transactions.

# We save the similar object in case we decide
# to use this similarity between products later:
save(similar,file='similarProducts.Rdata')
#############################################
