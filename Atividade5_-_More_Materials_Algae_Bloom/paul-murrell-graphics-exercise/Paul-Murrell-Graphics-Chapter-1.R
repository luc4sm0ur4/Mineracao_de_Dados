##############################################
#####       PAUL MURRELL GRAPHICS        #####
##############################################

# There is a package in R, "RGraphics," that 
# contains all of the data sets used in his book 
# (which are not available in other existing 
# packages); and functions to reproduce the figures 
# in the book.

# Install (with R command line install.packages("RGraphics") 
# and then load the package into your session (with 
# R command library("RGraphics")).

# Then call each of the functions below.

# Inspect the code for each function. You can
# view the definition of any function if you
# are using RStudio by placing the cursor
# inside the function call and selecting
# Code --> Go To Function Definition in the
# Rstudio menu system.
Some
# are quite complicated.

# You will need to have the indicated packages
# installed before you call the figure's
# function. If, for some reason, you need to
# restart R (this happened to me on figure 1.8),
# you will need to reload (but not reinstall)
# many of the preceding packages.

# Also, for several of the plots (as indicated
# below), you will need to "Clear All"
# plots in the RStudio plot menu for the
# next plot to show correctly.

--------------------------------------------
  
install.packages ("RGraphics")
library("RGraphics")

# a simple scatterplot:
figure1.1()

# some standard plots:
figure1.2() 
# Note must "Clear All" to reset
# par back to one plot per frame.
# Can also accomplish manually by
par(mfrow = c(1, 1))

# A customized scatterplot:
figure1.3()

# A dramaticized barplot (be patient):
install.packages("grImport")
install.packages("XML")
install.packages("colorspace")
figure1.4()

# A trellis dotplot:
install.packages ("lattice")
figure1.5()

# A ggplot2 plot showing the relationship
# between miles per gallon (on the highway)
# and engine displacement (in liters). The
# data are divided into four groups based
# on the number of cylinders in the engine
# and different plotting symbols are used
# for each group and a separate linear model
# fit is shown for each group:
figure1.6()

# A map of New Zealand:
install.packages("maps")
install.packages("mapdata")
figure1.7()

# A financial plot:
install.packages("quantmod")
figure1.8()
# "Clear All" plots afterwards

# Didactic diagrams:
install.packages("party")
install.packages("ipred")
figure1.9()
# "Clear All" plots afterwards

# A table-like plot:
figure1.10()

# Didactic Diagrams:
figure1.11()

# A music score:
figure1.12()

# An infographic showing proportion of aid
# money unaccounted for in the reconstruction
# of Iraq:
install.packages("pixmap")

figure1.13()
