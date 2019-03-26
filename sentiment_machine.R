## Build a simple CLI application that predicts the sentiment of food reviews.

## The data is a tab delimited file consisting of the food review text 
## and sentiment target (value of 0 or 1):
##  - train.txt - Food reviews with the sentiment target of 0 or 1
##  - test.txt - test data consisting of food reviews without sentiment target

## The assignment covers designing and building the algorithm that will accept 
## the text of any question and output the closest matching food reviewsentiment. 
## The application should be a simple CLI tool which allows the user to input any 
## question text. It should output the matching sentiment, with a relevance measure 
## for each (an indicator of how well it matched). 
## You are free to come up with a relevance measure, 
## but provide a clear description of your selection.

## Despite this being a small command line app, 
## please approach this as you would a production problem using whatever 
## approach to coding and testing you feel appropriate.

## Please include a Readme with any additional information you would like 
## to include such as assumptions you made. You may wish to use it to explain 
## any design decisions too.

## Libraries
library(dplyr)
library(tidytext)

## Load data
train <- read.delim("data/train.txt", sep = "\t", header = F, quote = "")
test  <- read.delim("data/test.txt", sep = "\t", header = F, quote = "")

names(train) <- c("text", "sentiment")
names(test)  <- c("text")

head(train)
head(test)

dim(train)
