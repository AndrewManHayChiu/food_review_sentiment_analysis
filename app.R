#!/usr/bin/env Rscript 

## This is a command line application
## Run this app by entering the following in a command line tool:
##      $ Rscript app.R [text]

## This application takes in 1 (one) argument, text, from the command line
## and returns a predicted sentiment.

library(sentimentr)

text <- commandArgs(trailingOnly = TRUE)

result <- sentiment_by(text)

avg_sentiment <- result$ave_sentiment

predicted_sentiment <- ifelse(avg_sentiment <= 0, 0, 1)

print(paste("The average sentiment of your review is:", avg_sentiment, sep = " "))

print(paste("The predicted sentiment of your review is:", predicted_sentiment, sep = " "))