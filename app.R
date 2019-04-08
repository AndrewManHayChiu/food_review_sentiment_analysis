#!/usr/bin/env Rscript 

## This is a command line application
## Run this app by entering the following in a command line tool:
##      $ Rscript app.R [text]

## This application takes in 1 (one) argument, text, from the command line
## and returns a predicted sentiment.

## set arguments that can be provided when calling the script
text <- commandArgs(trailingOnly = TRUE)

## Load libraries, silently
suppressMessages(require(sentimentr))
suppressMessages(require(dplyr))
suppressMessages(require(tidyr))
suppressMessages(require(tidytext))
suppressMessages(require(stringr))
suppressMessages(require(qdap))

## Load trained neural network
rf_model <- readRDS("model/rf_model.rds")

## Load word_matrix
word_matrix <- read.csv("word_matrix.csv", stringsAsFactors = F)

text <- "The food was pretty average"  # for testing

text <- data.frame(text = text)
text$text <- as.character(text$text)

text_tidy <- text %>%
  unnest_tokens(output = word, 
                input = text,
                token = "words") %>%
  count(word) %>%
  spread(word, n)

sentiment <- sapply(text$text, function(x) sentiment_by(x))
sentiment <- t(as.data.frame(sentiment))
sentiment <- data.frame(sentiment)
word_length <- as.numeric(sentiment$word_count)
sentiment <- as.numeric(sentiment$ave_sentiment)

## Fill in word_matrix with new data
word_matrix <- bind_rows(word_matrix, text_tidy)[, 1:ncol(word_matrix)]
word_matrix$sentiment <- sentiment
word_matrix$word.length <- word_length
word_matrix[is.na(word_matrix)] <- 0

## Make prediction
pred <- predict(rf_model, word_matrix)

prediction <- ifelse(pred <= 0, 0, 1)

## Print results
print(paste("The predicted sentiment of your review is:", prediction, sep = " "))

print(paste("The probability of of this prediction is:", pred, sep = " "))

