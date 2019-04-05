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
suppressMessages(require(purrr))
suppressMessages(require(keras))

## Load trained neural network
model <- load_model_hdf5("model/model.h5")

## Load word_index
word_index <- read.csv("word_index.csv", stringsAsFactors = F)

text <- data.frame(text = text)
text$text <- as.character(text$text)

text_tidy <- text %>%
  unnest_tokens(output = word, 
                input = text,
                token = "words") %>%
  left_join(word_index, by = ("word" = "word"))

text_data <- split(text_tidy$index, 1)
text_data <- unname(text_data)

text_data <- pad_sequences(
  text_data,
  maxlen = 35,
  padding = "post",
  truncating = "post",
)

## Make prediction
pred <- predict(model, text_data)

prediction <- ifelse(pred <= 0, 0, 1)

## Print results
print(paste("The predicted sentiment of your review is:", prediction, sep = " "))

print(paste("The probability of of this prediction is:", pred, sep = " "))

