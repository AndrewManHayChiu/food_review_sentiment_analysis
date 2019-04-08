# Assessment - Food Review Sentiment Analysis

## Introduction

This repository contains the submission for a tech evaluation.

The program is written in R. To run the program (see below),
the user should have the following libraries installed:

* sentimentr
* dplyr
* tidyr
* tidytext
* stringr
* qdap
* randomForest

## Methodology

As this is a natural language processing (NLP) problem,
a *bag of words* model is adopted, with additional features including:

* Word length
* Sentiment

### Word Length

Word length is simply the number of words that make up a food review / text.

### Sentiment

A sentiment is calculated by looking at the polarity of the food review.

This is based on the *sentimentr* package.

## Measure

The output of the model is a predicted sentiment, 
and the probability of the sentiment, based on a logistic distribution.

A logistic function was chosen because the target sentiment takes on a value of
0 or 1.

## Dev-Test

To replicate a product development environment, 
GitHub was used to develop the application.

## Running the app

Make sure to have the relevant R libraries installed (see Introduction)

To run the app:

1. Make sure R is executable through a Unix shell (in Windows, use Windows PowerShell).
2. Navigate to the directory containing the *app.R* file
3. Make sure the file can read as an executable (type "chmod +x app.R" to do this)
4. Run the app by typing "Rscript app.R [text]"
    + Where [text] is a string surrounded by quotation marks
    + For example, *Rscript app.R "tasted like dirt"*

