# Assessment - Food Review Sentiment Analysis

## Introduction

This repository contains the submission for the tech eval of a job interview.

The objective is to create a command-line interface (CLI) application that 
predicts the sentiment of food reviews.

The requirement is to build an algorithm that accepts a text as an input, 
and outputs:

1. The sentiment of the closest matching food review
2. A relevance measure for how well the text matched the food review.

## Methodology

As this is a natural language processing (NLP) problem,
a *bag of words* model is adopted, with additional features including:

* Topic
* Word length
* Average sentiment


From this feature set, measures of similarity is calculated, 
and the closest matching review is selected as the result.

### Topic

Analysis of the most common words in the training data set found 5 dominant topics:

1. Food
2. Service
3. Time
4. Restaurant
5. Experience

This was verified by conducting topic analysis, using Latent Dirichlet Allocation (LDA).

### Word Length

Word length is simply the number of words that make up a food review / text.

### Average sentiment

An average sentiment is calculated by looking at the polarity of the food review.

This is derived in R using the *sentimentr* package.

## Dev-Test

To replicate a product development environment, 
GitHub was used to develop the application.

Development of the command line application is performed in the *create_app* branch.

## Run the app

To run the app:

1. Make sure R is executable through a Unix shell (in Windows, use Windows PowerShell).
2. Navigate to the directory containing the *app.R* file
3. Make sure the file is read as an executable (type "chmod +x app.R")
4. Run the app by typing "Rscript app.R [text]"
    + Where [text] is a string surrounded by quotation marks

### Relevance Measure

