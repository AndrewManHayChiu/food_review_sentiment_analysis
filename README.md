# Assessment - Food Review Sentiment Analysis

## Introduction

This repository contains the submission for the tech eval of a job interview.

The objective is to create a command-line interface (CLI) application that 
predicts the sentiment of food reviews.

The requirement is to build an algorithm that accepts a text, 
and outputs:

1. The sentiment of the closest matching food review
2. A relevance measure for how well the text matched the food review.

## Methodology

As this is a natural language processing (NLP) problem,
a *bag of words* model is adopted, with additional features including:

* Average sentiment
* Topic
* Word length

From this feature set, measures of similarity 

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

