
# Libraries ---------------------------------------------------------------

library(dplyr)
library(tidyr)
library(tidytext)
library(tm)
library(topicmodels)

library(tensorflow)
library(keras)

# Load data ---------------------------------------------------------------

train <- read.delim("data/train.txt", sep = "\t", header = F, quote = "", stringsAsFactors = F)
test  <- read.delim("data/test.txt", sep = "\t", header = F, quote = "", stringsAsFactors = F)

names(train) <- c("text", "sentiment")
names(test)  <- c("text")

## Join train and test sets for data preprocessing and exploration
train$source <- "train"
test$source  <- "test"
data <- rbind(train[, c(1, 3)],
              test[,  c(1, 2)])

data_df <- tibble(line = 1:nrow(data), text = data$text)



# Preprocess data ---------------------------------------------------------



# Train model -------------------------------------------------------------

vocab_size = 10000

## Initiate keras model
model <- keras_model_sequential()

## Define keras model
model %>%
  layer_embedding(input_dim = vocab_size, output_dim = 16) %>%
  layer_global_average_pooling_1d() %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% summary()


# Save model --------------------------------------------------------------

model %>%
  save_model_hdf5("model/model.h5")