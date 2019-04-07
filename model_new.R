
library(tidytext)
library(dplyr)
library(purrr)
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


## random input text
text <- as.character("The food was amazing!")

# Process data ------------------------------------------------------------

data_df <- tibble(line = 1:nrow(data), text = data$text)

data_tidy <- data_df %>%
  unnest_tokens(output = word, 
                input = text,
                token = "words")

## Create word_index for each word in each review
word_index <- unique(data_tidy$word)
word_index <- data.frame(index = c(1:length(word_index)),
                         word = word_index)
data_tidy <- left_join(data_tidy, word_index)

head(data_tidy)

## Create list of reviews, with words replaced by word_index
list_data <- split(data_tidy$index, data_tidy$line)
list_data <- unname(list_data)

list_data[[10]]

## turn word_index into list
word_index_list <- split(as.character(word_index$word), word_index$index)
word_index_list <- unname(word_index_list)

word_index_list[1:5]

## Prepare data

data <- pad_sequences(
  list_data,
  maxlen = 35,
  padding = "post",
  truncating = "post"
)

data[1, ]


# build model -------------------------------------------------------------

model <- keras_model_sequential()

model %>%
  layer_embedding(input_dim = length(word_index_list), output_dim = 16) %>%
  layer_global_average_pooling_1d() %>%
  layer_dense(units = 8, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% summary()

model %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics = list('accuracy')
)

history <- model %>%
  fit(
    x = data[1:900, ],
    y = train$sentiment[1:900],
    epochs = 60,
    batch_size = 30,
    validation_data = list(data[901:959, ], train$sentiment[901:959]),
    verbose = 1
  )

results <- model %>% evaluate(data[901:959, ], train$sentiment[901:959])
results


# Process new text --------------------------------------------------------

text <- data.frame(text = text)
text$text <- as.character(text$text)

text_tidy <- text %>%
  unnest_tokens(output = word, 
                input = text,
                token = "words") %>%
  left_join(word_index)

text_data <- split(text_tidy$index, 1)
text_data <- unname(text_data)

text_data <- pad_sequences(
  text_data,
  maxlen = 35,
  padding = "post",
  truncating = "post",
)

text_data[1, ]


# make prediction ---------------------------------------------------------

predict(model, text_data)


# Save model --------------------------------------------------------------

write.csv(word_index, "word_index.csv")

model %>%
  save_model_hdf5("model/model.h5")