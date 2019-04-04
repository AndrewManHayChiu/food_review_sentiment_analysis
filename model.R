
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

data_tidy <- data_df %>%
  unnest_tokens(output = word, 
                input = text,
                token = "words")
filter(!str_detect(word, "[0-9]")) %>%
  anti_join(stop_words)

data_df <- tibble(line = 1:nrow(data), text = data$text)


# Preprocess data ---------------------------------------------------------

## Sentiments
sentiments_bing <- data_tidy %>%
  inner_join(get_sentiments("bing")) %>%
  group_by(line) %>%
  count(sentiment) %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(sentiment_bing = positive - negative) %>%
  select(-negative, -positive)

sentiments_bing <- data_df %>%
  select(-text) %>%
  left_join(sentiments_bing, by = c("line" = "line")) %>%
  mutate(sentiment_bing = ifelse(is.na(sentiment_bing), 0, sentiment_bing))

sentiments_afinn <- data_tidy %>%
  inner_join(get_sentiments("afinn")) %>%
  group_by(line) %>%
  summarise(sentiment_afinn = sum(score))

sentiments_afinn <- data_df %>%
  select(-text) %>%
  left_join(sentiments_afinn, by = c("line" = "line")) %>%
  mutate(sentiment_afinn = ifelse(is.na(sentiment_afinn), 0, sentiment_afinn))

sentiments_loughran <- data_tidy %>%
  inner_join(get_sentiments("loughran")) %>%
  group_by(line) %>%
  count(sentiment) %>%
  spread(sentiment, n, fill = 0)

sentiments_nrc <- data_tidy %>%
  inner_join(get_sentiments("nrc")) %>%
  group_by(line) %>%
  count(sentiment) %>%
  spread(sentiment, n, fill = 0)


sentiments <- sentiments_bing %>%
  left_join(sentiments_afinn, by = c("line" = "line")) %>%
  left_join(sentiments_loughran, by = c("line" = "line")) %>%
  left_join(sentiments_nrc, by = c("line" = "line"))

sentiments[is.na(sentiments)] <- 0

sentiments2 <- sapply(data$text, function(x) sentiment_by(x))
sentiments2 <- t(as.data.frame(sentiments2))
sentiments2 <- data.frame(sentiments2)
word_length <- as.numeric(sentiments2$word_count)
sentiments2 <- as.numeric(sentiments2$ave_sentiment)

## Topics
## Cast Document Term Matrix (DTM)
reviews_dtm <- data_tidy %>%
  anti_join(stop_words) %>%
  group_by(line) %>%
  count(word) %>%
  cast_dtm(document = line, 
           term = word, 
           value = n)

reviews_lda <- LDA(reviews_dtm, k = 5, control = list(seed = 2345))

reviews_topics <- tidy(reviews_lda, matrix = "beta")

## Document-topic probabilities
reviews_documents <- tidy(reviews_lda, matrix = "gamma") %>%
  mutate(document = as.numeric(document)) %>%
  spread(topic, gamma)

names(reviews_documents)[2:6] <- c("topic_service", 
                                   "topic_experience",
                                   "topic_time",
                                   "topic_food",
                                   "topic_restaurant")

## DTM
reviews_dtm <- removeSparseTerms(reviews_dtm,
                                 sparse = 0.995)

## Convert to data frame
reviews_df <- tidy(reviews_dtm) %>%
  mutate(document = as.numeric(document)) %>%
  spread(term, count, fill = 0)

data <- sentiments %>%
  left_join(reviews_documents, by = c("line" = "document")) %>%
  left_join(reviews_df, by = c("line" = "document")) %>%
  mutate(sentiments2 = sentiments2,
         # sentiments3 = 
         word.length = word_length)

data[is.na(data)] <- 0

## Partition data
set.seed(2345)

train_index <- createDataPartition(train$sentiment, p = 0.8, list = F)

x_train <- data[1:nrow(train), ][ train_index, -1]
x_test  <- data[1:nrow(train), ][-train_index, -1]

y_train <- train$sentiment[ train_index]
y_test  <- train$sentiment[-train_index]

# Train model -------------------------------------------------------------

# vocab_size = 182

## Initiate keras model
model <- keras_model_sequential()

## Define keras model
model %>%
  layer_dense(units = ncol(x_train) / 3, input_shape = ncol(x_train)) %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% summary()

## Configure the model with an optimiser and loss function
model %>%
  compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = list("accuracy")
  )

history <- model %>%
  fit(
    x = as.matrix(x_train),
    y = y_train,
    epochs = 40,
    batch_size = 50,
    validation_data = list(as.matrix(x_test), y_test),
    verbose = 1
  )

## Evaluate model
results <- model %>% evaluate(as.matrix(x_test), y_test)
results

predict(model, as.matrix(x_test))

# Save model --------------------------------------------------------------

model %>%
  save_model_hdf5("model/model.h5")