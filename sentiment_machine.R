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
library(caret)
library(rpart)
library(randomForest)

## Load data
train <- read.delim("data/train.txt", sep = "\t", header = F, quote = "")
test  <- read.delim("data/test.txt", sep = "\t", header = F, quote = "")

names(train) <- c("text", "sentiment")
names(test)  <- c("text")

## Join train and test sets for data preprocessing and exploration
train$source <- "train"
test$source  <- "test"
data <- rbind(train[, c(1, 3)],
              test[,  c(1, 2)])

data_df <- tibble(line = 1:nrow(data), text = data$text)

## tokenize words
data_tidy <- data_df %>%
  unnest_tokens(word, text)


# Create sentiment features -----------------------------------------------

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

sentiments


# Create topic model ------------------------------------------------------

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
reviews_documents


# Join all features -------------------------------------------------------

data <- sentiments %>%
  left_join(reviews_documents, by = c("line" = "document"))

data[is.na(data)] <- 0


# Partition data ----------------------------------------------------------

set.seed(2345)

train_index <- createDataPartition(train$sentiment, p = 0.8, list = F)

x_train <- data[1:nrow(train), ][ train_index, -1]
x_test  <- data[1:nrow(train), ][-train_index, -1]

y_train <- train$sentiment[ train_index]
y_test  <- train$sentiment[-train_index]

dim(x_train)
length(y_train)

dim(x_test)
length(y_test)

# test_df  <- train_df[-train_index, ]
# train_df <- train_df[ train_index, ]


# Models ------------------------------------------------------------------

## logistic model
lm.fit <- glm(sentiment ~ sentiment_bing + sentiment_afinn + litigious + superfluous + uncertainty + anger + anticipation + disgust + fear + joy + sadness + surprise + trust + negative.y + negative.x + positive.y + positive.x,
              data = train_df,
              family = "binomial")
lm.fit <- glm.fit(x = x_train, 
                  y = data.frame(y_train), 
                  family = "binomial")

summary(lm.fit)

lm.pred <- predict(lm.fit, newdata = test_df)
lm.pred <- ifelse(lm.pred < 0.5, 0, 1)

## 78%
table(lm.pred, test_df$sentiment)


rpart.fit <- rpart(sentiment ~ sentiment_bing + sentiment_afinn + litigious + superfluous + uncertainty + anger + anticipation + disgust + fear + joy + sadness + surprise + trust + negative.y + negative.x + positive.y + positive.x,
                   data = train_df)

summary(rpart.fit)

rpart.pred <- predict(rpart.fit, newdata = test_df)
rpart.pred <- ifelse(rpart.pred < 0.5, 0, 1)

## 78%
table(rpart.pred, test_df$sentiment)

## Random Forest
rf.fit <- randomForest(x = x_train,
                       y = as.factor(y_train),
                       ntree = 500)
rf.pred <- predict(rf.fit, newdata = x_test)

## 76%
table(rf.pred, y_test)

## matches
match <- rf.pred != test_df$sentiment
train[test_df[match, ]$line, ]
