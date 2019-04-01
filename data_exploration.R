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
library(stringr)
library(SnowballC)
library(sentimentr)
library(topicmodels)
library(ggplot2)

## Load data
train <- read.delim("data/train.txt", sep = "\t", header = F, quote = "", stringsAsFactors = F)
test  <- read.delim("data/test.txt", sep = "\t", header = F, quote = "", stringsAsFactors = F)

names(train) <- c("text", "sentiment")
names(test)  <- c("text")

head(train)
head(test)

dim(train)
dim(test)

str(train)
str(test)

## Sentiment
## 1: positive
## 0: negative

## Join train and test sets for data preprocessing and exploration
train$source <- "train"
test$source  <- "test"
data <- rbind(train[, c(1, 3)],
              test[,  c(1, 2)])

data_df <- tibble(line = 1:nrow(data), text = data$text)

## tokenize words
## Note: The unnest_tokens function also performs some cleaning of the text 
##       before tokenizing, including:
##       - Removing white space
##       - Removing punctuation
data_tidy <- data_df %>%
  unnest_tokens(word, text)

## Sentiment analysis does not work well when text is very short.
## what's the word length of each text?
## Min: 1
## Max: 32
## Mean: 10.9
data_tidy %>%
  count(line) %>%
  summary()

## This is about food reviews; what are the most common words?
## food, service, time, friendly, restaurant, delicious, experience
## Clearly, this dataset is about food!
data_tidy %>%
  count(word) %>%
  anti_join(stop_words) %>%  # Exclude stop words
  arrange(desc(n))

## What are some of the least common words?
## Uncommon words could potentially be modified, or point to spelling mistakes.
## Most of the least common words have numbers; 
## these should be removed prior to tokenizing.
## However, some numbers are likely to be informative, 
## as many "reviews" may incorporate a 5 or 10 star rating.
## Hence, words greater than 10 should be removed.
data_tidy %>%
  count(word) %>%
  anti_join(stop_words) %>%  # Exclude stop words
  arrange(n)

## After removing numbers, some words are different versions of the same word.
## For example, absolute and absolutely.
## Words should then be stemmed
data_df %>%
  unnest_tokens(word, text) %>%
  filter(!str_detect(word, "[0-9]")) %>%
  anti_join(stop_words) %>%
  count(word) %>%
  # arrange(n)
  arrange(desc(n))

## Stem words?
## Not before getting sentiments
data_df %>%
  unnest_tokens(word, text) %>%
  filter(!str_detect(word, "[0-9]")) %>%
  anti_join(stop_words) %>%
  mutate(word = wordStem(word)) %>%
  count(word) %>%
  arrange(desc(n))

## We can also glean that the reviews talk about certain aspects of food:
## 1. Food
## 2. Service
## 3. Time
## 4. Restaurant
## 5. Experience
## 
## While opinions include:
## 1. Friendly
## 2. Nice
## 3. Amazing
## 4. Delicious
## 
## With this in mind, it might help to tag each review with what the
## subject is about.
## Maybe make the default to be about food.
## Note: Try to incorporate topics as a feature

## N-grams
## There aren't a huge amount of text, so N grams may not provide much help
data_df %>%
  unnest_tokens(bigram, text, token = "ngrams", n = 2) %>%
  count(bigram) %>%
  arrange(desc(n))
data_df %>%
  unnest_tokens(trigram, text, token = "ngrams", n = 3) %>%
  count(trigram) %>%
  arrange(desc(n))

## Sentiments and target sentiment
sentiments_bing <- data_tidy %>%
  inner_join(get_sentiments("bing")) %>%
  group_by(line) %>%
  count(sentiment) %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(sentiment_bing = positive - negative) %>%
  select(-negative, -positive)

sentiments_bing <- train %>%
  mutate(line = 1:nrow(train)) %>%
  left_join(sentiments_bing, by = c("line" = "line")) %>%
  mutate(sentiment_bing = ifelse(is.na(sentiment_bing), 0, sentiment_bing))

## This shows there is a high correlation between calculated and actual sentiment.
## Therefore, this is an important feature to create for the ML model.
table(sentiments_bing$sentiment, sentiments_bing$sentiment_bing)
cor(sentiments_bing$sentiment, sentiments_bing$sentiment_bing, use = "complete.obs")

## Redo this exercise with another sentiment lexicon
## Pretty similar result, but with a larger spread of calculated sentiment
sentiments_afinn <- data_tidy %>%
  inner_join(get_sentiments("afinn")) %>%
  group_by(line) %>%
  summarise(sentiment_afinn = sum(score))

sentiments_afinn <- train %>%
  mutate(line = 1:nrow(train)) %>%
  left_join(sentiments_afinn, by = c("line" = "line")) %>%
  mutate(sentiments_afinn = ifelse(is.na(sentiments_afinn), 0, sentiments_afinn))

table(sentiments_afinn$sentiment, sentiments_afinn$sentiment_afinn)
cor(sentiments_afinn$sentiment, sentiments_afinn$sentiment_afinn, use = "complete.obs")

## Other features could be other types of sentiments from the
## loughran and nrc lexicons
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

## negating sentiments; valence shifters
sentiment_by("not good")
sentiment_by(text.var = "I am not very good")
sentiment_by(text.var = "I am not very good. But my partner is fantastic.")
sentiment_by(text.var = c("I am not very good.", 
                          "But my partner is fantastic."))
sentiment_by(data$text[1], by = NULL)
sentiment_by(data$text[2], by = NULL)


## takes a minute or two
# train_sentiments <- sapply(train$text, function(x) sentiment_by(x)$ave_sentiment)
train_sentiments <- sapply(train$text, function(x) sentiment_by(x))
train_sentiments <- t(as.data.frame(train_sentiments))
train_sentiments <- data.frame(train_sentiments)
train_sentiments_binary <- ifelse(train_sentiments$ave_sentiment <= 0, 0, 1)
train_sentiments_binary <- as.numeric(train_sentiments_binary)
table(train_sentiments_binary, train$sentiment) # This already gets 80% accuracy!
## What sentences are they going wrong?
## Most often, it's where the sentiment is close to 0.
## Furthermore, it's different depending on whether the TRUE sentiment is 0 or 1.
## Note: SENTIMENT should definitely be a feature.
train %>%
  mutate(sentiment_pred = train_sentiments, 
         sentiment_binary = train_sentiments_binary) %>%
  filter(sentiment != sentiment_binary) %>%
  select(-source) %>%
  # head(15)
  # summary()
  ggplot(aes(x = sentiment_pred)) +
  geom_histogram() +
  facet_wrap(sentiment ~ .)

train %>%
  mutate(sentiment_pred = train_sentiments, 
         sentiment_binary = train_sentiments_binary) %>%
  ggplot(aes(x = sentiment_pred)) +
  geom_histogram() +
  facet_wrap(sentiment ~ .)

## If the sentimentr got the sentiment wrong, what was the difference in word length?
## If it was wrong, and sentiment was bad, the review is usually longer.
## If it was wrong, and sentiment was good, the review is usually shorter
## Note: WORD LENGTH SHOULD ALSO BE A FEATURE
word_length <- as.numeric(train_sentiments$word_count)
train %>%
  mutate(sentiment_pred = train_sentiments$ave_sentiment, 
         sentiment_binary = train_sentiments_binary,
         word_length = word_length) %>%
  filter(sentiment_binary != sentiment) %>%
  select(-source) %>%
  # group_by(sentiment) %>%
  ggplot(aes(x = word_length)) +
  geom_histogram() +
  facet_wrap(sentiment ~ .)

## Parts of Speech tagging
## Most reviews should be about "something".
data_tidy %>%
  inner_join(parts_of_speech) %>%
  filter(pos == "Noun")


## Term Frequency - Inverse Document Frequency
## Calculate weights for each term, decreasing the weight for common words,
## and increasing the weight for words not used often.
data_tidy %>%
  group_by(line) %>%
  count(word) %>%
  bind_tf_idf(word, line, n)

## DTM
reviews_dtm <- data_tidy %>%
  anti_join(stop_words) %>%
  group_by(line) %>%
  count(word) %>%
  cast_dtm(document = line, 
           term = word, 
           value = n)

reviews_dtm

reviews_lda <- LDA(reviews_dtm, k = 5, control = list(seed = 2345))
reviews_lda

reviews_topics <- tidy(reviews_lda, matrix = "beta")

reviews_top_terms <- reviews_topics %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)


reviews_top_terms %>%
  ggplot(aes(x = term, y = beta)) +
  geom_col() +
  facet_wrap(topic~., scales = "free") +
  coord_flip()

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
