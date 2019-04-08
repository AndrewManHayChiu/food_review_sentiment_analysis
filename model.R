
# Libraries ---------------------------------------------------------------

library(dplyr)
library(tidyr)
library(tidytext)
library(qdap)
library(stringr)
library(caret)
library(randomForest)
library(sentimentr)


# Load data ---------------------------------------------------------------

train <- read.delim("data/train.txt", sep = "\t", header = F, quote = "", stringsAsFactors = F)
test  <- read.delim("data/test.txt", sep = "\t", header = F, quote = "", stringsAsFactors = F)

names(train) <- c("text", "sentiment")
names(test)  <- c("text")

## Join train and test sets for data processing
train$source <- "train"
test$source  <- "test"
data <- rbind(train[, c(1, 3)],
              test[,  c(1, 2)])

data_df <- tibble(line = 1:nrow(data), text = data$text)

data_tidy <- data_df %>%
  unnest_tokens(output = word, 
                input = text,
                token = "words") %>%
  mutate(word = replace_number(word)) %>%
  anti_join(stop_words)




# Preprocess data ---------------------------------------------------------

## Sentiments
# sentiments_bing <- data_tidy %>%
#   inner_join(get_sentiments("bing")) %>%
#   group_by(line) %>%
#   count(sentiment) %>%
#   spread(sentiment, n, fill = 0) %>%
#   mutate(sentiment_bing = positive - negative) %>%
#   select(-negative, -positive)
# 
# sentiments_bing <- data_df %>%
#   select(-text) %>%
#   left_join(sentiments_bing, by = c("line" = "line")) %>%
#   mutate(sentiment_bing = ifelse(is.na(sentiment_bing), 0, sentiment_bing))
# 
# sentiments_afinn <- data_tidy %>%
#   inner_join(get_sentiments("afinn")) %>%
#   group_by(line) %>%
#   summarise(sentiment_afinn = sum(score))
# 
# sentiments_afinn <- data_df %>%
#   select(-text) %>%
#   left_join(sentiments_afinn, by = c("line" = "line")) %>%
#   mutate(sentiment_afinn = ifelse(is.na(sentiment_afinn), 0, sentiment_afinn))
# 
# sentiments_loughran <- data_tidy %>%
#   inner_join(get_sentiments("loughran")) %>%
#   group_by(line) %>%
#   count(sentiment) %>%
#   spread(sentiment, n, fill = 0)
# 
# sentiments_nrc <- data_tidy %>%
#   inner_join(get_sentiments("nrc")) %>%
#   group_by(line) %>%
#   count(sentiment) %>%
#   spread(sentiment, n, fill = 0)
# 
# sentiments <- sentiments_bing %>%
#   left_join(sentiments_afinn, by = c("line" = "line")) %>%
#   left_join(sentiments_loughran, by = c("line" = "line")) %>%
#   left_join(sentiments_nrc, by = c("line" = "line"))
# 
# sentiments[is.na(sentiments)] <- 0

## Create sentiments from sentimentr package
sentiments2 <- sapply(data$text, function(x) sentiment_by(x))
sentiments2 <- t(as.data.frame(sentiments2))
sentiments2 <- data.frame(sentiments2)
word_length <- as.numeric(sentiments2$word_count)
sentiments2 <- as.numeric(sentiments2$ave_sentiment)


# Create Document Term Matrix ---------------------------------------------

reviews_dtm <- data_tidy %>%
  anti_join(stop_words) %>%
  group_by(line) %>%
  count(word) %>%
  cast_dtm(document = line, 
           term = word, 
           value = n) 
reviews_dtm <- removeSparseTerms(reviews_dtm,
                                 sparse = 0.995)

## Convert DTM to data frame
reviews_df <- tidy(reviews_dtm) %>%
  mutate(document = as.numeric(document)) %>%
  spread(term, count, fill = 0)


# Create training data ----------------------------------------------------

# data <- sentiments %>%
#   left_join(reviews_df, by = c("line" = "document")) %>%
#   mutate(sentiments2 = sentiments2,
#          word.length = word_length)

data <- data_df %>%
  select(line) %>%
  left_join(reviews_df, by = c("line" = "document")) %>%
  mutate(sentiment = sentiments2,
         word.length = word_length)

## Make sure there are no NAs
data[is.na(data)] <- 0

# Train model -------------------------------------------------------------

## Random Forest
rf_model <- randomForest(x = data[1:nrow(train), -1],
                       y = as.factor(train$sentiment),
                       ntree = 500)

# Save model --------------------------------------------------------------

saveRDS(rf_model, "model/rf_model.rds")


# Save column headers of data ---------------------------------------------

cols <- colnames(data)[-1]
empty_df <- data.frame(matrix(ncol = length(cols), nrow = 0))
colnames(empty_df) <- cols
# empty_df[is.na(empty_df)] <- 0 

write.csv(empty_df, "word_matrix.csv", row.names = F)
