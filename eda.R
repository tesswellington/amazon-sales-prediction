### ----------------------------------------------------------------------- ###
# Visualizations/EDA for Predicting Amazon Customer Order Totals 
# Author: Tess Wellington
# Course: STATS 101C @ UCLA
# Date: July 2024
# Description:
# This script implements basic data cleaning, EDA, and visualizations 
# for the purpose of exploring the Amazon customer order total training dataset.
### ----------------------------------------------------------------------- ###

### -------------------- ### LIBRARIES / FILES ### ------------------------ ###

library(tidyverse)
library(tidymodels)
library(ggplot2)
library(dplyr)
library(ranger)
library(corrr)
library(glmnet)
library(kknn)
library(xgboost)

# training files
amazon_order_train <- read_csv("amazon_order_details_train.csv")
customer_info_train <- read_csv("customer_info_train.csv")
train <- read_csv("train.csv")

# test files
amazon_order_test <- read_csv("amazon_order_details_test.csv")
customer_info_test <- read_csv("customer_info_test.csv")
test <- read_csv("test.csv")


# removing the variable `order_totals` as it will allow the prediction of `log_total` perfectly
train <- 
  train %>%
  select(-order_totals)

### ------------------------- ### MEGA TIBBLE ### ------------------------ ###

mega_train <-
  full_join(amazon_order_train, customer_info_train, by="survey_response_id")

mega_train <-
  mega_train %>%
  select(-title, 
         -asin_isbn_product_code, 
         -q_sell_your_data, 
         -q_sell_consumer_data, 
         -q_small_biz_use,
         -q_census_use, 
         -q_research_society)

head(mega_train)

train_cat <-
  mega_train %>%
  select(category, quantity, item_cost) %>%
  filter(!is.na(category)) %>%
  group_by(category) %>%
  mutate(median_cost = median(item_cost))

train_cat_count <-
  train_cat %>%
  count(category) %>%
  arrange(desc(n)) %>%
  filter(n > 7500)

head(train_cat)
train_cat_count

train_cat_join <- 
  left_join(train_cat_count, train_cat, by = "category") %>%
  group_by(category) %>%
  reframe(n, median_cost) %>%
  unique()

train_cat_join

megatrain %>%
  group_by(survey_response_id, order_date) %>%
  mutate(order_total = sum(item_cost))

### ----------------------- ### EDA (CORRELATION) ### --------------------- ###

train_lim <- 
  train %>%
  select(year, month, log_total, count)

head(train_lim)

train_cor <- train_lim %>%
  correlate() %>%
  rearrange() %>%
  shave()

rplot(train_cor)

### ----------------------- ### VISUALIZATIONS ### ------------------------ ###

# BUBBLE

ggplot(train_cat_join, aes(x = reorder(category, -n), y = n, size = median_cost, col = median_cost)) +
  geom_point(alpha = 0.7) +
  theme(plot.title = element_text(hjust=0.5), axis.text.x = element_text(angle = -90, hjust = 0, vjust = 0)) +
  xlab("Category") +
  ylab("Frequency") +
  ggtitle("Frequency and Median Cost of Top 10 Categories")

# graph 1: date vs order totals

train_by_state <- 
  train %>%
  mutate(state_id = unclass(factor(state))) %>%
  relocate(state_id, .before = state) %>%
  group_by(state)

train_by_state %>%
  print(n = 50)

train_by_date <-
  train %>%
  select(state, year, month, log_total, count) %>%
  mutate(total = 10^log_total) %>%
  group_by(year, month) %>%
  summarize(log_total_USA = log(sum(total)), count_total_USA = sum(count))


head(train_by_date)

col <- c(1, 2, 3, 4, 5)

ggplot(train_by_date, aes(x = month, y = log_total_USA, group = year, col = year)) +
  geom_line(lty = 1, size = 1.2)

train_by_state <-
  train %>%
  mutate(state_id = unclass(factor(state))) %>%
  relocate(state_id, .before = state) %>%
  select(state, year, month, log_total, count) %>%
  mutate(date = lubridate::make_date(year, month))

head(train_by_state)

ggplot(train_by_state, aes(x = date, y = log_total, group = state, col = state)) +
  geom_line(lty = 1, size = 0.8)

ggplot(train_by_state, aes(x = date, y = log_total, group = state, col = state)) +
  geom_boxplot()

# graph 2: state vs order totals

train_by_region <-
  train %>%
  select(state, year, month, log_total, count) %>%
  group_by(state) %>%
  mutate(avg_log_total = mean(log_total)) %>%
  arrange(desc(avg_log_total))

head(train_by_region)
train_by_region

ggplot(train_by_region, aes(x = reorder(state, -avg_log_total), y = log_total)) +
  geom_boxplot() +
  theme(plot.title = element_text(hjust=0.5), axis.text.x = element_text(angle = -90, hjust = 0, vjust = 0)) +
  xlab("State") +
  ylab("Average log_total") +
  ggtitle("Average log_total by State")