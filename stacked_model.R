### ----------------------------------------------------------------------- ###
# Stacked Ensemble Model for Predicting Amazon Customer Order Totals 
# Author: Tess Wellington
# Contributors: Kali Olmstead, Lou Qian
# Course: STATS 101C @ UCLA
# Date: July 2024
# Description:
# This script implements a stacked ensemble model combining multiple models
# developed by my teammates and myself. I developed the stacked model and
# optimized model performances.
# The stacked model predicts total order value (log_total) for Amazon customers 
# based on demographic and purchase history data. The dataset consists of about
# 5,000 Amazon customers from 2018-2022, divided into training and test sets.
### ----------------------------------------------------------------------- ###
# Based Models Used in Stack:
# - Model 1: Random Forest (Tess Wellington)
# - Model 2: Random Forest w/ Hyperparameters (Tess Wellington)
# - Model 3: Boosted Tree (Tess Wellington)
# - Model 4: Random Forest (Kali Olmstead)
# - Model 5: Random Forest (Lou Qian)
# - Model 6: KNN (Lou Qian)
# - Model 7: KNN w/ PCA (Tess Wellington)
### ----------------------------------------------------------------------- ###

### -------------------- ### LIBRARIES / FILES ### ------------------------ ###

library(tidyverse)
library(tidymodels)
library(stacks)
library(ggplot2)
library(dplyr)
library(ranger)  # for rf models
library(glmnet)  # for glm models
library(kknn)    # for knn models
library(xgboost) # for boosted tree models

# training files
amazon_order_train <- read_csv("amazon_order_details_train.csv")
customer_info_train <- read_csv("customer_info_train.csv")
train <- read_csv("train.csv")
train <- 
  train %>%
  select(-order_totals)

# test files
amazon_order_test <- read_csv("amazon_order_details_test.csv")
customer_info_test <- read_csv("customer_info_test.csv")
test <- read_csv("test.csv")

### ----------------------- ### TESS'S STACK v4 ### ----------------------- ###

# initializing seed & folds:

set.seed(124)
train_folds <- vfold_cv(train, v = 10, strat = log_total)

ctrl_grid <- control_stack_grid()
ctrl_res <- control_stack_resamples()

# candidates:

tess_rf_res  <- 
  fit_resamples(
    tess_rf_workflow,
    resamples = train_folds,
    metrics = metric_set(rmse, mae),
    control = ctrl_res
  )

tess_rf_res2  <- 
  tune_grid(
    tess_rf_workflow2,
    resamples = train_folds,
    metrics = metric_set(rmse, mae),
    grid = 6,
    control = ctrl_grid
  )

boosted_res  <- 
  fit_resamples(
    boosted_workflow,
    resamples = train_folds,
    metrics = metric_set(rmse, mae),
    control = ctrl_res
  )

kali_rf_res  <- 
  fit_resamples(
    kali_rf_workflow,
    resamples = train_folds,
    metrics = metric_set(rmse, mae),
    control = ctrl_res
  )

lou_rf_res  <- 
  fit_resamples(
    lou_rf_workflow,
    resamples = train_folds,
    metrics = metric_set(rmse, mae),
    control = ctrl_res
  )

lou_knn_res  <- 
  fit_resamples(
    lou_knn_workflow,
    resamples = train_folds,
    metrics = metric_set(rmse, mae),
    control = ctrl_res
  )

pca_knn_res <- 
  tune_grid(
    pca_knn_workflow,
    resamples = train_folds,
    metrics = metric_set(rmse, mae),
    grid = 6,
    control = ctrl_grid
  )

# creating the stack:

stacks()

amazon_stack <-
  stacks() %>%
  add_candidates(tess_rf_res) %>%
  add_candidates(tess_rf_res2) %>%
  add_candidates(boosted_res) %>%
  add_candidates(kali_rf_res) %>%
  add_candidates(lou_rf_res) %>%
  add_candidates(lou_knn_res) %>%
  add_candidates(pca_knn_res)

amazon_stack # basically a tibble with extra attr
as_tibble(amazon_stack) # gives response value and predictions of each model

# blend predictions:

amazon_stack <-
  amazon_stack %>%
  blend_predictions() # gives stacking coeff to each model candidate 
                      # (nonzero coeff become members)

autoplot(amazon_stack) # to ensure correct trade off btween minimizing num of 
                       # members and optimizing performance
autoplot(amazon_stack, type = "members")  # diff view of above graph

# check stacking coefficient of each candidate:

collect_parameters(amazon_stack, "tess_rf_res")
collect_parameters(amazon_stack, "tess_rf_res2")
collect_parameters(amazon_stack, "boosted_res")
collect_parameters(amazon_stack, "kali_rf_res")
collect_parameters(amazon_stack, "lou_rf_res")
collect_parameters(amazon_stack, "lou_knn_res")
collect_parameters(amazon_stack, "pca_knn_res")

# now we fit the members (candidates with nonzero stacking coeff):

amazon_stack <-
  amazon_stack %>%
  fit_members()

# predicting test data:

stack_predictions <-
  test %>% 
  select(id) %>%
  bind_cols(predict(amazon_stack, new_data = test))

stack_predictions %>% 
  rename(log_total = .pred)

write_csv(stack_predictions, "stack_v4.csv")

# recipes + models + workflows of all stack candidates below:

### ------------------ ###  TESS RANDOM FOREST v4 ### ------------------- ###

tess_rf_recipe <- 
  recipe(log_total ~ ., data = train) %>%
  na.omit() %>%
  step_normalize(year) %>%
  step_num2factor(month, 
                  levels = c("Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), 
                  ordered = FALSE) %>%
  step_dummy(all_nominal_predictors())

tess_rf_model <-
  rand_forest(trees = 1961, min_n = 9) %>% # hyperparameters were previously tuned
  set_engine("ranger") %>%
  set_mode("regression")

tess_rf_workflow <- 
  workflow() %>%
  add_recipe(tess_rf_recipe) %>%
  add_model(tess_rf_model)

### ----------------- ### TESS RANDOM FOREST TUNABLE ### ------------------ ###

# same recipe as first rf:
tess_rf_recipe <- 
  recipe(log_total ~ ., data = train) %>%
  na.omit() %>%
  step_normalize(year) %>%
  step_num2factor(month, 
                  levels = c("Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), 
                  ordered = FALSE) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_zv(all_predictors())

# set trees and min_n to tune:
tess_rf_model2 <-
  rand_forest(mtry = 15, trees = tune(), min_n = tune()) %>% #mtry prev tuned
  set_engine("ranger") %>%
  set_mode("regression")

tess_rf_workflow2 <- 
  workflow() %>%
  add_recipe(tess_rf_recipe) %>%
  add_model(tess_rf_model2)

### ------------------- ### TESS BOOSTED TREE ### -------------------- ###

# the recipe that was proven (using workflow sets and cross validation) to work 
# best with the boosted model:
boosted_recipe <- 
  recipe(log_total ~ ., data = train) %>%
  na.omit() %>%
  step_normalize(year) %>%
  step_num2factor(month, 
                  levels = c("Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), 
                  ordered = FALSE) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_zv(all_predictors())

boosted_model <-
  boost_tree(min_n = 33) %>%  # hyperparameters were previously tuned
  set_engine("xgboost") %>%
  set_mode("regression")

boosted_workflow <- 
  workflow() %>%
  add_recipe(boosted_recipe) %>%
  add_model(boosted_model)

### ------------------- ### KALI RANDOM FOREST ### -------------------- ###

kali_rf_recipe <- recipe(log_total ~ . , data = train)

kali_rf_model <- 
  rand_forest() %>%
  set_engine("ranger") %>%
  set_mode("regression")

kali_rf_workflow <- 
  workflow() %>%
  add_recipe(kali_rf_recipe) %>%
  add_model(kali_rf_model)


### ------------------- ### LOU RANDOM FOREST ### -------------------- ###

lou_rf_recipe <- recipe(log_total ~ . , data = train)

lou_rf_model <-
  rand_forest(trees = 1000) %>% 
  set_engine("ranger") %>%
  set_mode("regression")

lou_rf_workflow <-
  workflow() %>%
  add_recipe(lou_rf_recipe) %>%
  add_model(lou_rf_model)

### ------------------------- ### LOU KNN ### ------------------------- ###

lou_knn_recipe <- 
  recipe(log_total ~. , data = train )

lou_knn_spec <- 
  nearest_neighbor(neighbors = 25, weight_func = "rectangular") %>% 3 # hyperparameters were previously tuned
  set_engine("kknn") %>%
  set_mode("regression")

lou_knn_workflow <- 
  workflow() %>%
  add_recipe(lou_knn_recipe) %>%
  add_model(lou_knn_spec)


### ----------------------- ### TESS PCA+KNN ### ---------------------- ###

# pca recipe that was proven (using workflow sets and cross validation) to work best with this tunable knn model:
pca_knn_recipe <-
  recipe(log_total ~ . , data = train) %>%
  na.omit() %>%
  step_normalize(year) %>%
  step_pca(all_numeric_predictors(), threshold = tune())

# tunable knn model:
pca_knn_model <- 
  nearest_neighbor(neighbors = tune(), weight_func = tune()) %>%
  set_engine("kknn") %>%
  set_mode("regression")

pca_knn_workflow <-
  workflow() %>%
  add_recipe(pca_knn_recipe) %>%
  add_model(pca_knn_model)
