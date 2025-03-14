### ----------------------------------------------------------------------- ###
# Comparing Models for Predicting Amazon Customer Order Totals 
# Author: Tess Wellington
# Contributors: Kali Olmstead, Lou Qian, Romy Lou, Neha Jonnalagadda
# Course: STATS 101C @ UCLA
# Date: July 2024
# Description:
# This script compares the predictive performance of multiple models using
# 10-fold cross-validation and evaluating the models' RMSE values. I created
# this comparison script which uses models created by my team members and myself.
# All models predict total order value (log_total) for Amazon customers 
# based on demographic and purchase history data. The dataset consists of about
# 5,000 Amazon customers from 2018-2022, divided into training and test sets.
### ----------------------------------------------------------------------- ###
# Models Used in Comparison:
# - Model 1: Random Forest (Tess Wellington)
# - Model 2: KNN (Lou Qian)
# - Model 3: Decision Tree (Kali Olmstead)
# - Model 4: Boosted Tree (Neha Jonnalagadda)
# - Model 5: GLM (Romy Lou)
### ----------------------------------------------------------------------- ###


### -------------------- ### LIBRARIES / FILES ### ------------------------ ###

library(tidyverse)
library(tidymodels)
library(stacks)
library(ggplot2)
library(dplyr)
library(caret)
library(ranger)  # for rf models
library(glmnet)  # for glm models
library(kknn)    # for knn models
library(xgboost) # for boosted tree models

amazon_order_train <- read_csv("amazon_order_details_train.csv")
customer_info_train <- read_csv("customer_info_train.csv")
train <- read_csv("train.csv")
train <- 
  train %>%
  select(-order_totals)

amazon_order_test <- read_csv("amazon_order_details_test.csv")
customer_info_test <- read_csv("customer_info_test.csv")
test <- read_csv("test.csv")

### -------------------- ### MEASURING PERFORMANCE ### ---------------------- ###

# creating 10-fold cross-validation set, stratifying on `log_total`

set.seed(101)
folds <- vfold_cv(train, v=10, strat=log_total)
keep_pred <- control_resamples(save_pred = TRUE, save_workflow = TRUE)

# fitting the candidates to the resamples and collecting metrics: 
# NOTE: the workflow of all candidate models is below along with the code to create them


# tess_rf

tess_rf_fit <-
  tess_rf_workflow %>%
  fit_resamples(resamples = folds, control = keep_pred)

tess_rf_metrics <- collect_metrics(tess_rf_fit)

# lou_knn

lou_knn_fit <-
  lou_knn_workflow %>%
  fit_resamples(resamples = folds, control = keep_pred)

lou_knn_metrics <- collect_metrics(lou_knn_fit)

# kali_tree

kali_tree_fit <-
  kali_tree_workflow %>%
  fit_resamples(resamples = folds, control = keep_pred)

kali_tree_metrics <- collect_metrics(kali_tree_fit)

# neha_boosted (NOTE: RMSE found by printing 'gb_model' since this model was created differently)

boosted_fit <-
  boosted_workflow %>%
  fit_resamples(resamples = folds, control = keep_pred)

boosted_metrics <- collect_metrics(boosted_fit)

tess_boosted_metrics <- boosted_metrics

# romy_glm

romy_glm_fit <-
  romy_glm_workflow %>%
  fit_resamples(resamples = folds, control = keep_pred)

romy_glm_metrics <- collect_metrics(romy_glm_fit)

# amazon_stack

### consolidating metrics into one tibble:

names <- tibble(model_id = c("lou_knn", "lou_knn", 
                          "kali_tree", "kali_tree",
                          "tess_rf", "tess_rf",
                          "tess_boosted", "tess_boosted",
                          "romy_glm", "romy_glm"))
candidate_metrics <- 
  rbind(lou_knn_metrics, kali_tree_metrics, tess_rf_metrics, tess_boosted_metrics, romy_glm_metrics)

candidate_metrics <-
  cbind(names, candidate_metrics) %>%
  filter(.metric == "rmse")

candidate_metrics

all_models <- as_workflow_set(romy_glm = romy_glm_fit, lou_knn = lou_knn_fit, 
                kali_tree = kali_tree_fit, tess_rf = tess_rf_fit, boosted = boosted_fit)

autoplot(all_models)

### -------------------------- ### LOU'S KNN ### -------------------------- ###

lou_knn_recipe <- 
  recipe(log_total ~. , data = train )

# basic KNN model with k = 25
lou_knn_spec <- 
  nearest_neighbor(neighbors = 25, weight_func = "rectangular") %>%  # hyperparameters were tuned
  set_engine("kknn") %>%
  set_mode("regression")

lou_knn_workflow <- 
  workflow() %>%
  add_recipe(lou_knn_recipe) %>%
  add_model(lou_knn_spec)

lou_knn_train_fit <- lou_knn_workflow %>%
  fit(data = train)

lou_knn_pred <- lou_knn_fit %>%
  predict(new_data = test)

head(lou_knn_pred)


write_csv(test_results, "lou_KNN_submission4.csv")


### --------------------- ### KALI'S DECISION TREE ### -------------------- ###

cor_relation <- train %>%
  select_if(is.numeric) %>%
  cor()

cor_relation[abs(cor_relation) < 0.7] <- NA


low_cor <- c('year','month','count_over10', 'count_howmany3','count_howmany4','count_1824','count_5564','count_65up','count_und25k', 'count_2549k' ,'count_150kup' ,'count_lessHS')

cor_train <-
  train %>%
  select(!all_of(low_cor))


tree4 <-
  decision_tree(cost_complexity = 0.01, tree_depth = 4) %>%
  set_engine("rpart") %>%
  set_mode("regression")

dt_wflow4 <- workflow() %>%
  add_formula(log_total ~ .) %>%
  add_model(tree4)

tree_fit4 <-
  dt_wflow4 %>%
  fit(data = cor_train)


predictions <- tree_fit4 %>%
  predict(new_data = test)

dt4 <- test %>%
  select(id) %>%
  bind_cols(predictions)

dt4 <- rename(dt4, log_total = .pred)
write_csv(dt4, "decisiontree4.csv" )
dt4

tree_fit4 %>%
  extract_fit_engine() %>%
  summary() #Looking at the rpart summary results gives us an idea of the range of complexity parameter values to try: from around 0.01 to about 0.6


tree4_tune <-
  decision_tree(cost_complexity = tune(),
                tree_depth = tune()) %>%
  set_engine("rpart") %>%
  set_mode("regression")

tree_grid4 <-
  grid_regular(
    cost_complexity(range = c(-2, -0.2),
                    trans = log10_trans()),
    tree_depth(),
    levels = 10)

tree_grid4

tree_folds4 <- vfold_cv(cor_train)


tree4_wf <-
  workflow() %>%
  add_model(tree4_tune) %>%
  add_formula(log_total ~ .)


tree4_res <-
  tree4_wf %>%
  tune_grid(
    resamples = tree_folds4,
    grid = tree_grid4
  )

best_tree <- tree4_res %>%
  select_best(metric = "rmse")

# tree4_res %>%
#   collect_metrics() %>%
#   filter(.metric == "rmse") %>%
#   arrange(mean)

final_treewf4 <- tree4_wf %>%
  finalize_workflow(best_tree)

# --------------------------------
# (tess) for purposes of comparing:

kali_tree_workflow <- final_treewf4

# --------------------------------

final_fit4 <- final_treewf4 %>%
  fit(cor_train)

predictions <- final_fit4 %>%
  predict(new_data = test)

dt4 <- test %>%
  select(id) %>%
  bind_cols(predictions)

dt4 <- rename(dt4, log_total = .pred)
write_csv(dt4, "decisiontree4.csv" )

### --------------------- ### TESS RANDOM FOREST v4 ### -------------------- ###

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
  rand_forest(trees = 1961, min_n = 9) %>% # hyperparameters were tuned
  set_engine("ranger") %>%
  set_mode("regression")

tess_rf_workflow <- 
  workflow() %>%
  add_recipe(tess_rf_recipe) %>%
  add_model(tess_rf_model)

tree_folds <- vfold_cv(train, v=10, strat=log_total)
keep_pred <- control_resamples(save_pred = TRUE, save_workflow = TRUE)

tess_rf_fit <-
  tess_rf_workflow %>%
  fit_resamples(resamples = tree_folds, control = keep_pred)

tess_rf_fit <- 
  tess_rf_workflow %>% 
  fit(data = train)

tess_rf_predictions <-
  test %>% 
  select(id) %>%
  bind_cols(predict(tess_rf_fit, new_data = test))

tess_rf_predictions %>% 
  rename(log_total = .pred)

write_csv(tess_rf_predictions, "randomforest_v4.csv")

### ---------------------- ### TESS BOOSTED TREE ### ----------------------- ###

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

tess_boosted_model <-
  boost_tree(min_n = 33) %>%  # hyperparameters were previously tuned
  set_engine("xgboost") %>%
  set_mode("regression")

tess_boosted_workflow <- 
  workflow() %>%
  add_recipe(tess_boosted_recipe) %>%
  add_model(tess_boosted_model)

### ------------------------ ### ROMY GLM ### ------------------------- ###

lm_tune <- linear_reg(penalty = tune(),
                      mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("regression")

lm_grid <- grid_regular(penalty(),
                        mixture(),
                        levels = 10)

lm_recipe <- 
  recipe(log_total ~ ., train) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())
  

lm_wflow <- workflow() %>%
  add_recipe(lm_recipe) %>%
  add_model(lm_tune)

lm_res <- lm_wflow %>%
  tune_grid(resamples = folds,
            grid = lm_grid,
            metrics = metric_set(rmse))

best_lm <- lm_res %>%
  select_best(metric = "rmse")

final_lm_wflow <- lm_wflow %>% finalize_workflow(best_lm)

# --------------------------------
# (tess) for purposes of comparing:

romy_glm_workflow <- final_lm_wflow

# --------------------------------

final_lm_fit <- final_lm_wflow %>% fit(train)

predictions <- final_lm_fit %>% predict(new_data = test)
predictions <- bind_cols(test %>% select(id), predictions)
colnames(predictions)[1] <- "log_total"
write_csv(predictions, "lm_prediction.csv")

### ------------------------ ### TESS'S STACK v4 ### ------------------------- ###

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
  blend_predictions() # gives stacking coeff to each model candidate (nonzero coeff become members)

autoplot(amazon_stack) # to ensure correct trade off btween minimizing num of members and optimizing performance
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

### ------------------ ### TESS RANDOM FOREST TUNABLE ### ------------------- ###

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
  rand_forest(mtry = 15, trees = tune(), min_n = tune()) %>% #mtry was tuned previously
  set_engine("ranger") %>%
  set_mode("regression")

tess_rf_workflow2 <- 
  workflow() %>%
  add_recipe(tess_rf_recipe) %>%
  add_model(tess_rf_model2)

### ------------------- ### TESS BOOSTED TREE ### -------------------- ###

# the recipe that was proven (using workflow sets and cross validation) to work best with the boosted model:
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
  nearest_neighbor(neighbors = 25, weight_func = "rectangular") %>% # hyperparameters were previously tuned
set_engine("kknn") %>%
  set_mode("regression")

lou_knn_workflow <- 
  workflow() %>%
  add_recipe(lou_knn_recipe) %>%
  add_model(lou_knn_spec)

### ----------------------- ### TESS PCA+KNN ### ---------------------- ###

# pca recipe that was proven (using workflow sets and cross validation) 
# to work best with this tunable knn model:
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
