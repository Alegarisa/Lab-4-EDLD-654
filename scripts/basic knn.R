library(tidyverse)
library(tidymodels)

# prepare for parallel processing
all_cores <- parallel::detectCores(logical = FALSE)

library(doParallel)
cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)
foreach::getDoParWorkers()
clusterEvalQ(cl, {library(tidymodels)})

# read in the train data
full_train <- read_csv("data/train.csv") %>%
  mutate(classification = factor(classification,
                                 levels = 1:4,
                                 labels = c("far below", "below", "meets", "exceeds"),
                                 ordered = TRUE))

# select a proportion of the train data to work locally
# full_train <- full_train %>% 
#   sample_frac(0.005)

full_train <- full_train %>%
  sample_frac(0.2) 

# initial split object
splt <- initial_split(full_train)
train <- training(splt)

# k-fold cross validation
train_cv <- vfold_cv(train)

# preprocessing -- basic recipe
rec <- recipe(classification ~ id + ncessch + ethnic_cd + enrl_grd + econ_dsvntg + lang_cd + lat + lon, train) %>%
  step_mutate(enrl_grd = factor(enrl_grd)) %>%
  update_role(contains("id"), ncessch, new_role = "id_vars") %>%
  step_unknown(all_nominal(), -all_outcomes()) %>%
  step_zv(all_predictors(), -starts_with("lang_cd")) %>%
  step_medianimpute(all_numeric(), -has_role("id_vars")) %>%
  step_center(all_numeric(), -has_role("id_vars")) %>%
  step_scale(all_numeric(), -has_role("id_vars")) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_nzv(all_predictors(), -starts_with("lang_cd"))

# Up until here, everything will be the same for al model fits. Copy & paste the above for each new R script with subsequent models. 


##### Basic knn model #####

# model object
knn_m1 <- nearest_neighbor() %>%
  set_engine("kknn") %>%
  set_mode("classification")

# Fitting basic knn model
knn_m1_fit <- fit_resamples(
  knn_m1,
  rec,
  train_cv,
  control_resamples(save_pred = TRUE)
)

# saving fit resample object as .Rds file 
saveRDS(knn_m1_fit, "models/knn_m1_fit.Rds")






