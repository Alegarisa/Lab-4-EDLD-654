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
full_train <- read_csv("data/train.csv",
                       col_types = cols(.default = col_guess(), 
                                        calc_admn_cd = col_character())) %>%
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

# prepped_rec <- prep(rec)
# prepped_rec

##### Tuned knn model #####

# model object
knn_m2 <- nearest_neighbor() %>%
  set_engine("kknn") %>%
  set_mode("classification") %>%
  set_args(neighbors = tune(),
           dist_power = tune())

# Non-regular, space filling design grid
knn_params <- parameters(neighbors(c(1, 20)), dist_power())
knn_sfd <- grid_max_entropy(knn_params, size = 25)

# knn_sfd %>%
#   ggplot(aes(neighbors, dist_power)) +
#   geom_point()

# Fitting tuned knn model
knn_m2_fit <- tune::tune_grid(
  knn_m2,
  preprocessor = rec,
  resamples = train_cv,
  grid = knn_sfd,
  control = tune::control_resamples(save_pred = TRUE))

# saving tuned fit object as .Rds file 
saveRDS(knn_m2_fit, "models/knn-m2-tuning-fit.Rds")
