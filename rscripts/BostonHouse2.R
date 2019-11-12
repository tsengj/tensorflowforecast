library(caret)
library(keras)

dataset <- dataset_boston_housing()
c(c(train_data, train_targets), c(test_data, test_targets)) %<-% dataset

feature_names <- c('CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT')

train_df <- as.data.frame(train_data)
names(train_df) <- feature_names
test_df <- as.data.frame(test_data)
names(test_df) <- feature_names

train_targets <- as.numeric(train_targets)
test_targets <- as.numeric(test_targets)

#define a hyperparameter grid. In this example single setting of hyperparameter values is defined
hyperparameters <- expand.grid(
  dropout = 0.5,
  batch_size = 16,
  activation = "relu",
  decay = c(0.1),
  lr = 0.2,
  size = 10,
  rho = 0.1
)

#The trainControl function is used to specify the resampling method. In this example method is given the value "none" which indicates that there is no tuning.
ctrl_none <- trainControl(method="none")

#The train function is used to fit a model. Currently caret supports 237 different models and the type of model is specified using the method argument.
model <- train(train_df, train_targets, 
               method = "mlpKerasDropout", 
               trControl = ctrl_none,
               preProc = c("center", "scale"),
               verbose = 1,
               tuneGrid = hyperparameters,
               epochs = 10)

preds <- predict(model, test_df)

hyperparameters <- expand.grid(
  dropout = c(0.2, 0.5),
  batch_size = c(16, 32, 48),
  activation = c("tanh","relu"),
  decay = c(0.01, 0.02, 0.1),
  lr = c(0.01, 0.02),
  size = c(5, 10, 15),
  rho = c(0.1, 0.2)
)


fitControl <- trainControl(
  method="repeatedcv",
  number = 10,
  repeats = 5
)

set.seed(42)

large_grid <- expand.grid(
  dropout = seq(from=0.2, to=0.5, by=0.1),
  batch_size = seq(from=16, to=64, by=16),
  activation = c("tanh","relu"),
  decay = seq(from=0.5, to=0.9, by=0.1),
  lr = seq(from=0.01, to=0.2, by=0.01),
  size = seq(from=5, to=20, by=5),
  rho = seq(from=0.1, to=0.8, by=0.1)
)

#In practice you should obtain at least a sample of size equal to 60
grid_index <- sample(nrow(large_grid),
                     size=5)

hyperparameters <- large_grid[grid_index,]


fit_control <- trainControl(method = "cv",
                            number = 5,
                            search = "grid")

model <- train(train_df, train_targets, 
               method = "mlpKerasDropout", 
               trControl = fit_control,
               preProc = c("center", "scale"),
               verbose = TRUE,
               tuneGrid = hyperparameters,
               epochs = 20)



set.seed(42)

fit_control <- trainControl(method = "cv", 
                            number = 3,
                            search = "random")

model <- train(train_df, train_targets, 
               method = "mlpKerasDropout", 
               trControl = fit_control,
               preProc = c("center", "scale"),
               verbose = FALSE,
               tuneLength = 30,
               epochs = 30)

results <- model$results
min_rmse_index <- which.min(results$RMSE)
rmse_min <- results[min_rmse_index,]$RMSE

#plot 
plot(1:nrow(results), results$RMSE, xlab="Trial", ylab="RMSE", cex=results$size / max(results$size), col="red", ylim=c(1, 30))
text(1:nrow(results), results$RMSE, results$activation, offset=.3, pos=3, cex=0.8)
rect(min_rmse_index-1, rmse_min-1, min_rmse_index+2, rmse_min+2)

preds <- predict(model, test_df)


set.seed(42)

fit_control <- trainControl(method = "adaptive_cv", 
                            adaptive = list(min=2, 
                                            alpha=0.05, 
                                            method="BT", 
                                            complete=FALSE),
                            search = "random")

model <- train(train_df, train_targets, 
               method = "mlpKerasDropout", 
               trControl = fit_control,
               preProc = c("center", "scale"),
               verbose = TRUE,
               tuneLength = 20,
               epochs = 50)

preds <- predict(model, test_df)

mae <- function(y, t) {
  100 * mean(abs(y-t) / t)
}

