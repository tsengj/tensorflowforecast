# https://mlr.mlr-org.com/articles/tutorial/usecase_regression.html

data(BostonHousing, package = "mlbench")

BostonHousing$chas <- as.integer(levels(BostonHousing$chas))[BostonHousing$chas]

library('mlr')
library('parallel')
library("parallelMap")

# ---- define learning tasks -------
regr.task = makeRegrTask(id = "Boston Housing", data = BostonHousing, target = "medv")
regr.task

#accessing tasks data
# https://mlr.mlr-org.com/articles/tutorial/task.html#accessing-a-learning-task
getTaskDesc(regr.task)

# Accessing the data set in tasks
str(getTaskData(regr.task))

# ---- tune Hyperparameters -------- 

set.seed(1234)

# Define a search space for each learner'S parameter
ps_ksvm = makeParamSet(
  makeNumericParam("sigma", lower = -12, upper = 12, trafo = function(x) 2^x)
)

ps_rf = makeParamSet(
  makeIntegerParam("num.trees", lower = 1L, upper = 200L)
)

ps_xgb = makeParamSet(
  makeIntegerParam("nrounds",lower=5,upper=50),
  makeIntegerParam("max_depth",lower=3,upper=15),
  # makeNumericParam("lambda",lower=0.55,upper=0.60),
  # makeNumericParam("gamma",lower=0,upper=5),
  makeNumericParam("eta", lower = 0.01, upper = 1),
  makeNumericParam("subsample", lower = 0, upper = 1),
  makeNumericParam("min_child_weight",lower=1,upper=10),
  makeNumericParam("colsample_bytree",lower = 0.1,upper = 1)
)

# Choose a resampling strategy
rdesc = makeResampleDesc("CV", iters = 5L)

# Choose a performance measure
meas = rmse

# Choose a tuning method
# https://mlr.mlr-org.com/articles/tutorial/tune.html
ctrl = makeTuneControlRandom(maxit = 30L) #makeTuneControlGrid  makeTuneControlRandom

# Make tuning wrappers
tuned.lm = makeLearner("regr.lm")
tuned.ksvm = makeTuneWrapper(learner = "regr.ksvm", resampling = rdesc, measures = meas,
                             par.set = ps_ksvm, control = ctrl, show.info = FALSE)
tuned.rf = makeTuneWrapper(learner = "regr.ranger", resampling = rdesc, measures = meas,
                           par.set = ps_rf, control = ctrl, show.info = FALSE)
tuned.xgb = makeTuneWrapper(learner = "regr.xgboost", resampling = rdesc, measures = meas,
                           par.set = ps_xgb, control = ctrl, show.info = FALSE)

# -------- Benchmark experiements -----------
# Four learners to be compared
lrns = list(tuned.lm, tuned.ksvm, tuned.rf, tuned.xgb)

#setup Parallelization 
parallelStart(mode = "socket", #multicore #socket
              cpus = detectCores(),
              # level = "mlr.tuneParams",
              mc.set.seed = TRUE)

# Conduct the benchmark experiment
bmr = benchmark(learners = lrns, 
                tasks = regr.task,
                resamplings = rdesc,
                measures = rmse, 
                keep.extract = T,
                models = F,
                show.info = F)

parallelStop()

# -------- Evaluate performance ----------
getBMRAggrPerformances(bmr)
rmat = convertBMRToRankMatrix(bmr)
print(rmat)
plotBMRSummary(bmr)
plotBMRBoxplots(bmr)
# getBMRTuneResults(bmr)

# -------- Tuning for hyper parameters-------
# https://mlr.mlr-org.com/articles/tutorial/nested_resampling.html
# https://www.r-spatial.org/r/2018/03/03/spatial-modeling-mlr.html#specification-of-the-learner

# res <-
#   resample(
#     tuned.xgb,
#     regr.task,
#     resampling = rdesc,
#     extract = getTuneResult, #getFeatSelResult, getTuneResult
#     show.info = TRUE,
#     measures = meas
#   )

# res$extract
# getNestedTuneResultsX(res)


# ------ Extract HyperParameters -----
bmr_hp <- getBMRTuneResults(bmr)
bmr_hp <- bmr_hp$bh$regr.xgboost.tuned[[1]]

res$extract

# create learner from tuned model
lrn_tune = setHyperPars(makeLearner("regr.xgboost"),par.vals = bmr_hp$x)

#train model
m1 <- train(learner = lrn_tune,task = regr.task)

# -------- Make prediction ---------------
xgpred <- predict(m1,testtask)
