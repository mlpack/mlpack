## Test that refactored binding and their S3 dispatched `predict()` generic method work.
## As this tests the _binding_ aspect we are not honing in on method prediction results
## but essentially check that we get the expected range of predictions or probabilities.

suppressMessages({
    library(stats)
    library(mlpack)
    library(tinytest)
    library(datasets)
})

## Prepare data. We use 'iris' as it is present, and multiclass.

data(iris)
X <- as.matrix(iris[,1:4])
y <- as.matrix(as.integer(iris[,5]))    # mlpack prefers {0, 1, 2}
X2 <- X[1:100,]                         # two class subset
y2 <- y[1:100,,drop=FALSE]              # idem

## adaboost
expect_silent(m <- adaboost_train(training=X, labels=y))
expect_silent(pv <- predict(m, X))
expect_true(all(c(1,2,3) %in% pv))
expect_silent(pp <- predict(m, X, type="probabilities"))
expect_equal(min(pp), 0)
expect_true(max(pp) > 0.5)

## logistic_regression
expect_silent(m <- logistic_regression_train(training=X2, labels=y2))
expect_silent(p <- predict(m, X2))
expect_true(all(c(1,2) %in% pv))
expect_silent(pp <- predict(m, X, type="probabilities"))
expect_equal(min(pp), 0)
expect_true(max(pp) > 0.5)

## linear_regression
expect_silent(m <- linear_regression_train(training=X, training_responses=y))
expect_silent(p <- predict(m, X))
expect_equal(mean(p-y), 0) 	# regression residuals are unbiased

## lars_regression
expect_silent(m <- lars_train(input=X, responses=y))
expect_silent(p <- predict(m, X))
expect_equal(mean(p-y), 0) 	# regression residuals are unbiased

## bayesian_linear_regression
expect_silent(m <- bayesian_linear_regression_train(input=X, responses=y))
expect_silent(p <- predict(m, X))
expect_equal(mean(p-y), 0, tolerance = 1e-2) # biased regression, near zero

## random_forest
expect_silent(m <- random_forest_train(training=X, labels=y))
expect_silent(pv <- predict(m, X))
expect_true(all(c(1,2,3) %in% pv))
expect_silent(pp <- predict(m, X, type="probabilities"))
expect_equal(min(pp), 0)
expect_true(max(pp) > 0.5)

## nbc
expect_silent(m <- nbc_train(training=X, labels=y))
expect_silent(pv <- predict(m, X))
expect_true(all(c(1,2,3) %in% pv))
expect_silent(pp <- predict(m, X, type="probabilities"))
expect_equal(min(pp), 0)
expect_true(max(pp) > 0.5)

## perceptron
expect_silent(m <- perceptron_train(training=X, labels=y))
expect_silent(pv <- predict(m, X))
expect_true(all(c(1,2,3) %in% pv))

## linear_svm
expect_silent(m <- linear_svm_train(training=X, labels=y, optimizer="psgd"))
expect_silent(pv <- predict(m, X))
expect_true(all(c(1,2,3) %in% pv))
expect_silent(pp <- predict(m, X, type="scores"))
expect_equal(ncol(pp), 3L)

## softmax_regression
expect_silent(m <- softmax_regression_train(training=X, labels=y))
expect_silent(pv <- predict(m, X))
expect_true(all(c(1,2,3) %in% pv))
expect_silent(pp <- predict(m, X, type="probabilities"))
expect_equal(min(pp), 0)
expect_true(max(pp) > 0.5)
