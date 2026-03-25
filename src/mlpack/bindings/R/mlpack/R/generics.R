
#' @export
print.mlpack_model_binding <- function(x, ...) {
    cat("<mlpack object of class '", class(x)[1], "'>\n", sep="")
}

#' @rdname adaboost_train
#' @examples
#' data(iris)
#' pp <- preprocess_split(input=iris[,1:4], input_label=as.matrix(as.integer(iris[,5])))
#' trn <- adaboost_train(training = pp$training, labels = pp$training_labels)
#' trn
#' tst <- predict(trn, pp$test)
#' table(tst)
#' table(levels(iris[,5])[tst])
#' @export
predict.mlpack_adaboost_train <- function(object, newdata, type=c("predictions", "probabilities"), ...) {
    if (missing(newdata)) {
        stop("Need 'newdata'")
    }
    type <- match.arg(type)
    if (type == "predictions") {
        res <- adaboost_classify(input_model=object, newdata, ...)
        res[,1,drop=TRUE]
    } else {
        res <- adaboost_probabilities(input_model=object, newdata, ...)
        res
    }
}

#' @rdname logistic_regression_train
#' @examples
#' data(iris)
#' pp <- preprocess_split(input=iris[,1:4], input_label=as.matrix(as.integer(iris[,5])))
#' trn <- logistic_regression_train(training = pp$training,
#'                                  labels = as.matrix(as.integer(pp$training_labels == 2) + 1))
#' trn
#' tst <- predict(trn, pp$test)
#' table(tst)
#' table(levels(iris[,5])[tst])
#' @export
predict.mlpack_logistic_regression_train <- function(object, newdata, type=c("predictions", "probabilities"), ...) {
    if (missing(newdata)) {
        stop("Need 'newdata'")
    }
    type <- match.arg(type)
    if (type == "predictions") {
        res <- logistic_regression_classify(input_model=object, newdata, ...)
        res[,1,drop=TRUE]
    } else {
        res <- logistic_regression_probabilities(input_model=object, newdata, ...)
        res
    }
}

#' @rdname linear_regression_train
#' @examples
#' data(mtcars)
#' pp <- preprocess_split(input=mtcars[,-1], input_label=as.matrix(as.integer(mtcars[,1])))
#' trn <- linear_regression_train(training = pp$training, training_responses = pp$training_labels)
#' trn
#' tst <- predict(trn, pp$test)
#' tst
#' @export
predict.mlpack_linear_regression_train <- function(object, newdata, ...) {
    if (missing(newdata)) {
        stop("Need 'newdata'")
    }
    res <- linear_regression_predict(input_model=object, newdata, ...)
    res
}

#' @rdname lars_train
#' @examples
#' data(mtcars)
#' pp <- preprocess_split(input=mtcars[,-1], input_label=as.matrix(as.integer(mtcars[,1])))
#' trn <- lars_train(input = pp$training, responses = pp$training_labels)
#' trn
#' tst <- predict(trn, t(pp$test))
#' tst
#' @export
predict.mlpack_lars_train <- function(object, newdata, ...) {
    if (missing(newdata)) {
        stop("Need 'newdata'")
    }
    res <- lars_predict(input_model=object, newdata, ...)
    res
}
