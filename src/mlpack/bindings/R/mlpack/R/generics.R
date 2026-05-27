
#' @export
print.mlpack_model_binding <- function(x, ...) {
    cat("<mlpack object of class '", class(x)[1], "'>\n", sep="")
}

#' @rdname adaboost_classify
#' @param object An instantiated model object for which prediction is desired
#' @param newdata A test data set
#' @param type A character value selection predictions or probabilities
#' @param ... Additional optional arguments affecting the prediction
#' @export
predict.mlpack_adaboost <- function(object, newdata, type=c("predictions", "probabilities"), ...) {
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

#' @rdname logistic_regression_classify
#' @param object An instantiated model object for which prediction is desired
#' @param newdata A test data set
#' @param type A character value selection predictions or probabilities
#' @param ... Additional optional arguments affecting the prediction
#' @export
predict.mlpack_logistic_regression <- function(object, newdata, type=c("predictions", "probabilities"), ...) {
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

#' @rdname linear_regression_predict
#' @param object An instantiated model object for which prediction is desired
#' @param newdata A test data set
#' @param ... Additional optional arguments affecting the prediction
#' @export
predict.mlpack_linear_regression <- function(object, newdata, ...) {
    if (missing(newdata)) {
        stop("Need 'newdata'")
    }
    res <- linear_regression_predict(input_model=object, newdata, ...)
    res
}

#' @rdname lars_predict
#' @param object An instantiated model object for which prediction is desired
#' @param newdata A test data set
#' @param ... Additional optional arguments affecting the prediction
#' @export
predict.mlpack_lars <- function(object, newdata, ...) {
    if (missing(newdata)) {
        stop("Need 'newdata'")
    }
    res <- lars_predict(input_model=object, newdata, ...)
    res
}

#' @rdname bayesian_linear_regression
#' @param object An instantiated model object for which prediction is desired
#' @param newdata A test data set
#' @param stddevs A flag selecting standard deviation estimation returned along with
#' point estimates
#' @param ... Additional optional arguments affecting the prediction
#' @export
predict.mlpack_bayesian_linear_regression <-
    function(object, newdata, stddevs=FALSE, ...) {

    if (missing(newdata)) {
        stop("Need 'newdata'")
    }
    res <- bayesian_linear_regression_predict(input_model=object,
                                              newdata, stddevs, ...)
    # For prediction return the single column, otherwise return both
    if (is.matrix(res) && ncol(res) == 1)
        res <- res[,1,drop=TRUE]

    res
}

#' @rdname random_forest
#' @param object An instantiated model object for which prediction is desired
#' @param newdata A test data set
#' @param stddevs A flag selecting standard deviation estimation returned along with
#' point estimates
#' @param ... Additional optional arguments affecting the prediction
#' @export
predict.mlpack_random_forest <- function(object, newdata, type=c("predictions", "probabilities"), ...) {
    if (missing(newdata)) {
        stop("Need 'newdata'")
    }
    type <- match.arg(type)
    if (type == "predictions") {
        res <- random_forest_classify(input_model=object, newdata, ...)
    } else {
        res <- random_forest_probabilities(input_model=object, newdata, ...)
    }
}
