
#' @export
print.mlpack_model_binding <- function(x, ...) {
    cat("<mlpack object of class '", class(x)[1], "'>\n", sep="")
}

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

#' @export
predict.mlpack_linear_regression <- function(object, newdata, ...) {
    if (missing(newdata)) {
        stop("Need 'newdata'")
    }
    res <- linear_regression_predict(input_model=object, newdata, ...)
    res
}

#' @export
predict.mlpack_lars <- function(object, newdata, ...) {
    if (missing(newdata)) {
        stop("Need 'newdata'")
    }
    res <- lars_predict(input_model=object, newdata, ...)
    res
}
