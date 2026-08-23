## Default print method for returned mlpack object, a list with an external pointer
## to the trained model in the first element

#' @export
print.mlpack_model_binding <- function(x, ...) {
    cat("<mlpack object of class '", class(x)[1], "'>\n", sep="")
}
