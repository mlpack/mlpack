#' mlpack
#'
#' mlpack is a fast, flexible machine learning library, written in C++, that
#' aims to provide fast, extensible implementations of cutting-edge machine
#' learning algorithms. mlpack provides these algorithms as simple command-line
#' programs, C++ classes and bindings for Python, Julia, Go and R which can
#'
#' @section Compile-time configuration:
#' Three different \code{#define} variables can be used to turn on optional
#' functionality that is off by default. These are \code{MLPACK_R_ENABLE_STB}
#' and \code{MLPACK_R_ENABLE_DR_LIBS} to enable the \sQuote{STB} and
#' \sQuote{DR_LIBS} libraries for image and audio processing, respectively, and
#' \code{MLPACK_DISABLE_HTTPLIB} to enable \sQuote{HTTPLIB} to permit data-loading
#' from remote URLs (which R has via \code{libcurl}).
#'
#' @name mlpack
#' @aliases mlpack-package
#' @importFrom Rcpp evalCpp
#' @useDynLib mlpack, .registration = TRUE
"_PACKAGE"
