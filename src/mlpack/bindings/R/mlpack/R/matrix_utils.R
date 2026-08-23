# matrix_utils.R: utilities for matrix conversion
#
# This file defines the to_matrix() function, which can be used to convert
# data.frame or other types of matrix-like objects to matrix for use in
# mlpack bindings(IO_SetParamMat/UMat).
#
# This file also defines the to_matirx_with_info() function, which can be used
# to construct dataset information vector from the given dataset for use in
# mlpack bindings(IO_SetParamMatWithInfo).
#
#
# mlpack is free software; you may redistribute it and/or modify it under the
# terms of the 3-clause BSD license.  You should have received a copy of the
# 3-clause BSD license along with mlpack.  If not, see
# http://www.opensource.org/licenses/BSD-3-Clause for more information.

# Given some matrix-like x (which should be either a matrix or
# data.frame or vector), convert it into a matrix (or leave as is).
to_matrix <- function(x) {
  if (!is.matrix(x) && !is.vector(x) && !is.data.frame(x)) {
    stop("Input must be either a 'matrix' or 'vector' or 'data.frame' not '",
         class(x)[1], "'.", call. = FALSE)
  }
  if (is.matrix(x)) {
    return(x)
  } else if (is.vector(x)) {
    if (length(x) == 1 || is.character(x)) {
      stop("Scalar (i.e. length one) or character arguments not admissible as input.",
        call. = FALSE)
    }
    return(as.matrix(x))
  } else if (is.data.frame(x)) {
    y <- data.matrix(x) # requires R 4.0.0 for factor conversion.
    return(y)
  }
  ## no fallback as we allow only for matrix, vector or data.frame
}

# Determine column classes
mark_categorical_variable = function(x) {
  d <- sapply(x, class) %in% c("factor", "character", "logical")
  d
}

# Given some matrix-like x (which should be either a matrix or
# data.frame), convert it into a matrix.
to_matrix_with_info <- function(x) {

  # Handle transformation
  transformed_x <- to_matrix(x)

  # Figure out categoricals
  info <- mark_categorical_variable(x)

  # Return needed data.
  return(list("info" = info, "data" = transformed_x))
}
