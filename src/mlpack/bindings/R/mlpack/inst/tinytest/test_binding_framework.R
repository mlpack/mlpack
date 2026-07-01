## Test that when we run the binding correctly (with correct input parameters),
## we get the expected output.

suppressMessages({
    library(stats)
    library(mlpack)
    library(tinytest)
})

# Basic functionality test.
output <- test_r_binding(4.0, 12, "hello", flag1=TRUE)
expect_true(output$double_out == 5.0)
expect_true(output$int_out == 13)
expect_true(output$string_out == "hello2")

# If we forget the mandatory flag, we should get wrong results.
output <- test_r_binding(4.0, 12, "hello")
expect_true(output$double_out != 5.0)
expect_true(output$int_out != 13)
expect_true(output$string_out != "hello2")

# If we give the wrong string, we should get wrong results.
output <- test_r_binding(4.0, 12, "goodbye", flag1=TRUE)
expect_true(output$string_out != "hello2")

# If we give the wrong int, we should get wrong results.
output <- test_r_binding(4.0, 15, "hello", flag1=TRUE)
expect_true(output$int_out != 13)

# If we give the wrong double, we should get wrong results.
output <- test_r_binding(2.0, 12, "hello", flag1=TRUE)
expect_true(output$double_out != 5.0)

# If we give the second flag, this should fail.
output <- test_r_binding(4.0, 12, "hello", flag1=TRUE, flag2=TRUE)
expect_true(output$double_out != 5.0)
expect_true(output$int_out != 13)
expect_true(output$string_out != "hello2")

# The matrix we pass in, we should get back with the third dimension doubled and
# the fifth forgotten.
x <- matrix(rexp(500, rate = .1), ncol = 5)
output <- test_r_binding(4.0, 12, "hello", matrix_in=x)
expect_identical(dim(output$matrix_out), as.integer(c(100, 4)))
for (i in c(1, 2, 4)) {
    for (j in 1:100) {
        expect_true(output$matrix_out[j, i] == x[j, i])
    }
}
for (j in 1:100) {
    expect_true(output$matrix_out[j, 3] == 2 * x[j, 3])
}

# The data.frame we pass in, we should get back with the third dimension doubled
# and the fifth forgotten.
y <- matrix(c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), nrow = 3)
x <- data.frame(y)
output <- test_r_binding(4.0, 12, "hello", matrix_in=x)
expect_identical(dim(output$matrix_out), as.integer(c(3, 4)))
for (i in c(1, 2, 4)) {
    for (j in 1:3) {
        expect_true(output$matrix_out[j, i] == x[j, i])
    }
}
for (j in 1:3) {
    expect_true(output$matrix_out[j, 3] == 2 * x[j, 3])
}

# Same as TestMatrix but with an unsigned matrix.
x <- matrix(as.integer(rexp(500, rate = .1)), ncol = 5)
output <- test_r_binding(4.0, 12, "hello", umatrix_in=x)
expect_identical(dim(output$umatrix_out), as.integer(c(100, 4)))
for (i in c(1, 2, 4)) {
    for (j in 1:100) {
        expect_true(output$umatrix_out[j, i] == x[j, i])
    }
}

for (j in 1:100) {
    expect_true(output$umatrix_out[j, 3] == 2 * x[j, 3])
}

# Test a transposed matrix.
x <- matrix(rexp(500, rate = .1), ncol = 5)
output <- test_r_binding(4.0, 12, "hello", tmatrix_in=x, matrix_in=x)
## If the binding succeeds, the output double will be 10.
expect_true(output$double_out == 10.0)

# Test a column vector input parameter.
x <- matrix(rexp(100, rate = .1), nrow = 1)
output <- test_r_binding(4.0, 12, "hello", col_in=x)
expect_identical(dim(output$col_out), as.integer(c(1, 100)))
expect_identical(output$col_out, 2 * x)

# Test an unsigned column vector input parameter.
x <- matrix(as.integer(rexp(100, rate = .1)), nrow = 1) + 1
output <- test_r_binding(4.0, 12, "hello", ucol_in=x)
expect_identical(dim(output$ucol_out), as.integer(c(1, 100)))
expect_identical(output$ucol_out, 1 + x)

# Test a row vector input parameter.
x <- matrix(rexp(100, rate = .1), ncol = 1)
output <- test_r_binding(4.0, 12, "hello", row_in=x)
expect_identical(dim(output$row_out), as.integer(c(100, 1)))
expect_identical(output$row_out, 2 * x)

# Test an unsigned row vector input parameter.
x <- matrix(as.integer(rexp(100, rate = .1)), ncol = 1) + 1
output <- test_r_binding(4.0, 12, "hello", urow_in=x)
expect_identical(dim(output$urow_out), as.integer(c(100, 1)))
expect_identical(output$urow_out, 1 + x)

# Test that we can pass a matrix with all numeric features.
x <- matrix(rexp(500, rate = .1), ncol = 5)
output <- test_r_binding(4.0, 12, "hello", matrix_and_info_in=x)
expect_identical(dim(output$matrix_and_info_out), as.integer(c(100, 5)))
expect_identical(output$matrix_and_info_out, 2 * x)

# Test that we can pass a data.frame with all numeric features.
y <- matrix(rexp(500, rate = .1), ncol = 5)
x <- data.frame(y)
output <- test_r_binding(4.0, 12, "hello", matrix_and_info_in=x)
expect_identical(dim(output$matrix_and_info_out), as.integer(c(100, 5)))
for (i in 1:100) {
    for (j in 1:5) {
        expect_true(output$matrix_and_info_out[i, j] == 2 * x[i, j])
    }
}

# Test that we can pass a data.frame with numeric and categorical features.
y <- matrix(rexp(90, rate = .1), ncol = 9)
x <- data.frame(y, "e" = letters[1:10])
output <- test_r_binding(4.0, 12, "hello", matrix_and_info_in=x)
expect_identical(dim(output$matrix_and_info_out), as.integer(c(10, 10)))
for (i in 1:9) {
    for (j in 1:10) {
        expect_true(output$matrix_and_info_out[j, i] == 2 * x[j, i])
    }
}
for (j in 1:10) {
    expect_true(output$matrix_and_info_out[j, 10] == j)
}

# Test that we can pass a data.frame with numeric and categorical features.
y <- matrix(rexp(90, rate = .1), ncol = 9)
x <- data.frame(y)
x["e"] <- c(T, F, F, T, T, F, F, F, F, T)
output <- test_r_binding(4.0, 12, "hello", matrix_and_info_in=x)
expect_identical(dim(output$matrix_and_info_out), as.integer(c(10, 10)))
for (i in 1:9) {
    for (j in 1:10) {
        expect_true(output$matrix_and_info_out[j, i] == 2 * x[j, i])
    }
}
expect_identical(output$matrix_and_info_out[, 10], as.numeric(x[, "e"]))

# Test that we can pass a vector of ints and get back that same vector but with
# the last element removed.
x <- c(1, 2, 3, 4, 5)
output <- test_r_binding(4.0, 12, "hello", vector_in=x)
expect_identical(output$vector_out, c(1:4))

# Test that we can pass a vector of strings and get back that same vector but
# with the last element removed.
x <- letters[1:5]
output <- test_r_binding(4.0, 12, "hello", str_vector_in=x)
expect_identical(output$str_vector_out, letters[1:4])

# If we give data other than matrix/data.frame in matrix_in/matrix_and_info_in,
# we should get an error.
expect_error(test_r_binding(4.0, 12, "hello", matrix_in="wrong"))
expect_error(test_r_binding(4.0, 12, "hello", matrix_in=12))
expect_error(test_r_binding(4.0, 12, "hello", matrix_in=1e6))
expect_error(test_r_binding(4.0, 12, "hello", matrix_and_info_in="wrong"))
expect_error(test_r_binding(4.0, 12, "hello", matrix_and_info_in=12))
expect_error(test_r_binding(4.0, 12, "hello", matrix_and_info_in=1e6))

# If we pass labels that start from 0, we should get an error.
x <- vector(mode="integer", 10)
expect_error(test_r_binding(4.0, 12, "hello", urow_in=x))
y <- matrix(0L, 10, 1)
## this triggers a console message which we cannot suppress
expect_error(test_r_binding(4.0, 12, "hello", ucol_in=y))

# First create a GaussianKernel object, then send it back and make sure we get
# the right double value.
output1 <- test_r_binding(4.0, 12, "hello", build_model=TRUE)
output2 <- test_r_binding(4.0, 12, "hello", model_in=output1$model_out)
expect_true(output2$model_bw_out == 20)

# Test that we can serialize a model to disk and then use it again.
output1 <- test_r_binding(4.0, 12, "hello", build_model=TRUE)
tempbinfile <- tempfile(fileext=".bin")
Serialize(output1$model_out, tempbinfile)
new_model <- Unserialize(tempbinfile)
unlink(tempbinfile)
output2 <- test_r_binding(4.0, 12, "hello", model_in=new_model)
expect_true(output2$model_bw_out == 20)

# Make sure that the verbose argument does anything at all.
expect_stdout(test_r_binding(4.0, 12, "hello", build_model=TRUE, verbose=TRUE))

# Test that we get no output when verbose output is disabled.
# Make sure that global verbosity is turned off.
options(mlpack.verbose = FALSE)
expect_silent(test_r_binding(4.0, 12, "hello", build_model=TRUE))

# Test that we get no output when verbose output is explicitly disabled.
expect_silent(test_r_binding(4.0, 12, "hello", build_model=TRUE, verbose=FALSE))

# Make sure that the mlpack verbose global option does anything at all.
options(mlpack.verbose = TRUE)
expect_stdout(test_r_binding(4.0, 12, "hello", build_model=TRUE))

# Test that we get no output when the global verbose option is set to false.
options(mlpack.verbose = FALSE)
expect_silent(test_r_binding(4.0, 12, "hello", build_model=TRUE))

# Test that we can override the global verbose option.
options(mlpack.verbose = TRUE)
expect_silent(test_r_binding(4.0, 12, "hello", build_model=TRUE, verbose=FALSE))
options(mlpack.verbose = FALSE)

# Test that we get an S3 class
saved_seed <- .Random.seed
set.seed(1234)
x <- matrix(rnorm(10*5), ncol = 5)
res <- mlpack::knn(query = x, reference = x, k = 3)
expect_inherits(res, c("mlpack_knn", "mlpack_model_binding"))
.Random.seed <- saved_seed
