# Test that when we run the binding correctly (with correct input parameters),
# we get the expected output.
test_that("TestRunBindingCorrectly", {
  output <- test_r_binding(4.0, 12, "hello",
                           flag1=TRUE)

  expect_true(output$double_out == 5.0)
  expect_true(output$int_out == 13)
  expect_true(output$string_out == "hello2")
})

# If we forget the mandatory flag, we should get wrong results.
test_that("TestRunBindingNoFlag", {
  output <- test_r_binding(4.0, 12, "hello")

  expect_true(output$double_out != 5.0)
  expect_true(output$int_out != 13)
  expect_true(output$string_out != "hello2")
})

# If we give the wrong string, we should get wrong results.
test_that("TestRunBindingWrongString", {
  output <- test_r_binding(4.0, 12, "goodbye",
                          flag1=TRUE)

  expect_true(output$string_out != "hello2")
})

# If we give the wrong int, we should get wrong results.
test_that("TestRunBindingWrongInt", {
  output <- test_r_binding(4.0, 15, "hello",
                           flag1=TRUE)

  expect_true(output$int_out != 13)
})

# If we give the wrong double, we should get wrong results.
test_that("TestRunBindingWrongDouble", {
  output <- test_r_binding(2.0, 12, "hello",
                           flag1=TRUE)

  expect_true(output$double_out != 5.0)
})

# If we give the second flag, this should fail.
test_that("TestRunBadFlag", {
  output <- test_r_binding(4.0, 12, "hello",
                          flag1=TRUE,
                          flag2=TRUE)

  expect_true(output$double_out != 5.0)
  expect_true(output$int_out != 13)
  expect_true(output$string_out != "hello2")
})

# The matrix we pass in, we should get back with the third dimension doubled and
# the fifth forgotten.
test_that("TestMatrix", {
  x <- matrix(rexp(500, rate = .1), ncol = 5)

  output <- test_r_binding(4.0, 12, "hello",
                           matrix_in=x)

  expect_identical(dim(output$matrix_out), as.integer(c(100, 4)))
  for (i in c(1, 2, 4)) {
    for (j in 1:100) {
      expect_true(output$matrix_out[j, i] == x[j, i])
    }
  }

  for (j in 1:100) {
    expect_true(output$matrix_out[j, 3] == 2 * x[j, 3])
  }
})

# The data.frame we pass in, we should get back with the third dimension doubled
# and the fifth forgotten.
test_that("TestDataFrame", {
  y <- matrix(c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), nrow = 3)
  x <- data.frame(y)

  output <- test_r_binding(4.0, 12, "hello",
                           matrix_in=x)

  expect_identical(dim(output$matrix_out), as.integer(c(3, 4)))
  for (i in c(1, 2, 4)) {
    for (j in 1:3) {
      expect_true(output$matrix_out[j, i] == x[j, i])
    }
  }

  for (j in 1:3) {
    expect_true(output$matrix_out[j, 3] == 2 * x[j, 3])
  }
})

# Same as TestMatrix but with an unsigned matrix.
test_that("TestUMatrix", {
  x <- matrix(as.integer(rexp(500, rate = .1)), ncol = 5)

  output <- test_r_binding(4.0, 12, "hello",
                           umatrix_in=x)

  expect_identical(dim(output$umatrix_out), as.integer(c(100, 4)))
  for (i in c(1, 2, 4)) {
    for (j in 1:100) {
      expect_true(output$umatrix_out[j, i] == x[j, i])
    }
  }

  for (j in 1:100) {
    expect_true(output$umatrix_out[j, 3] == 2 * x[j, 3])
  }
})

# Test a column vector input parameter.
test_that("TestCol", {
  x <- matrix(rexp(100, rate = .1), nrow = 1)

  output <- test_r_binding(4.0, 12, "hello",
                           col_in=x)

  expect_identical(dim(output$col_out), as.integer(c(1, 100)))
  expect_identical(output$col_out, 2 * x)
})

# Test an unsigned column vector input parameter.
test_that("TestUCol", {
  x <- matrix(as.integer(rexp(100, rate = .1)), nrow = 1)

  output <- test_r_binding(4.0, 12, "hello",
                           ucol_in=x)

  expect_identical(dim(output$ucol_out), as.integer(c(1, 100)))
  expect_identical(output$ucol_out, 1 + x)
})

# Test a row vector input parameter.
test_that("TestRow", {
  x <- matrix(rexp(100, rate = .1), ncol = 1)

  output <- test_r_binding(4.0, 12, "hello",
                           row_in=x)

  expect_identical(dim(output$row_out), as.integer(c(100, 1)))
  expect_identical(output$row_out, 2 * x)
})

# Test an unsigned row vector input parameter.
test_that("TestURow", {
  x <- matrix(as.integer(rexp(100, rate = .1)), ncol = 1)

  output <- test_r_binding(4.0, 12, "hello",
                           urow_in=x)

  expect_identical(dim(output$urow_out), as.integer(c(100, 1)))
  expect_identical(output$urow_out, 1 + x)
})

# Test that we can pass a matrix with all numeric features.
test_that("TestMatrixAndInfo", {
  x <- matrix(rexp(500, rate = .1), ncol = 5)

  output <- test_r_binding(4.0, 12, "hello",
                           matrix_and_info_in=x)

  expect_identical(dim(output$matrix_and_info_out), as.integer(c(100, 5)))
  expect_identical(output$matrix_and_info_out, 2 * x)
})

# Test that we can pass a data.frame with all numeric features.
test_that("TestDataFrameWithNoInfo", {
  y <- matrix(rexp(500, rate = .1), ncol = 5)
  x <- data.frame(y)

  output <- test_r_binding(4.0, 12, "hello",
                           matrix_and_info_in=x)

  expect_identical(dim(output$matrix_and_info_out), as.integer(c(100, 5)))

  for (i in 1:100) {
    for (j in 1:5) {
      expect_true(output$matrix_and_info_out[i, j] == 2 * x[i, j])
    }
  }
})

# Test that we can pass a data.frame with numeric and categorical features.
test_that("TestDataFrameWithInfo", {
  y <- matrix(rexp(90, rate = .1), ncol = 9)
  x <- data.frame(y, "e" = letters[1:10])

  output <- test_r_binding(4.0, 12, "hello",
                           matrix_and_info_in=x)

  expect_identical(dim(output$matrix_and_info_out), as.integer(c(10, 10)))

  for (i in 1:9) {
    for (j in 1:10) {
      expect_true(output$matrix_and_info_out[j, i] == 2 * x[j, i])
    }
  }

  for (j in 1:10) {
    expect_true(output$matrix_and_info_out[j, 10] == j)
  }
})

# Test that we can pass a data.frame with numeric and categorical features.
test_that("TestDataFrameWithLogicalInfo", {
  y <- matrix(rexp(90, rate = .1), ncol = 9)
  x <- data.frame(y)
  x["e"] <- c(T, F, F, T, T, F, F, F, F, T)

  output <- test_r_binding(4.0, 12, "hello",
                           matrix_and_info_in=x)

  expect_identical(dim(output$matrix_and_info_out), as.integer(c(10, 10)))

  for (i in 1:9) {
    for (j in 1:10) {
      expect_true(output$matrix_and_info_out[j, i] == 2 * x[j, i])
    }
  }
  expect_identical(output$matrix_and_info_out[, 10], as.numeric(x[, "e"]))
})

# Test that we can pass a vector of ints and get back that same vector but with
# the last element removed.
test_that("TestIntVector", {
  x <- c(1, 2, 3, 4, 5)

  output <- test_r_binding(4.0, 12, "hello",
                           vector_in=x)

  expect_identical(output$vector_out, c(1:4))
})

# Test that we can pass a vector of strings and get back that same vector but
# with the last element removed.
test_that("TestStringVector", {
  x <- letters[1:5]

  output <- test_r_binding(4.0, 12, "hello",
                           str_vector_in=x)

  expect_identical(output$str_vector_out, letters[1:4])
})

# If we give data other than matrix/data.frame in matrix_in/matrix_and_info_in,
# we should get an error.
test_that("TestNotMatrix", {
  expect_error(test_r_binding(4.0, 12, "hello",
                              matrix_in="wrong"))

  expect_error(test_r_binding(4.0, 12, "hello",
                              matrix_in=12))

  expect_error(test_r_binding(4.0, 12, "hello",
                              matrix_in=1e6))

  expect_error(test_r_binding(4.0, 12, "hello",
                              matrix_and_info_in="wrong"))

  expect_error(test_r_binding(4.0, 12, "hello",
                              matrix_and_info_in=12))

  expect_error(test_r_binding(4.0, 12, "hello",
                              matrix_and_info_in=1e6))
})

# First create a GaussianKernel object, then send it back and make sure we get
# the right double value.
test_that("TestModel", {
  output1 <- test_r_binding(4.0, 12, "hello",
                            build_model=TRUE)

  output2 <- test_r_binding(4.0, 12, "hello",
                            model_in=output1$model_out)

  expect_true(output2$model_bw_out == 20)
})

# Test that we can serialize a model to disk and then use it again.
test_that("TestSerialization", {
  output1 <- test_r_binding(4.0, 12, "hello",
                            build_model=TRUE)

  Serialize(output1$model_out, "model.bin")

  new_model <- Unserialize("model.bin")
  unlink("model.bin")

  output2 <- test_r_binding(4.0, 12, "hello", model_in=new_model)

  expect_true(output2$model_bw_out == 20)
})
