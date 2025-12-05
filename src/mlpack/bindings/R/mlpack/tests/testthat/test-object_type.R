test_that("Test for mlpack S3 class ", {

    set.seed(1234)
    x <- matrix(rnorm(10*5), ncol = 5)

    res <- mlpack::knn(query = x, reference = x, k = 3)

    expect_s3_class(res, c("mlpack_knn", "mlpack_model_binding"))

})
