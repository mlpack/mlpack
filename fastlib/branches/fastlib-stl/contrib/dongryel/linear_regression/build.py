binrule(
  name = "vif_test",
  sources = ["vif_test.cc"],
  headers = ["givens_rotate_dev.h",
             "givens_rotate.h",
             "linear_regression_model_dev.h",
             "linear_regression_model.h",
             "linear_regression_result_dev.h",
             "linear_regression_result.h",
             "qr_least_squares_dev.h",
             "qr_least_squares.h"],
  deplibs = ["fastlib:fastlib_int"]
)
